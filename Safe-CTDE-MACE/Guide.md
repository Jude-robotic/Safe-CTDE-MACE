# Safe-CTDE-MACE 中文指南

## 1. 当前工程状态

Safe-CTDE-MACE 是一个面向多无人机三维覆盖探索的仿真算法框架。工程目标是在体素化三维场景中，让多架无人机在安全约束、局部感知和有限通信条件下协同探索未知空间，并尽快达到目标覆盖率。

当前仓库保留两条可运行路线：

1. **Shared DQN + A\***：已验证基线，用于确认环境、训练、评估和可视化链路正常。
2. **QMIX + EGO-style planner**：当前重点路线，采用集中式训练、分布式执行，并在运动层使用连续轨迹规划。

当前正式大场景配置是 `safe_ctde_mace/configs/qmix_ego_large.yaml`：

- 地图：`20 x 20 x 8`
- 无人机数量：`3`
- 感知半径：`sensor_range=3.5`
- 通信半径：`comm_range=20.0`
- 目标覆盖率：`target_coverage_ratio=0.90`
- 最大步数：`max_steps=100`
- planner：`ego`
- 并行环境：`num_envs=4`
- 纯物理通信：`global_sync_interval=0`

最新 4000 episode 长训已经在该配置下跑通：

```powershell
python -m safe_ctde_mace.scripts.train_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --episodes 4000 `
  --device cuda `
  --num-envs 4 `
  --artifact-dir artifacts/qmix_large_train `
  --output checkpoints/qmix_large_final.pt
```

训练末尾结果：

```text
episodes=4000 reward=1862.489 coverage=0.900 length=100 success=True
eval_coverage=0.902 eval_success_rate=1.000
artifacts=artifacts\qmix_large_train
```

独立评估结果：

```powershell
python -m safe_ctde_mace.scripts.evaluate_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --checkpoint checkpoints/qmix_large_final.pt `
  --device cuda `
  --episodes 10 `
  --artifact-dir artifacts/qmix_large_eval
```

```text
episodes=10 coverage_mean=0.902 success_rate=1.000 episode_length_mean=81.4
last_trace plateau_step=80 first_hover_step=20 first_collision_step=26
first_planner_failure_step=None max_zero_gain_streak=2 planner_failures=0
physical_links_mean=2.91 effective_links_mean=2.91
```

这说明当前大场景已经从“不可稳定求解”推进到“可稳定成功”。但 trace 和 `evaluation_history.csv` 也显示一个明确缺陷：`repeated_coverage_ratio` 往往仍在 `0.95-0.99`，后期仍存在较严重的重复探索。后续优化重点应从“能否成功”转为“如何更快训练、如何降低重复覆盖、如何让多机分工更接近最优”。

## 2. 工程架构

工程可以分成 6 层。

### 2.1 环境层

- `envs/voxel_world.py`：真实三维体素世界、静态障碍物和随机障碍物生成。
- `envs/multi_uav_env.py`：环境主循环、奖励、终止条件、全局状态、planner 调度和 trace 诊断。

`MultiUAVCoverageEnv.step(actions)` 的核心流程：

1. 解析每架无人机的高层动作；
2. 生成并预订 frontier 目标；
3. 执行 late-stage reassignment 和目标去冲突；
4. 通过 `SafetyShield` 筛选安全目标；
5. 使用 A* 或 EGO-style planner 生成路径/轨迹；
6. 执行一步轨迹并更新无人机状态；
7. 更新局部地图、融合邻居信息并刷新覆盖统计；
8. 计算奖励、终止条件和诊断信息。

当前碰撞采用软约束语义：轨迹命中障碍物或无人机间距过近时，环境记录碰撞并按 `w_obstacle` / `w_collision` 重罚，同时把无人机回退到上一安全状态、速度清零，而不是永久失活。

### 2.2 无人机与地图层

- `agents/uav_agent.py`：无人机位置、速度、目标、轨迹和局部地图。
- `mapping/coverage_map.py`：体素状态、感知更新、覆盖统计和预订区域。
- `mapping/frontier_detector.py`：frontier 检测、聚类、候选目标生成和候选特征计算。
- `mapping/voxel_map.py`：体素状态定义和可通行状态集合。

当前候选 frontier 特征使用 `feature_schema_version=2`。每个候选包含基础特征、邻居分工 margin、空间上下文和局部未覆盖密度：

- `distance_to_uav`
- `expected_information_gain`
- `obstacle_risk`
- `reserved_penalty`
- `neighbor_overlap`
- `path_cost`
- `assignment_margins`
- `grid_quadrant`
- `layer_height`
- `uncovered_density_near_candidate`

这些字段通过 `CandidateFeatureLayout` 动态访问，避免三机大场景下因为 `max_neighbors` 改变而读错特征位置。

### 2.3 协同与通信层

- `communication/comm_graph.py`：根据通信半径构建物理邻接关系。
- `communication/map_fusion.py`：融合邻居地图、覆盖状态、障碍物和预订信息。

当前正式大场景使用纯物理通信，`global_sync_interval=0`。trace 会同时记录：

- `physical_communication_links`
- `effective_communication_links`
- `global_sync_applied`

在 `comm_range=20.0` 的当前配置下，三机大多数时间保持完整物理连通。`MapFusion` 已修正为不允许邻居的 `FREE` 覆盖本地 `COVERED`，避免覆盖信息被错误擦除。

### 2.4 安全与规划层

- `planning/safety_shield.py`：筛掉不安全、不可达或与邻居冲突的目标，并在必要时回退到备选目标或悬停。
- `planning/astar_3d.py`：离散三维 A* planner。
- `planning/ego_planner.py`：Python 版 EGO-style 连续轨迹优化器。
- `planning/trajectory_tracker.py`：执行离散一步或连续轨迹采样。

当前 EGO-style planner 使用 A* 种子路径、控制点优化、平滑项和障碍物距离项。大场景配置中已将 `ego_optimize_iterations` 提升到 `80`，并增加轨迹碰撞采样以减少薄障碍漏检。

### 2.5 强化学习层

- `marl/shared_dqn.py`：共享 DQN 基线。
- `marl/qmix.py`：QMIX agent、target network、mixer、双 Q 估计和 checkpoint 结构校验。
- `marl/networks.py`：观测展平和 Q 网络输入维度计算。
- `marl/replay_buffer.py`：单机和联合 replay buffer。
- `marl/trainer.py`：DQN 与 QMIX 的训练、评估、trace 和 checkpoint 流程。
- `marl/parallel_rollout.py`：QMIX 多进程并行环境采样。

QMIX 执行阶段只使用局部 observation；集中训练阶段额外使用紧凑全局状态。当前全局状态已加入 coarse residual coverage distribution，包括 XY 四象限残余覆盖和高度层残余覆盖，使 mixer 能看到“哪里还没扫完”。

### 2.6 工具与脚本层

- `scripts/train.py` / `scripts/evaluate.py`：Shared DQN 训练和评估。
- `scripts/train_qmix.py` / `scripts/evaluate_qmix.py`：QMIX 训练和评估。
- `scripts/demo_random_policy.py`：随机策略演示。
- `scripts/demo_heuristic_policy.py`：启发式 frontier 策略演示。
- `scripts/compare_planners.py`：A* 与 EGO-style planner 对比。
- `scripts/benchmark_qmix.py`：QMIX 吞吐 benchmark。
- `utils/reporting.py`：CSV、JSON trace 和失败诊断导出。
- `utils/visualization.py`：训练曲线、诊断图、三维图和 GIF 回放。

## 3. 环境准备与健康检查

```powershell
conda activate Safe-CTDE-MACE
pip install -r requirement.txt
```

如需 CUDA 训练：

```powershell
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m safe_ctde_mace.scripts.check_cuda --device cuda
```

先运行单元测试：

```powershell
python -m pytest -q
```

重要配置文件：

- `safe_ctde_mace/configs/verified_baseline.yaml`：Shared DQN + A* 已验证基线。
- `safe_ctde_mace/configs/qmix_ego.yaml`：小场景 QMIX + EGO-style 回归配置。
- `safe_ctde_mace/configs/qmix_ego_large.yaml`：三机大场景正式配置。
- `safe_ctde_mace/configs/qmix_ego_large_no_obstacles.yaml`：无障碍 ablation。
- `safe_ctde_mace/configs/qmix_ego_large_full_sync.yaml`：全局同步 ablation。
- `safe_ctde_mace/configs/qmix_ego_large_astar.yaml`：A* planner ablation。

## 4. 推荐实验顺序

### 步骤 1：pytest

```powershell
python -m pytest -q
```

目标：确认配置、地图、环境、MARL、可视化和 reporting 单元测试通过。

### 步骤 2：随机策略演示

```powershell
python -m safe_ctde_mace.scripts.demo_random_policy `
  --config safe_ctde_mace/configs/verified_baseline.yaml `
  --steps 50 `
  --save-dir artifacts/random_demo
```

目标：确认环境能跑通，三维可视化正常生成。

### 步骤 3：启发式策略演示

```powershell
python -m safe_ctde_mace.scripts.demo_heuristic_policy `
  --config safe_ctde_mace/configs/verified_baseline.yaml `
  --steps 50 `
  --save-dir artifacts/heuristic_demo
```

目标：确认 frontier 检测、候选评分、planner 和覆盖更新正常。

### 步骤 4：复现 Shared DQN + A*

```powershell
python -m safe_ctde_mace.scripts.train `
  --config safe_ctde_mace/configs/verified_baseline.yaml `
  --episodes 30 `
  --artifact-dir artifacts/verified_train `
  --output checkpoints/shared_dqn_final.pt
```

```powershell
python -m safe_ctde_mace.scripts.evaluate `
  --config safe_ctde_mace/configs/verified_baseline.yaml `
  --checkpoint checkpoints/shared_dqn_final.pt `
  --episodes 5 `
  --artifact-dir artifacts/verified_eval
```

目标：确认已验证基线能达到目标覆盖率。

### 步骤 5：比较 A* 与 EGO-style planner

```powershell
python -m safe_ctde_mace.scripts.compare_planners `
  --config safe_ctde_mace/configs/verified_baseline.yaml `
  --steps 20 `
  --artifact-dir artifacts/verified_planner_comparison
```

重点查看：

- `planner_comparison.csv`
- `planner_comparison.png`
- `astar_episode.gif`
- `ego_episode.gif`

目标：对比覆盖率、平均加速度、最大加速度、smoothness cost 和碰撞情况。

### 步骤 6：小场景 QMIX + EGO-style

```powershell
python -m safe_ctde_mace.scripts.train_qmix `
  --config safe_ctde_mace/configs/qmix_ego.yaml `
  --episodes 30 `
  --artifact-dir artifacts/qmix_train `
  --output checkpoints/qmix_final.pt
```

```powershell
python -m safe_ctde_mace.scripts.evaluate_qmix `
  --config safe_ctde_mace/configs/qmix_ego.yaml `
  --checkpoint checkpoints/qmix_final.pt `
  --episodes 5 `
  --artifact-dir artifacts/qmix_eval
```

目标：确认 QMIX 的局部观测、全局状态、联合 replay、mixer 和 EGO planner 链路正常。

### 步骤 7：大场景 QMIX 长训

开发期不建议直接从 4000 episodes 开始。推荐先按阶梯验证：

```powershell
python -m safe_ctde_mace.scripts.train_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --episodes 500 `
  --device cuda `
  --num-envs 4 `
  --artifact-dir artifacts/qmix_large_train `
  --output checkpoints/qmix_large_final.pt
```

```powershell
python -m safe_ctde_mace.scripts.evaluate_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --checkpoint checkpoints/qmix_large_final.pt `
  --device cuda `
  --episodes 10 `
  --artifact-dir artifacts/qmix_large_eval
```

正式复现实验可使用 4000 episodes。当前已验证 4000 episodes 能达到 `eval_coverage=0.902`、`eval_success_rate=1.000`，但耗时约 10 小时，因此它更适合作为最终确认实验，而不是日常调参入口。

### 步骤 8：吞吐 benchmark

```powershell
python -m safe_ctde_mace.scripts.benchmark_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --episodes 4 `
  --device cuda `
  --num-envs 4 `
  --append-csv artifacts/qmix_benchmark/benchmark_matrix.csv
```

建议矩阵：

- `num_envs=4/8/12/16`
- `ego_optimize_iterations=30/50/80`
- 开发期评估 `seed_count=3`
- 确认期评估 `seed_count=10`

目标：找到本机上 CPU 环境采样、EGO planner 和 GPU 网络更新之间的真实瓶颈。

## 5. 如何阅读产物和 trace

常用产物目录：

- `artifacts/verified_train`
- `artifacts/verified_eval`
- `artifacts/qmix_train`
- `artifacts/qmix_eval`
- `artifacts/qmix_large_train`
- `artifacts/qmix_large_eval`
- `artifacts/qmix_benchmark`
- `checkpoints/`

常用文件：

- `train_history.csv`：训练 episode 摘要。
- `evaluation_history.csv`：评估 episode 摘要。
- `last_train_trace.json`：最后一次训练 episode 的逐步诊断。
- `evaluation_trace.json` / `last_evaluation_trace.json`：评估 episode 的逐步诊断。
- `*_diagnostics.png`：覆盖曲线、重复覆盖、通信、hover、planner 状态等图。
- `*_replay.gif`：探索过程回放。

如果 `success=False`，按下面顺序排查：

1. 看 `termination_reason` 是 `max_steps`、`all_failed` 还是 `coverage_target`。
2. 看 `coverage_curve` 是否过早平台化。
3. 看 `frontier_counts` 是否仍很高，同时 `team_new_coverage` 接近 0。
4. 看 `hover_reasons` 中是 `neighbor_conflict`、`planner_unavailable` 还是 `no_valid_candidate`。
5. 看 `planner_failure_counts` 是否为 0。如果为 0，优先改任务分配和候选选择，而不是继续调 EGO。
6. 看 `repeated_coverage_ratio` 是否长期高于 `0.95`。如果是，说明当前主要瓶颈是多机重复探索。
7. 看 `physical_communication_links` 与 `effective_communication_links`。当前正式配置应保持较高连通，不应依赖全局同步。

## 6. 当前关键修复合并摘要

大场景能够成功，主要依赖以下已经合并的结构性改动：

- 将 `sensor_range` 从 `2.5` 提高到 `3.5`，并将目标从 `0.95` 调整到 `0.90`，使问题从物理上可解。
- 将 `max_steps` 调整为 `100`，匹配当前覆盖目标和感知半径。
- 修复 `MapFusion` 中邻居 `FREE` 覆盖本地 `COVERED` 的问题。
- 增大 `comm_range` 到 `20.0`，让三机在大场景中保持物理连通。
- 加入 reward 归一化、milestone bonus 和 `w_progress`。
- 加入 `candidate_min_separation`，使候选 frontier 保持基本空间分散。
- 加入动态 `CandidateFeatureLayout` 和 `feature_schema_version=2`，保证三机候选特征读取正确。
- 加入 residual coverage global state，让 QMIX mixer 看到粗粒度未覆盖分布。
- 加入 late-stage reassignment，在高覆盖或低增益阶段重新分配候选目标。
- 加入 action-conditioned diversity bonus，使奖励跟随实际选择的候选。
- 加入多 seed 评估和失败诊断导出。
- 加入 QMIX 并行采样 `num_envs`，支持多环境同步 rollout。

## 7. 计划代理给出的下一阶段改进方案

### 7.1 当前诊断

当前大场景不再是“跑不成功”的问题。4000 episodes 训练和 10 episode 独立评估均已达到 `success_rate=1.000`。真正的剩余缺陷是：

- 训练成本高：4000 episodes 约 10 小时，不适合作为常规调参循环。
- 重复覆盖高：评估中 `repeated_coverage_ratio` 经常在 `0.95-0.99`。
- 后期新增覆盖低：trace 中后期 `team_new_coverage` 经常接近 0。
- planner 失败不是主因：最新评估 `planner_failures=0`。
- 通信不是主因：`physical_links_mean` 和 `effective_links_mean` 接近满连通。

因此，下一阶段不应继续盲目加长训练，而应先提高实验效率，再针对协同分工和重复覆盖做结构性改进。

### 7.2 P0：把长训拆成可停止的分阶段实验

将一次 4000 episodes 长训拆成：

```text
30 -> 100 -> 300 -> 500 -> 1000 -> 4000
```

每个阶段都做 10 seed 评估，并记录：

- `success_rate`
- `coverage_mean`
- `episode_length_mean`
- `repeated_coverage_ratio_mean`
- `zero_gain_streak`
- `late_reassignment_steps`
- `planner_failure_total`
- `wall_clock_per_episode`

建议提前停止规则：

- 若连续两次评估 `success_rate >= 0.8` 且 `coverage_mean >= 0.90`，停止加长训练。
- 若 100-300 episodes 后 `repeated_coverage_ratio > 0.95` 且无下降趋势，暂停长训，优先改 reward 或分工。
- 若 `planner_failure_total=0`，不要继续优先调 EGO 参数。

预期收益：把一次 10 小时试错拆成多个短决策点，避免在重复覆盖瓶颈上浪费长训时间。

### 7.3 P1：训练加速优先做环境和 planner 侧优化

优先做吞吐 profile，区分耗时来自：

- QMIX 网络前向/反向；
- 环境 step；
- frontier 检测与 BFS 可达性；
- A* 种子路径；
- EGO 控制点优化；
- 地图融合与 trace 写入；
- 评估和可视化。

推荐实验：

```text
num_envs: 4 / 8 / 12 / 16
ego_optimize_iterations: 30 / 50 / 80
eval_seed_count: 开发期 3，确认期 10
```

可尝试的训练参数：

- `batch_size=128`
- `replay_capacity=16000`
- `target_update_interval=200`
- 开发期减少 GIF 和三维图生成，只保留 CSV 和 JSON trace。

如果 GPU 利用率低，瓶颈大概率在 Python 环境、frontier 检测或 EGO planner，而不是网络容量。此时继续加大网络通常不能明显加速收敛。

风险：降低 EGO 迭代可能改变策略分布，所以最终确认仍需用正式 `ego_optimize_iterations=80` 复评。

### 7.4 P2：用奖励直接抑制重复覆盖

当前 `w_repeat` 已存在，但高重复覆盖说明它还不足以驱动最优分工。建议加入阶段化重复惩罚：

- 当 `coverage_ratio < 0.50` 时，保留温和重复惩罚，允许必要回访。
- 当 `coverage_ratio >= 0.50` 时，提高 `w_repeat` 和 `w_overlap`。
- 当 `zero_gain_streak >= 3` 时，额外惩罚连续低收益动作。
- 对同一步多机感知 footprint 重叠加入团队级 overlap penalty。
- 对选择不同象限、不同高度层、不同未覆盖密集区的动作给予 diversity bonus。

预期收益：直接降低 `repeated_coverage_ratio`，提升后期覆盖效率。

风险：重复惩罚过强会让无人机绕远路或离开必要通信链路，因此建议使用随覆盖率变化的动态权重。

### 7.5 P3：增强观测，让 QMIX 看见“谁负责哪里”

建议从低维、稳定的协同特征开始：

- 在 `candidate_features` 中加入邻居当前目标到候选点的距离。
- 加入候选附近最近若干步新增覆盖收益。
- 加入候选与邻居预测路径 footprint 的重叠率。
- 在 `neighbor_states` 中加入邻居 `zero_gain_streak` 和最近新增覆盖。
- 将全局 residual coverage 从当前粗粒度扩展到 `4 x 4 x 2` 或 `5 x 5 x 2`。

预期收益：提高 mixer 的信用分配能力，让网络更容易学出区域分工。

风险：改 observation 或 global state 会导致旧 checkpoint 不兼容，应继续维护 schema version。

### 7.6 P4：加入显式分工层，减少 RL 独自承担组合分配

如果 P2/P3 后重复覆盖仍高，建议加入轻量任务分配：

- 先做软 Voronoi responsibility region：跨区候选加 cost，但不硬禁止。
- 再做 frontier cluster penalty：相同 cluster 中最多允许一个 UAV 获得高优先级。
- 最后尝试 Hungarian 或 auction assignment：先把 UAV 分配到不同 frontier cluster，再让 QMIX 在各自 cluster 内选候选点。
- late-stage 超过 `coverage_ratio=0.70` 后切换到残余团块清扫模式，优先分配孤立未覆盖区域。

预期收益：显著减少多机抢同一片 frontier，使 RL 更专注于局部候选选择。

风险：显式分工会把一部分协同能力从 RL 转移到规则层，论文中应明确它属于 hierarchical policy 或 safety/coordination prior。

### 7.7 P5：课程学习降低所需 episode 数

建议课程：

1. 小地图、少障碍、低覆盖目标；
2. 当前地图尺寸、少障碍；
3. 当前地图尺寸、当前障碍、`target=0.85`；
4. 当前正式配置：`20 x 20 x 8`、3 UAV、`target=0.90`；
5. 多 seed 泛化评估。

目标是减少从零开始探索所需 episode 数，并让策略更快学会“早期扩张、后期分工收尾”的节奏。

### 7.8 最推荐的近期路线

1. 先跑 `num_envs` 与 `ego_optimize_iterations` 吞吐矩阵，拿到本机最省时配置。
2. 将默认大场景长训从 4000 改为阶段式 500/1000，4000 只作最终复现。
3. 加动态重复覆盖惩罚和 footprint overlap penalty。
4. 加 coarse residual coverage 细化版和 teammate intent。
5. 若重复覆盖仍高，再加入软分区或 frontier cluster 分配。
6. 最后用 1000/4000 episodes 做正式确认。

## 8. 后续研究方向

### 8.1 强化学习

- 将 MLP observation encoder 升级为 `3D CNN + MLP`。
- 尝试 prioritized replay、n-step return 和 dueling head。
- 在 QMIX 后对比 `DNQMIX`、`MADDPG`、`MASAC`。
- 把 frontier 特征排序从手工评分逐步过渡到可学习表示。

### 8.2 安全与动力学

- 在 `SafetyShield.cbf_qp` 中接入真正的 CBF-QP。
- 将 Python EGO-style 近似推进到更完整的 B-spline 优化。
- 加入速度、加速度、jerk 的显式约束投影。
- 增加动态障碍和更细粒度的时空避碰。

### 8.3 协同与通信

- 从确定性通信扩展到概率通信。
- 引入链路丢包、带宽限制和事件触发通信。
- 让 reservation 同时编码空间和预计到达时间。
- 尝试图神经网络编码邻居状态。

### 8.4 评估与工程化

- 建立多地图、多随机种子的 benchmark。
- 统一输出 success rate、覆盖率、完成时间、重复覆盖率、通信代价和平滑性。
- 增加 TensorBoard 或 WandB。
- 增加配置 schema 校验和 CI smoke test。
