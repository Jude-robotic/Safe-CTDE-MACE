# Safe-CTDE-MACE: 基于多智能体强化学习的安全协作多无人机三维覆盖探索

## 1. 项目概述

Safe-CTDE-MACE 是一个面向多无人机三维覆盖探索的研究框架。该框架以体素化三维环境为基础，将多智能体强化学习、集中式训练分布式执行、协作通信、frontier 探索、安全屏蔽和连续轨迹规划整合为统一系统，使多架无人机能够在未知空间中自主协同探索。

### 1.1 问题陈述

本框架研究的是多无人机自主覆盖探索问题：在包含静态障碍物的有界三维空间中，引导多架无人机尽可能快地发现并覆盖未知体素，同时满足有限感知、物理通信、障碍物避让和机间安全距离约束。优化目标不只是最大化最终覆盖率，还包括降低重复探索、缩短完成时间、减少碰撞风险，并保持轨迹平滑。

### 1.2 研究动机

协作式多无人机覆盖探索是灾害响应、基础设施巡检、搜救行动和环境测绘等任务的核心能力。该问题天然具有部分可观测性、强安全约束和多智能体信用分配难题。单架无人机只能观察局部球形区域，多架无人机之间又需要通过有限通信共享地图和意图，因此本框架采用多智能体强化学习来学习高层探索策略，并使用安全与规划层约束底层运动。

## 2. 系统架构

Safe-CTDE-MACE 由六个层级组成：环境层、无人机与地图层、协同通信层、安全规划层、强化学习层和工具脚本层。

### 2.1 环境层

`VoxelWorld` 将三维工作空间离散化为体素网格。体素状态包括 `UNKNOWN`、`FREE`、`OBSTACLE`、`COVERED` 和 `RESERVED`。场景既支持手动障碍箱体，也支持随机障碍箱体，从而兼顾可复现实验和多 seed 泛化测试。

`MultiUAVCoverageEnv` 是主环境。它管理无人机状态、局部感知、地图融合、frontier 候选、动作执行、奖励计算、终止条件和诊断记录。每一步环境交互会依次执行动作解析、候选去冲突、安全屏蔽、轨迹规划、状态推进、覆盖更新和 trace 写入。

当前碰撞处理采用软约束语义。当无人机命中障碍物或与其他无人机过近时，环境记录碰撞并施加重罚，但不会永久失活无人机，而是将其回退到上一安全状态并清零速度。这一设计保留了安全反馈，又避免训练早期因一次错误动作导致整回合失效。

### 2.2 无人机与地图层

`UAVAgent` 封装单架无人机的位置、速度、当前目标、轨迹和局部体素地图。无人机通过以自身位置为中心的球形感知更新局部地图，并把观察到的自由空间、障碍物和覆盖状态写入全局统计。

`CoverageMap` 负责维护覆盖状态和重复覆盖统计。当前奖励中与覆盖有关的项按“本步观测比例”归一化，而不是直接使用原始体素计数，从而避免三机大场景中奖励尺度因重复观测快速失衡。

`FrontierDetector` 负责从已知自由区域与未知区域之间提取 frontier，并生成固定数量的候选目标。当前候选特征采用 `feature_schema_version=2`，包括距离、信息增益、障碍风险、预订惩罚、邻居重叠、路径代价、邻居分工 margin、XY 象限、高度层和候选附近未覆盖密度。`CandidateFeatureLayout` 以 `max_neighbors` 为参数动态定位字段，解决了三机大场景中固定索引读取错误的问题。

候选生成还加入 `candidate_min_separation`，使排名靠前的候选保持基本空间分散，减少多个智能体被同一局部 frontier 吸引。

### 2.3 协同通信层

`CommGraph` 根据通信半径构建动态物理邻接图。`MapFusion` 在邻居之间融合障碍物、覆盖状态和预订信息。当前正式大场景设置 `comm_range=20.0` 且 `global_sync_interval=0`，即评估过程中不依赖全局同步，而是使用物理通信链路。

为避免覆盖信息被错误擦除，`MapFusion` 已修复为不允许邻居的 `FREE` 状态覆盖本地 `COVERED` 状态。trace 同时记录 `physical_communication_links`、`effective_communication_links` 和 `global_sync_applied`，用于区分真实物理连通与调试性全局同步。

### 2.4 安全与规划层

安全与规划层采用分层结构。高层强化学习策略只选择 frontier 候选目标，底层由安全屏蔽和轨迹规划负责可达性与避障。

`SafetyShield` 在目标交给 planner 前检查其可通行性、可达性、障碍距离和机间安全距离。若当前目标不可用，它会尝试候选备选目标，并在必要时回退为悬停。`cbf_qp` 方法作为控制屏障函数二次规划接口保留，可用于后续形式化安全约束扩展。

`AStar3D` 是离散基线 planner，在体素网格上运行 A* 搜索。`EGOStylePlanner` 是当前重点使用的连续轨迹 planner，它先用 A* 生成种子路径，再用控制点优化平衡轨迹平滑性和障碍物避让。`TrajectoryTracker` 负责执行离散路径或连续轨迹采样，并把规划结果转化为环境步进。

当前大场景中 EGO-style planner 使用 `ego_optimize_iterations=80`、`ego_obstacle_weight=1.2` 和更密集的轨迹碰撞采样，以降低薄障碍漏检概率。

### 2.5 强化学习层

工程支持两类多智能体强化学习方法。

`SharedDQN` 是已验证基线。所有无人机共享同一个 Q 网络，并在各自局部观测上选择 frontier 候选动作。该路线配合 A* planner，用于验证环境、训练、评估和可视化链路。

`QMIXAgent` 是当前正式大场景路线。执行阶段，每架无人机只使用自己的局部观测选择动作；训练阶段，mixer 使用中心化全局状态学习联合 Q 值分解。QMIX 实现包含目标网络、双 Q 估计、动作掩码、梯度裁剪和 checkpoint 元数据校验。checkpoint 会记录 `obs_dim`、`state_dim`、`num_agents`、`num_actions` 与 `feature_schema_version`，避免错误加载不兼容模型。

`QMIXTrainer` 支持串行训练和多进程并行采样。当前大场景使用 `num_envs=4`，由 `ParallelRolloutManager` 同步多个环境 worker，再统一写入联合 replay buffer 并训练 QMIX。

### 2.6 工具脚本层

训练入口包括 `train.py`、`evaluate.py`、`train_qmix.py` 和 `evaluate_qmix.py`。演示和诊断入口包括 `demo_random_policy.py`、`demo_heuristic_policy.py`、`compare_planners.py`、`benchmark_qmix.py` 和 `check_cuda.py`。

`utils/reporting.py` 输出 episode 摘要、逐步 trace、失败摘要和 step diagnostics。`utils/visualization.py` 输出训练曲线、评估曲线、三维轨迹图、诊断图和 GIF 回放。

## 3. 观测、状态与动作空间

每架无人机接收结构化局部观测，包括局部体素 patch、自身状态、邻居状态、当前覆盖率、frontier 候选特征和动作掩码。

| 组成部分 | 含义 |
|---|---|
| `local_voxel_map` | 以无人机为中心的局部体素 patch |
| `self_state` | 位置、速度和当前目标等自身状态 |
| `neighbor_states` | 邻居无人机的位置、速度和目标 |
| `coverage_ratio` | 当前全局覆盖率 |
| `candidate_features` | 固定数量 frontier 候选的特征矩阵 |
| `action_mask` | 当前有效候选动作掩码 |

QMIX 的中心化全局状态在局部状态之外加入 coarse residual coverage distribution。当前实现已编码 XY 四象限残余覆盖率和逐高度层残余覆盖率，使 mixer 能感知全局未覆盖空间分布。

动作空间是固定大小的离散 frontier 候选集合。每个动作对应选择一个候选目标，实际执行前还要经过目标去冲突、late-stage reassignment 和 SafetyShield 检查。

## 4. 奖励结构与协作机制

奖励函数由覆盖收益、信息增益、重复覆盖惩罚、重叠惩罚、碰撞惩罚、时间惩罚、能耗惩罚、预订惩罚和完成奖励组成。

当前大场景奖励和协作机制包括：

- `w_new` 与 `w_info` 使用归一化观测比例，保留新增覆盖学习信号。
- `w_repeat` 惩罚重复覆盖，抑制反复扫过已覆盖区域。
- `w_overlap` 惩罚无人机间过近或感知区域重叠。
- `w_collision` / `w_obstacle` 对机间碰撞和障碍碰撞施加重罚。
- `w_progress` 根据当前覆盖率相对目标覆盖率提供密集进度奖励。
- 覆盖率达到中间里程碑时给予额外 milestone bonus。
- 当高覆盖阶段出现低增益停滞时，late-stage reassignment 会重新分配候选目标。
- diversity bonus 根据实际选择候选的未覆盖密度、空间象限和高度层奖励更有区分度的探索方向。

这些机制使大场景从早期的失败配置推进到稳定成功配置。然而最新评估仍显示重复覆盖比例较高，说明当前奖励已经足以达成目标覆盖率，但尚未使多机分工达到较优效率。

## 5. 训练与评估工作流

推荐工作流先从单元测试和基线复现开始，再进入 QMIX 大场景。

1. 运行 `python -m pytest -q`。
2. 运行随机策略和启发式策略 demo，确认环境与可视化正常。
3. 训练并评估 `verified_baseline.yaml` 的 Shared DQN + A*。
4. 运行 `compare_planners` 比较 A* 与 EGO-style 的覆盖率和平滑性。
5. 训练并评估 `qmix_ego.yaml` 小场景 QMIX。
6. 进入 `qmix_ego_large.yaml` 三机大场景。

当前正式大场景训练命令为：

```powershell
python -m safe_ctde_mace.scripts.train_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --episodes 4000 `
  --device cuda `
  --num-envs 4 `
  --artifact-dir artifacts/qmix_large_train `
  --output checkpoints/qmix_large_final.pt
```

独立评估命令为：

```powershell
python -m safe_ctde_mace.scripts.evaluate_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --checkpoint checkpoints/qmix_large_final.pt `
  --device cuda `
  --episodes 10 `
  --artifact-dir artifacts/qmix_large_eval
```

训练和评估默认采用多 seed 口径。`train_qmix.py` 默认使用 `--eval-seed-count 10`，`evaluate_qmix.py` 默认使用 `--seed-count 10`，从配置中的 `seed` 开始连续评估多个场景。

## 6. 最新实验结果

当前大场景配置为 `20 x 20 x 8`、3 架无人机、`sensor_range=3.5`、`target_coverage_ratio=0.90`、`max_steps=100`。该配置是在原始高难配置基础上调整而来。早期 `sensor_range=2.5`、`target=0.95` 的组合在 120 步内对三机协作要求过高，即使启发式策略也难以达到目标覆盖率，因此它更接近问题规模不可解，而不是单纯的学习失败。

最新 4000 episode 训练结果如下：

| 指标 | 数值 |
|---|---|
| training episodes | `4000` |
| final train coverage | `0.900` |
| final train success | `True` |
| train-time eval coverage | `0.902` |
| train-time eval success rate | `1.000` |
| independent eval coverage mean | `0.902` |
| independent eval success rate | `1.000` |
| independent eval episode length mean | `81.4` |
| planner failures | `0` |

最新独立评估 trace 显示，`planner_failures=0`，物理通信和有效通信均接近满连通，说明当前主要瓶颈不在 planner 失败或通信断裂。与此同时，评估 episode 的 `repeated_coverage_ratio` 常处于 `0.95-0.99`，表明策略虽然能达到覆盖目标，但仍存在明显重复探索和后期低效率问题。

## 7. 当前缺陷与改进路线

### 7.1 训练耗时问题

4000 episodes 长训约需 10 小时，不适合作为日常调参循环。后续应将训练改为分阶段实验：

```text
30 -> 100 -> 300 -> 500 -> 1000 -> 4000
```

每阶段固定 10 seed 评估，记录成功率、平均覆盖率、平均完成步数、重复覆盖比例、zero-gain streak、late reassignment 触发次数和每 episode wall-clock。若 500 episodes 已稳定达到 `success_rate >= 0.8` 且 `coverage_mean >= 0.90`，则不应继续默认跑 4000 episodes。4000 episodes 应保留为最终复现实验。

训练加速应优先 profile 环境和 planner。建议比较 `num_envs=4/8/12/16` 与 `ego_optimize_iterations=30/50/80`。如果 GPU 利用率较低，则瓶颈更可能在 Python 环境 step、frontier 检测、A* seed 或 EGO 优化，而不是 QMIX 网络本身。

### 7.2 重复探索问题

当前策略成功率已经足够高，但重复覆盖比例仍偏高。后续应从三方面改进。

第一，加入动态重复覆盖惩罚。在低覆盖阶段保留温和惩罚，允许必要回访；当覆盖率超过 `0.50` 后逐步提高 `w_repeat` 和 `w_overlap`；当 `zero_gain_streak >= 3` 时额外惩罚连续低收益动作。

第二，增强协同观测。候选特征可加入邻居当前目标到候选点的距离、邻居预测路径 footprint 重叠率、候选区域最近几步新增覆盖收益；邻居状态可加入最近新增覆盖和 zero-gain streak；全局 residual coverage 可从当前粗粒度扩展到 `4 x 4 x 2` 或 `5 x 5 x 2`。

第三，引入显式分工先验。可先使用软 Voronoi responsibility region，对跨区候选加 cost；再引入 frontier cluster penalty；若仍有严重抢目标现象，再采用 Hungarian 或 auction assignment 将无人机分配到不同 frontier cluster。

### 7.3 推荐近期实验路线

1. 运行吞吐 benchmark，确定本机最优 `num_envs` 和训练期 EGO 迭代数。
2. 将大场景默认开发训练改为 500/1000 episode 阶段式验证。
3. 加入动态重复覆盖惩罚和 footprint overlap penalty。
4. 细化 residual coverage global state，并加入 teammate intent。
5. 若重复覆盖仍高，再加入软分区或 frontier cluster 分配。
6. 最后用 1000/4000 episodes 做正式确认。

## 8. 主要贡献

Safe-CTDE-MACE 当前形成了以下贡献：

1. **统一的 CTDE 多无人机覆盖框架**：将 Shared DQN、QMIX、局部感知、地图融合、frontier 候选、安全屏蔽和轨迹规划集成到单一可运行工程中。
2. **面向三维体素空间的层次化动作建模**：将巨大三维动作空间压缩为固定数量 frontier 候选，同时保留信息增益、路径代价和空间上下文。
3. **安全强化与连续规划结合**：通过 SafetyShield 和 EGO-style planner 在高层策略下方提供可达性、避障和平滑轨迹约束。
4. **大场景 QMIX 成功配置**：在三机 `20 x 20 x 8` 场景中实现 `success_rate=1.000` 的 10 seed 评估结果。
5. **可诊断的协同探索实验体系**：trace 同时记录覆盖曲线、重复覆盖、通信链路、hover 原因、planner 状态、goal conflict 和 late reassignment。
6. **面向后续研究的可扩展结构**：保留 CBF-QP、3D CNN、概率通信、动态障碍和先进 MARL 算法扩展点。

## 9. 结论

Safe-CTDE-MACE 已经从基础环境与基线实现推进到可稳定求解三机大场景的 QMIX + EGO-style 框架。最新 4000 episode 长训和独立评估表明，当前配置可以稳定达到 90% 覆盖目标，并保持 planner 失败为零和物理通信连通。

下一阶段的研究重点不再是单纯提高覆盖成功率，而是降低训练成本和重复探索比例。通过分阶段训练、环境与 planner 侧加速、动态重复覆盖惩罚、协同观测增强和显式分工先验，框架有望从“可成功完成探索”进一步推进到“更高效、更低重复、更接近多机最优协作”的覆盖探索系统。
