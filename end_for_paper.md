# Safe-Ego-Planner 论文收尾复现实验指南

本文档用于在结束工程阶段重新生成论文所需的数据、图表和 ROS 可视化结果。它以当前仓库状态为准，覆盖 Safe-CTDE-MACE 的 Python 训练/评估/可视化、Python 到 ROS 的数据导出、Safe-Ego-Planner 真动力学执行和 RVIZ 展示。

## 0. 对旧指南现状的判断

### 0.1 `Safe-CTDE-MACE/Guide.md` 的 `## 4. 推荐实验顺序`

该章节已经不完全符合当前工程，不能直接作为论文收尾流程使用。

主要过期点：

- 旧指南仍写 `uav_rl` 是 Python 3.14 且 CUDA PyTorch 不可用，并推荐 `uav_rl_gpu`。当前实测 `uav_rl` 是 Python 3.12.13、`torch 2.5.1+cu121`，CUDA 可用，设备为 `NVIDIA GeForce RTX 3060 Laptop GPU`。
- 多处训练/评估命令仍使用 `--device cpu`，与“所有训练使用 CUDA 版 PyTorch”的当前要求冲突。
- 大场景评估命令中混入 `checkpoints/qmix_final.pt` 等旧命名，容易和 `qmix_ego_large.yaml` 的大场景 checkpoint 混淆。
- 课程学习章节重复且有维度不兼容问题；当前仓库没有完整可执行的 `configs/course/` 或课程训练入口，因此课程学习只能作为未来工作，不应进入论文收尾主流程。

结论：`## 4. 推荐实验顺序` 需要以本文件第 4-8 节替代。

### 0.2 `Guide_ego.md` 的 `# Safe-Ego-Planner ROS 验证指南`

ROS 闭环主线仍基本符合当前工程：包名为 `ego_planner`，最终入口为 `roslaunch ego_planner simple_run.launch`，链路是 JSON 地图与轨迹导出、`map_generator` 建图、`ros_eval_playback.py` 发布 Python 轨迹、`traj_bridge_node` 转 B-spline、`traj_server + so3_control + quadrotor_simulator` 执行。

但仍有几个必须注意的过期点：

- Python 导出命令应进入 `uav_rl`，且导出时也应使用 `--device cuda`。
- `src/planner/plan_manage/launch/simple_run.launch` 当前默认 JSON 指向 `result/ros_eval/qmix3000_episode.json`，但当前目录里存在的是 `result/ros_eval/ros_eval_episode.json`。因此运行 ROS 时必须显式传入 `episode_json:=.../ros_eval_episode.json`，不要依赖 launch 默认值。
- 旧优化记录中的 `randomize_obstacles: true`、`w_repeat=0.8`、`reservation_radius=4` 等值不是当前正式配置。当前 `qmix_ego_large.yaml` 使用 `randomize_obstacles: false`、`w_repeat=0.3`、`reservation_radius=2`。

结论：ROS 链路说明可沿用，但执行命令和配置解释以本文件第 9-11 节为准。

阅读旧 `Guide_ego.md` 时，建议把前 1-7 节当作 ROS 链路历史记录；第 8 节之后的多轮优化记录只作研发过程参考，其中部分配置已被当前正式配置覆盖。

## 1. 当前工程目标

本工程由两部分组成：

- `Safe-CTDE-MACE/`：Python 多无人机三维覆盖探索框架，核心路线是 `QMIX + EGO-style planner`。
- `src/`：ROS Noetic 下的 Safe-Ego-Planner / quadrotor simulator 工作空间，用于复现 Python 策略导出的障碍物、轨迹和动力学执行效果。

论文收尾建议采用如下主线：

```text
CUDA/PyTorch 环境检查
  -> Python 单元测试
  -> 原理演示与基线图
  -> 大场景 QMIX 训练
  -> 多 seed 独立评估
  -> 生成论文图表与 CSV
  -> 导出 ROS episode JSON
  -> catkin_make
  -> ROS smoke test
  -> RVIZ 真动力学可视化
  -> 收集 ros_execution_metrics.csv
```

当前正式大场景配置为：

```text
Safe-CTDE-MACE/safe_ctde_mace/configs/qmix_ego_large.yaml
```

关键参数：

| 项目 | 当前值 |
|---|---|
| 地图尺寸 | `20 x 20 x 8` |
| 体素尺度 | `1 voxel = 1 m` |
| 无人机数量 | `3` |
| 初始位置 | `[1,1,1]`, `[1,18,1]`, `[18,1,1]` |
| 目标覆盖率 | `0.90` |
| 最大步数 | `100` |
| planner | `ego` |
| EGO 优化迭代 | `150` |
| 通信半径 | `20.0` |
| 全局同步 | `global_sync_interval=0` |
| 并行环境 | `num_envs=4` |
| 训练设备 | `cuda` |
| 域随机化 | `randomize_obstacles: false` |

## 2. 基础环境

### 2.1 Python 环境

所有 Python 命令统一进入 `uav_rl`：

```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE
conda activate uav_rl
export PYTHONPATH=/home/jude/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH
```

CUDA 检查：

```bash
python -m safe_ctde_mace.scripts.check_cuda --device cuda
```

当前实测输出应类似：

```text
torch_version=2.5.1+cu121
cuda_available=True
cuda_version=12.1
device_count=1
resolved_device=cuda
device_name=NVIDIA GeForce RTX 3060 Laptop GPU
```

如果 `--device cuda` 报错，先不要开始训练；需要修复 `uav_rl` 中的 CUDA PyTorch。

注意：`Safe-CTDE-MACE/requirement.txt` 仍写有 `torch>=2.6`，但当前可用环境实测为 `torch 2.5.1+cu121` 且 CUDA 正常。论文收尾阶段不要为了“对齐 requirement”盲目重装 PyTorch，优先保持已验证可用的 CUDA 运行时。

### 2.2 ROS 环境

ROS 部分使用 ROS Noetic 和 `catkin_make`：

```bash
cd /home/jude/Safe_ego_planner
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

不要把本工程切到 `catkin build`/`catkin clean` 流程。当前工作空间使用的是 `catkin_make`。

## 3. 工程结构与关键文件

| 路径 | 作用 |
|---|---|
| `Safe-CTDE-MACE/safe_ctde_mace/configs/qmix_ego_large.yaml` | 论文主实验配置 |
| `Safe-CTDE-MACE/safe_ctde_mace/scripts/train_qmix.py` | QMIX 训练入口 |
| `Safe-CTDE-MACE/safe_ctde_mace/scripts/evaluate_qmix.py` | QMIX 评估入口 |
| `Safe-CTDE-MACE/safe_ctde_mace/scripts/export_ros_eval.py` | 导出 ROS JSON 与指标 |
| `Safe-CTDE-MACE/safe_ctde_mace/scripts/compare_planners.py` | A* 与 EGO-style 规划器对比 |
| `Safe-CTDE-MACE/safe_ctde_mace/envs/multi_uav_env.py` | 环境主循环、奖励、通信、候选和规划调度 |
| `Safe-CTDE-MACE/safe_ctde_mace/mapping/frontier_detector.py` | frontier 候选生成与特征计算 |
| `Safe-CTDE-MACE/safe_ctde_mace/mapping/frontier_graph.py` | region-level frontier graph |
| `Safe-CTDE-MACE/safe_ctde_mace/planning/ego_planner.py` | Python EGO-style 连续轨迹优化 |
| `Safe-CTDE-MACE/safe_ctde_mace/marl/qmix.py` | QMIX agent、mixer、checkpoint |
| `src/planner/plan_manage/launch/simple_run.launch` | ROS 最终入口 |
| `src/planner/plan_manage/launch/safe_ctde_multi_uav.launch` | 三机真执行验证 launch |
| `src/uav_simulator/traj_bridge/scripts/ros_eval_playback.py` | 发布 Python 轨迹、coverage 和 ROS 执行指标 |
| `src/uav_simulator/traj_bridge/src/traj_bridge_node.cpp` | `nav_msgs/Path` 转 `ego_planner/Bspline` |
| `src/uav_simulator/map_generator/src/random_forest_sensing.cpp` | 从 JSON 障碍物 box 生成 ROS 点云 |
| `src/planner/plan_manage/launch/safe_ctde.rviz` | Safe-CTDE 专用 RVIZ 视图 |

## 4. 原理展示建议

论文中的方法展示建议按以下层次组织。

### 4.1 CTDE 与 QMIX

训练阶段使用集中式信息：

- 每个 agent 的局部 observation。
- 团队 global state，包括覆盖率、残余覆盖分布、通信邻接、frontier 数量、有效动作比例等。
- QMIX mixer 用全局状态混合各 agent 的 Q 值，保持单调性约束。

执行阶段使用分布式策略：

- 每架 UAV 只根据自己的局部地图、自身状态、邻居摘要、coverage ratio、candidate features 和 action mask 选择候选 frontier。
- 选出的候选目标再经过 safety shield、目标去冲突和 EGO-style planner。

### 4.2 Frontier 候选与协同

当前 candidate feature schema 使用动态布局，三机大场景下 `max_neighbors=2`，候选特征包括：

- 到 UAV 的距离。
- 预期信息增益。
- 障碍物风险。
- reservation penalty。
- 邻居 overlap。
- A* path cost。
- responsibility penalty，但当前 `w_division=0.0`，即不强制软分区。
- assignment margins。
- grid quadrant / height layer。
- local uncovered density。
- spatial exclusivity。
- neighbor intent penalty。

当前已集成 region-level frontier graph 和空间候选筛选，用于提升候选多样性；课程学习和强制域随机化不进入最终复现实验主线。

### 4.3 安全与连续规划

执行链路：

```text
QMIX action
  -> frontier candidate
  -> SafetyShield 过滤不可达/冲突目标
  -> late reassignment 与目标去冲突
  -> EGOStylePlanner 使用 A* seed 生成连续轨迹
  -> TrajectoryTracker 执行一步
  -> 更新局部地图、全局覆盖和奖励
```

`EGOStylePlanner` 当前包含：

- 26 邻接 A* seed path。
- 控制点压缩。
- minimum-snap 风格平滑项。
- 动态 smooth/obstacle 权重。
- jerk 与 acceleration 约束投影。
- raw seed / axis-aligned fallback。

## 5. 健康检查

每次重新生成论文数据前先确认 CUDA 和关键导出链路：

```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE
conda activate uav_rl
export PYTHONPATH=/home/jude/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH

python -m safe_ctde_mace.scripts.check_cuda --device cuda
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_config.py tests/test_ros_export.py -q
```

如果只想快速检查 ROS 导出相关逻辑：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ros_export.py -q
```

### 5.1 当前全量 pytest 状态

当前全量测试命令为：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

截至本文档生成时，实测结果为 `70 passed, 7 failed`。失败集中在旧测试仍按老版本 candidate feature 宽度和 offset 断言，例如期望 `(4, 10)`、`(12, 11)`，而当前代码已经加入 responsibility penalty、spatial exclusivity、neighbor intent penalty 等新特征，实际宽度变为 `(4, 13)`、`(12, 14)`。因此论文收尾复现实验先以 `test_config.py`、`test_ros_export.py`、训练/评估 smoke 和 ROS smoke 作为运行前门槛；若要把工程测试也一并收尾，需要另行更新这些陈旧断言。

可选脚本帮助检查：

```bash
python -m safe_ctde_mace.scripts.train_qmix --help
python -m safe_ctde_mace.scripts.evaluate_qmix --help
python -m safe_ctde_mace.scripts.export_ros_eval --help
```

## 6. 原理演示与基础图

这些步骤用于生成论文中“环境可运行、frontier 逻辑、规划器对比”的辅助材料，不是最终主结果。

### 6.1 随机策略演示

```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE
conda activate uav_rl
export PYTHONPATH=/home/jude/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH

python -m safe_ctde_mace.scripts.demo_random_policy \
  --config safe_ctde_mace/configs/verified_baseline.yaml \
  --steps 50 \
  --save-dir artifacts/paper_random_demo
```

产物：

- `artifacts/paper_random_demo/episode.png`
- `artifacts/paper_random_demo/coverage_curve.png`

### 6.2 启发式 frontier 策略演示

```bash
python -m safe_ctde_mace.scripts.demo_heuristic_policy \
  --config safe_ctde_mace/configs/verified_baseline.yaml \
  --steps 50 \
  --save-dir artifacts/paper_heuristic_demo
```

产物：

- `artifacts/paper_heuristic_demo/episode.png`
- `artifacts/paper_heuristic_demo/coverage_curve.png`

### 6.3 Shared DQN + A* 基线

```bash
python -m safe_ctde_mace.scripts.train \
  --config safe_ctde_mace/configs/verified_baseline.yaml \
  --episodes 30 \
  --artifact-dir artifacts/paper_verified_train \
  --output checkpoints/paper_shared_dqn_final.pt

python -m safe_ctde_mace.scripts.evaluate \
  --config safe_ctde_mace/configs/verified_baseline.yaml \
  --checkpoint checkpoints/paper_shared_dqn_final.pt \
  --episodes 5 \
  --artifact-dir artifacts/paper_verified_eval
```

说明：`scripts/train.py` 没有 `--device` 参数；`SharedDQNAgent` 会在 `torch.cuda.is_available()` 为真时自动使用 CUDA。执行前仍以 `check_cuda.py --device cuda` 作为前置检查。

### 6.4 A* 与 EGO-style planner 对比

```bash
python -m safe_ctde_mace.scripts.compare_planners \
  --config safe_ctde_mace/configs/verified_baseline.yaml \
  --steps 20 \
  --artifact-dir artifacts/paper_planner_comparison
```

重点产物：

- `artifacts/paper_planner_comparison/planner_comparison.csv`
- `artifacts/paper_planner_comparison/planner_comparison.png`
- `artifacts/paper_planner_comparison/astar_episode.gif`
- `artifacts/paper_planner_comparison/ego_episode.gif`

## 7. 大场景 QMIX 训练

论文主实验从 `qmix_ego_large.yaml` 重新训练。所有训练使用 CUDA：

```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE
conda activate uav_rl
export PYTHONPATH=/home/jude/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH

python -m safe_ctde_mace.scripts.check_cuda --device cuda
```

### 7.1 正式复现实验

建议先用短训练确认流程，再用一个明确命名的 final run 作为论文结果来源。不要把某个固定 episode 数直接视为最终结果；历史记录中 2500/3000 episodes 评估并不都达标，最终应以独立评估是否通过第 8.1 节验收为准。

可先做 500 episodes smoke：

```bash
python -m safe_ctde_mace.scripts.train_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --episodes 500 \
  --device cuda \
  --num-envs 4 \
  --artifact-dir artifacts/paper_qmix_large_smoke_train \
  --output checkpoints/paper_qmix_large_smoke.pt
```

正式复现实验建议使用变量固定命名：

```bash
export PAPER_RUN=paper_qmix_large_final
export PAPER_EPISODES=2000
export PAPER_TRAIN_DIR=artifacts/${PAPER_RUN}_train
export PAPER_CKPT=checkpoints/${PAPER_RUN}.pt

python -m safe_ctde_mace.scripts.train_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --episodes ${PAPER_EPISODES} \
  --device cuda \
  --num-envs 4 \
  --artifact-dir ${PAPER_TRAIN_DIR} \
  --output ${PAPER_CKPT}
```

训练脚本会生成：

- `train_history.csv`
- `training_curves.png`
- `last_train_trace.json`
- `last_train_diagnostics.png`
- `evaluation_history.csv`
- `evaluation_summary.png`
- `evaluation_trace.json`
- `evaluation_diagnostics.png`
- `evaluation_episode.png`
- `evaluation_replay.gif`

### 7.2 当前已有历史结果的使用建议

仓库中已有多个历史 checkpoint 与 artifact，例如：

- `checkpoints/qmix_large_final.pt`
- `checkpoints/qmix_large_3000.pt`
- `checkpoints/qmix_large_2500.pt`
- `artifacts/qmix_large_train_3000/`
- `artifacts/qmix_large_train_2500/`
- `result/qmix_large_eval/`

这些结果可用于对照和调试，但不能默认作为论文最终结果。尤其是 `qmix_large_train_2500/`、`qmix_large_train_3000/` 等历史评估存在成功率不足或重复覆盖较高的问题。论文最终数据建议以本轮新生成并通过验收的 `${PAPER_RUN}` 目录为准，避免把不同配置阶段的结果混在一起。

## 8. 大场景独立评估

训练完成后必须单独评估，论文表格只从评估产物读取。

```bash
export PAPER_RUN=paper_qmix_large_final
export PAPER_CKPT=checkpoints/${PAPER_RUN}.pt
export PAPER_EVAL_DIR=artifacts/${PAPER_RUN}_eval

python -m safe_ctde_mace.scripts.evaluate_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --checkpoint ${PAPER_CKPT} \
  --device cuda \
  --seed-count 10 \
  --artifact-dir ${PAPER_EVAL_DIR}
```

评估输出重点看：

| 文件 | 用途 |
|---|---|
| `evaluation_history.csv` | 每个 seed 的 reward、coverage、success、episode length、碰撞、重复覆盖等 |
| `evaluation_summary.png` | 多 episode 汇总图 |
| `last_evaluation_trace.json` | 最后一轮逐步 trace |
| `last_evaluation_step_diagnostics.csv` | 每步 coverage、通信、hover、planner 状态 |
| `evaluation_failure_summary.csv` | 失败原因汇总 |
| `last_evaluation_diagnostics.png` | 覆盖、重复覆盖、通信、hover 等诊断图 |
| `last_evaluation_episode.png` | 最后一轮 3D 场景图 |
| `last_evaluation_replay.gif` | 最后一轮探索 GIF |

### 8.1 论文指标读取

建议论文中的数值都从 CSV 读取，不从终端日志手抄。

`evaluation_history.csv` 主要列：

- `coverage_ratio`：最终覆盖率。
- `success`：是否达到 `target_coverage_ratio=0.90`。
- `episode_length`：达到目标或被截断时的步数。
- `collision_count`：碰撞次数。
- `repeated_coverage_ratio`：重复覆盖比例。
- `mean_acceleration`、`max_acceleration`、`smoothness_cost`：Python 轨迹平滑性指标。
- `termination_reason`：`coverage_target` 或 `max_steps` 等。

快速统计命令：

```bash
python - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ.get("PAPER_EVAL_DIR", "artifacts/paper_qmix_large_final_eval")) / "evaluation_history.csv"
rows = list(csv.DictReader(path.open()))
success = sum(row["success"] == "True" for row in rows) / len(rows)
coverage = sum(float(row["coverage_ratio"]) for row in rows) / len(rows)
length = sum(float(row["episode_length"]) for row in rows) / len(rows)
repeat = sum(float(row["repeated_coverage_ratio"]) for row in rows) / len(rows)
print(f"episodes={len(rows)} success_rate={success:.3f} coverage_mean={coverage:.3f} episode_length_mean={length:.1f} repeated_coverage_mean={repeat:.3f}")
PY
```

建议验收标准：

- `coverage_mean >= 0.90`
- `success_rate >= 0.90`
- `planner_failures` 尽量为 0
- `collision_count` 尽量为 0
- `repeated_coverage_ratio` 是当前主要短板，若仍较高，需要在论文中如实作为局限性或消融讨论。

## 9. 导出 ROS episode

选定最终 checkpoint 后导出 ROS 使用的 JSON 与指标。

```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE
conda activate uav_rl
export PYTHONPATH=/home/jude/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH
export PAPER_RUN=paper_qmix_large_final
export PAPER_CKPT=checkpoints/${PAPER_RUN}.pt

python -m safe_ctde_mace.scripts.export_ros_eval \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --checkpoint ${PAPER_CKPT} \
  --device cuda \
  --seed 7 \
  --artifact-dir result/ros_eval
```

输出：

```text
result/ros_eval/ros_eval_episode.json
result/ros_eval/python_metrics.csv
result/ros_eval/coverage_curve.csv
```

JSON 约定：

- `metadata.trajectory_units = meter`
- `metadata.trajectory_point = voxel_center`
- `metadata.coordinate_frame = world`
- `metadata.time_step_sec = trajectory_execution_dt`
- 障碍物 box 使用闭区间 voxel 语义。
- Python voxel 坐标到 ROS meter 坐标的转换为：

```text
ros_x = (voxel_x + 0.5) * voxel_resolution
ros_y = (voxel_y + 0.5) * voxel_resolution
ros_z = (voxel_z + 0.5) * voxel_resolution
```

三机初始位置对应：

| UAV | Python voxel | ROS meter |
|---|---|---|
| UAV1 | `[1, 1, 1]` | `[1.5, 1.5, 1.5]` |
| UAV2 | `[1, 18, 1]` | `[1.5, 18.5, 1.5]` |
| UAV3 | `[18, 1, 1]` | `[18.5, 1.5, 1.5]` |

## 10. ROS 构建与静态检查

```bash
cd /home/jude/Safe_ego_planner
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

检查包名：

```bash
rospack find ego_planner
rospack find traj_bridge
rospack find map_generator
```

检查 launch 展开：

```bash
roslaunch --files ego_planner simple_run.launch \
  use_rviz:=false \
  episode_json:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json

roslaunch --nodes ego_planner simple_run.launch \
  use_rviz:=false \
  episode_json:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json
```

`roslaunch --nodes` 应包含：

```text
/map_generator
/traj_bridge
/ros_eval_playback
/uav1/quadrotor_simulator_so3
/uav1/so3_control
/uav1/traj_server
/uav2/quadrotor_simulator_so3
/uav2/so3_control
/uav2/traj_server
/uav3/quadrotor_simulator_so3
/uav3/so3_control
/uav3/traj_server
```

注意：当前 `simple_run.launch` 默认 JSON 不是可靠路径，必须显式传 `episode_json`。

## 11. ROS 运行与 RVIZ 可视化

### 11.1 无 RVIZ smoke test

```bash
cd /home/jude/Safe_ego_planner
source /opt/ros/noetic/setup.bash
source devel/setup.bash

timeout 12s roslaunch ego_planner simple_run.launch \
  use_rviz:=false \
  episode_json:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json \
  metrics_dir:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval
```

预期日志：

- `Loaded ... voxel boxes from .../ros_eval_episode.json`
- `published /uav1/python_traj with ... points`
- `UAV 1 published B-spline traj_id=1`
- `UAV 2 published B-spline traj_id=1`
- `UAV 3 published B-spline traj_id=1`
- 退出时写入 `result/ros_eval/ros_execution_metrics.csv`

### 11.2 RVIZ 真执行可视化

```bash
roslaunch ego_planner simple_run.launch \
  episode_json:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json \
  metrics_dir:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval \
  use_rviz:=true
```

RVIZ 使用：

```text
src/planner/plan_manage/launch/safe_ctde.rviz
```

核心 topic：

| Topic | 含义 |
|---|---|
| `/map_generator/global_cloud` | JSON 障碍物生成的全局点云 |
| `/uav{i}/python_traj` | Python 导出的米制 `nav_msgs/Path` |
| `/safe_ctde_mace/uav{i}/python_path_marker` | Python 轨迹 marker |
| `/safe_ctde_mace/coverage_ratio` | coverage ratio |
| `/uav{i}/planning/bspline` | `traj_bridge_node` 发布的 B-spline |
| `/uav{i}/planning/pos_cmd` | `traj_server` 输出的位置命令 |
| `/uav{i}/sim/odom` | quadrotor simulator 里真实执行的 odometry |
| `/uav{i}/odom_visualization/path` | RVIZ 中实时执行轨迹 |
| `/uav{i}/odom_visualization/robot` | RVIZ 中 UAV mesh |

ROS 数据链路：

```text
ros_eval_episode.json
  -> map_generator/random_forest
       -> /map_generator/global_cloud
  -> ros_eval_playback.py
       -> /uav{i}/python_traj
       -> /safe_ctde_mace/uav{i}/python_path_marker
       -> /safe_ctde_mace/coverage_ratio
  -> traj_bridge_node
       -> /uav{i}/planning/bspline
  -> traj_server
       -> /uav{i}/planning/pos_cmd
  -> so3_control + quadrotor_simulator_so3
       -> /uav{i}/sim/odom
       -> /uav{i}/odom_visualization/path
  -> ros_execution_metrics.csv
```

## 12. ROS 指标与论文可视化截图

ROS 侧最终指标：

```text
Safe-CTDE-MACE/result/ros_eval/ros_execution_metrics.csv
```

该文件由 `ros_eval_playback.py` 在 shutdown 时根据 `/uav{i}/sim/odom` 采样写出，包含：

- `sample_count`
- `path_length`
- `mean_speed`
- `max_speed`
- `mean_acceleration`
- `max_acceleration`
- `mean_jerk`
- `max_jerk`

建议论文中把 Python 侧指标和 ROS 侧指标分开：

- Python 侧用于证明策略覆盖性能：`evaluation_history.csv`、`coverage_curve.csv`。
- Python 侧用于策略轨迹指标：`python_metrics.csv`。
- ROS 侧优先用于展示 B-spline + quadrotor simulator 执行结果：RVIZ 截图和执行轨迹 topic。

### 12.1 ROS metrics sanity check

`ros_execution_metrics.csv` 必须重新生成并做数值检查后才能作为论文表格。当前历史文件中曾出现过 `max_speed` 上百、`mean_acceleration` 上万的异常值，说明 odom 时间戳/采样间隔可能产生尖峰；这种文件只能用于排查，不能直接入论文。

检查命令：

```bash
python - <<'PY'
import csv
from pathlib import Path

path = Path("/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_execution_metrics.csv")
rows = list(csv.DictReader(path.open()))
for row in rows:
    uav = row["uav_id"]
    sample_count = int(row["sample_count"])
    max_speed = float(row["max_speed"])
    mean_acc = float(row["mean_acceleration"])
    max_acc = float(row["max_acceleration"])
    print(f"uav={uav} samples={sample_count} max_speed={max_speed:.3f} mean_acc={mean_acc:.3f} max_acc={max_acc:.3f}")
    if sample_count < 20 or max_speed > 5.0 or mean_acc > 20.0 or max_acc > 100.0:
        print("  WARN: ROS metrics look implausible; use RVIZ/odom trace for visualization and regenerate before quoting numbers.")
PY
```

建议：论文中可先使用 RVIZ 截图和 `/uav{i}/odom_visualization/path` 说明 ROS 闭环执行效果；只有 sanity check 通过的 `ros_execution_metrics.csv` 才写入定量表格。

截图建议：

- RVIZ 全局视角：同时显示障碍物、三机 mesh、三条动态 path。
- 局部视角：展示 B-spline 执行轨迹和障碍物间距。
- 论文图表：使用 `evaluation_summary.png`、`last_evaluation_diagnostics.png`、`last_evaluation_episode.png`、`last_evaluation_replay.gif` 中的关键帧。

## 13. 推荐最终复现顺序

1. 检查 CUDA：

```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE
conda activate uav_rl
export PYTHONPATH=/home/jude/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH
python -m safe_ctde_mace.scripts.check_cuda --device cuda
```

2. 跑关键 Python 测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_config.py tests/test_ros_export.py -q
```

全量 `pytest -q` 当前仍有旧 feature schema 断言失败，见第 5.1 节。

3. 生成基础演示图：

```bash
python -m safe_ctde_mace.scripts.demo_random_policy --config safe_ctde_mace/configs/verified_baseline.yaml --steps 50 --save-dir artifacts/paper_random_demo
python -m safe_ctde_mace.scripts.demo_heuristic_policy --config safe_ctde_mace/configs/verified_baseline.yaml --steps 50 --save-dir artifacts/paper_heuristic_demo
python -m safe_ctde_mace.scripts.compare_planners --config safe_ctde_mace/configs/verified_baseline.yaml --steps 20 --artifact-dir artifacts/paper_planner_comparison
```

4. 训练大场景：

```bash
export PAPER_RUN=paper_qmix_large_final
export PAPER_EPISODES=4000
export PAPER_TRAIN_DIR=artifacts/${PAPER_RUN}_train
export PAPER_CKPT=checkpoints/${PAPER_RUN}.pt

python -m safe_ctde_mace.scripts.train_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --episodes ${PAPER_EPISODES} \
  --device cuda \
  --num-envs 4 \
  --artifact-dir ${PAPER_TRAIN_DIR} \
  --output ${PAPER_CKPT}
```

5. 独立评估：

```bash
export PAPER_RUN=paper_qmix_large_final
export PAPER_CKPT=checkpoints/${PAPER_RUN}.pt
export PAPER_EVAL_DIR=artifacts/${PAPER_RUN}_eval

python -m safe_ctde_mace.scripts.evaluate_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --checkpoint ${PAPER_CKPT} \
  --device cuda \
  --seed-count 10 \
  --artifact-dir ${PAPER_EVAL_DIR}
```

6. 导出 ROS episode：

```bash
export PAPER_RUN=paper_qmix_large_final
export PAPER_CKPT=checkpoints/${PAPER_RUN}.pt

python -m safe_ctde_mace.scripts.export_ros_eval \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --checkpoint ${PAPER_CKPT} \
  --device cuda \
  --seed 7 \
  --artifact-dir result/ros_eval
```

7. 构建 ROS：

```bash
cd /home/jude/Safe_ego_planner
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

8. ROS smoke test：

```bash
timeout 12s roslaunch ego_planner simple_run.launch \
  use_rviz:=false \
  episode_json:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json \
  metrics_dir:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval
```

9. RVIZ 可视化：

```bash
roslaunch ego_planner simple_run.launch \
  use_rviz:=true \
  episode_json:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json \
  metrics_dir:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval
```

## 14. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA was requested but is not available` | `uav_rl` 中不是 CUDA PyTorch 或驱动不可见 | 先修复 `python -m safe_ctde_mace.scripts.check_cuda --device cuda` |
| `ModuleNotFoundError: safe_ctde_mace` | 没设置 `PYTHONPATH` | 在 `Safe-CTDE-MACE` 下执行 `export PYTHONPATH=/home/jude/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH` |
| `pytest` 被 ROS2/Jazzy 插件污染 | 外部 pytest 插件干扰 | 关键检查用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_config.py tests/test_ros_export.py -q`；全量状态见第 5.1 节 |
| checkpoint architecture mismatch | checkpoint 来自不同地图/候选数/agent 数/feature schema | 使用同一配置重新训练和评估 |
| `roslaunch plan_manage ...` 失败 | ROS 包名不是 `plan_manage` | 使用 `roslaunch ego_planner simple_run.launch` |
| 找不到 `qmix3000_episode.json` | `simple_run.launch` 默认值过期 | 显式传 `episode_json:=.../ros_eval_episode.json` |
| RVIZ 没有轨迹 | JSON 路径错、playback 未到 start delay、bridge 未收到 path | 检查 `/uav{i}/python_traj`、`/uav{i}/planning/bspline` |
| 没有 `ros_execution_metrics.csv` | launch 未正常退出或 playback 没收到 odom | 先跑 `timeout 12s ... use_rviz:=false`，看 shutdown 日志 |
| 课程学习无法迁移 checkpoint | 地图尺寸和 observation 维度变化 | 当前论文收尾不启用课程学习 |

## 15. 论文结果归档建议

建议最终把以下目录和文件作为论文结果集：

```text
Safe-CTDE-MACE/artifacts/paper_random_demo/
Safe-CTDE-MACE/artifacts/paper_heuristic_demo/
Safe-CTDE-MACE/artifacts/paper_planner_comparison/
Safe-CTDE-MACE/artifacts/paper_qmix_large_final_train/
Safe-CTDE-MACE/artifacts/paper_qmix_large_final_eval/
Safe-CTDE-MACE/checkpoints/paper_qmix_large_final.pt
Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json
Safe-CTDE-MACE/result/ros_eval/python_metrics.csv
Safe-CTDE-MACE/result/ros_eval/coverage_curve.csv
Safe-CTDE-MACE/result/ros_eval/ros_execution_metrics.csv  # 仅在 sanity check 通过后作为定量指标
```

写论文时，每个数字都应标注来源：

- 训练曲线：`paper_qmix_large_final_train/train_history.csv`
- 成功率与覆盖率：`paper_qmix_large_final_eval/evaluation_history.csv`
- 失败诊断：`paper_qmix_large_final_eval/evaluation_failure_summary.csv`
- Python 轨迹平滑性：`result/ros_eval/python_metrics.csv`
- ROS 执行动力学指标：`result/ros_eval/ros_execution_metrics.csv`，但必须通过第 12.1 节 sanity check。
- ROS 可视化截图：`safe_ctde.rviz` 运行窗口截图
