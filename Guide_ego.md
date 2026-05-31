# Safe-Ego-Planner ROS 验证指南

## 1. 当前闭环目标

本工程把 Safe-CTDE-MACE 的三机 QMIX 覆盖探索结果导出为离线 JSON，再由 ROS Noetic 读取同一份 JSON 搭建障碍物环境、发布 Python 轨迹、转换为 B-spline，并驱动 `quadrotor_simulator_so3 + so3_control + traj_server` 做真实动力学执行和 RVIZ 可视化。

当前约定已经统一为：

| 项目 | 约定 |
|---|---|
| ROS 包名 | `ego_planner`，源码目录仍是 `src/planner/plan_manage` |
| 最终入口 | `roslaunch ego_planner simple_run.launch` |
| 坐标尺度 | `1 voxel = 1 m` |
| 坐标原点 | Python 风格 `[0, 20] x [0, 20] x [0, 8]`，不使用 ego-planner 原始居中坐标 |
| 轨迹 topic | `/uav{i}/python_traj` 使用 `nav_msgs/Path`，单位已经是 ROS meter |
| 默认 JSON | `Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json` |

## 2. 一次完整运行

### 2.1 导出 Python 评估结果

在容器 `0163e95e4185` 中执行：

```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE

python3 -m safe_ctde_mace.scripts.export_ros_eval \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --checkpoint checkpoints/qmix_large_final.pt \
  --device cpu \
  --seed 7 \
  --artifact-dir result/ros_eval
```

导出成功后应看到类似：

```text
ros_eval_export=result/ros_eval/ros_eval_episode.json coverage=0.901 success=True episode_length=98
```

该步骤会生成：

| 文件 | 内容 |
|---|---|
| `result/ros_eval/ros_eval_episode.json` | 地图参数、9 个障碍物 box、三机米制轨迹、trace 和 summary |
| `result/ros_eval/python_metrics.csv` | Python 侧每机路径长度、加速度、平滑指标 |
| `result/ros_eval/coverage_curve.csv` | 覆盖率、每步新增覆盖、重复覆盖、碰撞计数 |

### 2.2 构建 ROS 工作空间

必须先 source Noetic，否则 `catkin_make` 不在 PATH 中：

```bash
cd /home/jude/Safe_ego_planner
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

不要使用 `catkin clean` 或 `catkin build`。当前容器里没有 `catkin_tools`，已有 `build/.built_by` 和 `devel/.built_by` 也是 `catkin_make`。

### 2.3 启动 ROS 真执行验证

```bash
cd /home/jude/Safe_ego_planner
source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch ego_planner simple_run.launch \
  episode_json:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json \
  metrics_dir:=/home/jude/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval
```

如果只做终端 smoke test，不打开 RVIZ：

```bash
roslaunch ego_planner simple_run.launch use_rviz:=false
```

启动后应看到：

- `map_generator` 从 JSON 读取 `9` 个 voxel boxes；
- `ros_eval_playback` 发布 `/uav1/python_traj`、`/uav2/python_traj`、`/uav3/python_traj`；
- `traj_bridge` 为三机各发布一条 `/uav{i}/planning/bspline`；
- `traj_server` 发布 `/uav{i}/planning/pos_cmd`；
- `quadrotor_simulator_so3` 发布 `/uav{i}/sim/odom`；
- 退出 launch 时写入 `result/ros_eval/ros_execution_metrics.csv`。

### 2.4 RVIZ 视图约定

`safe_ctde.rviz` 统一做了以下显示调整，便于对比轨迹与障碍物：

- 背景色使用淡暖色 `255; 253; 224`，减少灰度遮蔽；
- 障碍物点云（Safe-CTDE Obstacles）使用 ego-planner 蓝色 `85; 170; 255`，Style=Boxes，Size=0.1，Alpha=1；
- 增加三机 mesh marker：`/uav1/odom_visualization/robot`、`/uav2/odom_visualization/robot`、`/uav3/odom_visualization/robot`；
- 增加动态 path：`/uav{i}/odom_visualization/path`，对应每机实时轨迹。

## 3. ROS 图与文件职责

### 3.1 核心链路

```text
Safe-CTDE-MACE export_ros_eval
  -> result/ros_eval/ros_eval_episode.json
  -> map_generator/random_forest
       -> /map_generator/global_cloud
  -> traj_bridge/ros_eval_playback.py
       -> /uav{i}/python_traj
       -> /safe_ctde_mace/uav{i}/python_path_marker
       -> /safe_ctde_mace/coverage_ratio
  -> traj_bridge_node
       -> /uav{i}/planning/bspline
  -> ego_planner/traj_server
       -> /uav{i}/planning/pos_cmd
  -> so3_control + so3_quadrotor_simulator
       -> /uav{i}/sim/odom
  -> ros_execution_metrics.csv
```

### 3.2 关键文件

| 文件 | 功能 |
|---|---|
| `Safe-CTDE-MACE/safe_ctde_mace/scripts/export_ros_eval.py` | 运行 1 个 QMIX episode 并导出 ROS JSON |
| `Safe-CTDE-MACE/safe_ctde_mace/utils/ros_export.py` | JSON schema、坐标转换、Python 指标 CSV |
| `src/uav_simulator/map_generator/src/random_forest_sensing.cpp` | 从 JSON 闭区间 box 生成 ROS 点云 |
| `src/uav_simulator/traj_bridge/scripts/ros_eval_playback.py` | 读取 JSON，发布轨迹/marker/覆盖率，并记录 ROS odom 指标 |
| `src/uav_simulator/traj_bridge/src/traj_bridge_node.cpp` | 米制 path 到 `ego_planner/Bspline`，每条轨迹只发布一次 |
| `src/planner/plan_manage/launch/safe_ctde_multi_uav.launch` | 三机真执行验证 launch |
| `src/planner/plan_manage/launch/simple_run.launch` | 最终入口，include 三机验证 launch |
| `src/planner/plan_manage/launch/safe_ctde.rviz` | Safe-CTDE 专用 RVIZ 视图 |

## 4. 坐标与地图契约

Python `qmix_ego_large.yaml` 的关键配置：

```yaml
grid_size: [20, 20, 8]
voxel_resolution: 1.0
initial_positions:
  - [1, 1, 1]
  - [1, 18, 1]
  - [18, 1, 1]
manual_boxes:
  - min_corner: [8, 8, 0]
    max_corner: [11, 11, 5]
```

导出到 ROS 时使用体素中心：

```text
ros_x = (voxel_x + 0.5) * voxel_resolution
ros_y = (voxel_y + 0.5) * voxel_resolution
ros_z = (voxel_z + 0.5) * voxel_resolution
```

因此三机初始位置为：

| UAV | Python voxel | ROS meter |
|---|---:|---:|
| UAV1 | `[1, 1, 1]` | `[1.5, 1.5, 1.5]` |
| UAV2 | `[1, 18, 1]` | `[1.5, 18.5, 1.5]` |
| UAV3 | `[18, 1, 1]` | `[18.5, 1.5, 1.5]` |

障碍物 box 使用闭区间 voxel 语义，与 `VoxelWorld.add_box()` 一致。ROS map generator 会把每个 occupied voxel 填成致密点云，供 RVIZ 和后续碰撞/地图检查使用。

## 5. 验收命令

### 5.1 Python 测试

宿主机如果被 ROS2/Jazzy pytest 插件污染，可禁用自动插件：

```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

最小导出测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ros_export.py -q
```

### 5.2 ROS 静态检查

```bash
source /opt/ros/noetic/setup.bash
source /home/jude/Safe_ego_planner/devel/setup.bash

rospack find ego_planner
rospack find traj_bridge
rospack find map_generator

roslaunch --files ego_planner simple_run.launch use_rviz:=false
roslaunch --nodes ego_planner simple_run.launch use_rviz:=false
```

`roslaunch --nodes` 应包含：

```text
/map_generator
/traj_bridge
/ros_eval_playback
/uav1/quadrotor_simulator_so3
/uav1/so3_control
/uav1/traj_server
...
/uav3/traj_server
```

### 5.3 短时运行 smoke

```bash
timeout 12s roslaunch ego_planner simple_run.launch use_rviz:=false
```

关键日志：

```text
Loaded 9 voxel boxes
published /uav1/python_traj with 99 points
UAV 1 published B-spline traj_id=1
UAV 2 published B-spline traj_id=1
UAV 3 published B-spline traj_id=1
wrote .../ros_execution_metrics.csv
```

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `roslaunch plan_manage ...` 失败 | 包名不是 `plan_manage` | 使用 `roslaunch ego_planner simple_run.launch` |
| `catkin: command not found` | 容器没有 `catkin_tools` | `source /opt/ros/noetic/setup.bash && catkin_make` |
| 找不到 `ros_eval_episode.json` | 尚未导出 Python episode | 先执行 `python3 -m safe_ctde_mace.scripts.export_ros_eval ...` |
| Python 3.8 报 `dataclass(slots=True)` 或 `zip(strict=True)` | 项目原代码使用 Python 3.10 特性 | `safe_ctde_mace/__init__.py` 已加入 3.8 兼容 shim |
| RVIZ 没有轨迹 | playback 还未到 `start_delay` 或 JSON 路径错误 | 检查 `/uav{i}/python_traj` 和 launch 参数 `episode_json` |
| B-spline 反复从头执行 | 旧版 bridge 定时重发并刷新 `start_time` | 当前 `traj_bridge_node` 已改成每条 path 只发布一次 |

## 7. 当前验证记录

2026-05-26 已完成以下验证：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ros_export.py -q` 通过；
- 容器内 `catkin_make` 通过；
- `export_ros_eval.py` 生成 `coverage=0.901 success=True episode_length=98` 的 JSON；
- `roslaunch --files/--nodes ego_planner simple_run.launch use_rviz:=false` 通过；
- `timeout 12s roslaunch ego_planner simple_run.launch use_rviz:=false` 可发布三机 path/B-spline 并写出 ROS 执行指标。

## 8. 优化记录 (2026-05-28)

### 8.1 问题诊断

当前大场景训练虽然 `success=True`，达到成功指标，但存在以下缺陷：

| 问题 | 现状 | 根因 |
|---|---|---|
| `repeated_coverage_ratio` | 0.95-0.99 | Candidate 生成缺乏跨机协调感知，reservation 事后生效 |
| 轨迹折线飞行 | 平滑性差 | EGO planner 用 A* seed + 简陋 smoothing，缺少 kinodynamic optimization |

### 8.2 解决方案架构

```
┌─────────────────────────────────────────────────────────────┐
│  高层RL策略 (Region-level policy)                            │
│  输入: global state + graph embedding                        │
│  输出: 每个agent的region_id + intra-region执行优先级          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  图结构层 (Frontier Region Graph)                           │
│  - Nodes: frontier cluster centroids                        │
│  - Edges: spatial adjacency + A* distance                   │
│  - Node features: unknown count, density, position, status   │
│  - 更新频率: 每 step 重建                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  底层执行 (Intra-region path planning)                      │
│  - A*/RRT* 在 region 内部选具体 voxel goal                  │
│  - EGO-style trajectory optimization (minimum-snap)         │
│  - Safety shield 碰撞检测                                   │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 P0 快速修复 [已完成]

**配置文件 `qmix_ego_large.yaml`**（2026-05-28 已合并）：

```yaml
reward:
  w_repeat: 0.8    # 0.6 → 0.8，增强重复覆盖惩罚
  w_overlap: 1.0   # 1.5 → 1.0，调整重叠惩罚

environment:
  reservation_radius: 4  # 2 → 4，扩大预留半径
```

**完成状态**：已实现。`w_repeat=0.8`、`w_overlap=1.0`、`reservation_radius=4` 均已在 `qmix_ego_large.yaml` 中配置。

**效果**：立竿见影，约 1 天可完成，主要缓解重复覆盖问题。

### 8.4 P1 核心改动：图引导的候选生成 [已完成]

**完成状态**：
- `mapping/frontier_graph.py` - 已实现并集成到 `frontier_detector.py`
- `marl/graph_attention.py` - 已创建但**未集成**到 QNetwork

**新增文件 `mapping/frontier_graph.py`**：

```python
class RegionNode:
    """区域节点：代表一组相邻 frontier voxel 的聚类"""
    centroid: tuple[int, int, int]      # 聚类中心
    voxels: list[tuple[int, int, int]]   # 聚类内所有 voxel
    coverage_potential: float            # 覆盖潜力

class FrontierGraph:
    """区域级 frontier 图结构"""
    connection_tolerance: float          # 聚类连接容忍度
    nodes: list[RegionNode]              # 区域节点列表
    adjacency: dict[int, set[int]]       # 邻接关系

    def build_from_frontiers(frontiers)  # BFS 聚类构建图
    def get_region_candidates(max_regions) # 返回按 coverage_potential 排序的候选
    def mark_reservation(region_idx)     # 标记某 region 被 reservation
```

**修改 `mapping/frontier_detector.py`**（已集成）：

- 导入 `FrontierGraph`
- `generate_candidates()` 集成 region-level candidate 生成：
  1. 先用 `FrontierGraph.build_from_frontiers(frontiers)` 构建区域图
  2. 用 `region_graph.get_region_candidates()` 获取 region-level 候选
  3. 再与原有 voxel-level candidate 合并评分
  4. 最终选择时既考虑 region 多样性，也考虑 voxel 粒度

**新增文件 `marl/graph_attention.py`**（已创建但未集成）：

```python
class GraphAttentionLayer(nn.Module):
    """图注意力层：基于空间关系的多智能体意图预测"""
    # 多头注意力机制
    # 输入: agent_features, edge_index
    # 输出: 聚合后的特征

class AgentIntentPredictor(nn.Module):
    """基于图注意力的邻居意图预测器"""
    # 预测每个 agent 的邻居可能会选择哪个 candidate

def build_communication_edges(positions, comm_range):
    """基于通信范围构建边索引"""
```

**预留接口（待集成到 QNetwork）**：

```python
class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=256,
                 region_embed_dim=0,   # P1: region embedding 维度
                 use_gat_attention=False):  # P1: GAT attention 预留
    # 支持 region_embedding 输入拼接
```

**效果**：1-2 周，核心解决重复覆盖问题。

### 8.5 P2 轨迹平滑 [已完成]

**修改 `planning/ego_planner.py`**（已实现）：

```python
class EGOStylePlanner:
    def __init__(self, ..., optimize_iterations=100):  # 30 → 100
        self.optimize_iterations = 100

    def _optimize_control_points(self, points, states):
        # P2: 添加 minimum-snap 目标函数
        for iteration in range(self.optimize_iterations):
            progress = iteration / (self.optimize_iterations - 1)
            # 动态调整权重：前期重平滑，后期重避障
            current_smooth = self.smooth_weight * (1.0 - 0.3 * progress)
            current_obstacle = self.obstacle_weight * (1.0 + 0.5 * progress)

            for index in range(1, len(optimized) - 1):
                # minimum-snap 平滑项（4阶差分）
                snap_term = (optimized[index-2] - 4*optimized[index-1] + 6*current
                             - 4*optimized[index+1] + optimized[index+2])
                smooth_push = 0.5*(optimized[index-1] + optimized[index+1]) - current \
                              + 0.1 * snap_term
                # ... 避障项 ...
```

**关键改动**（已实现）：
- `optimize_iterations`: 30 → 100（已提升）
- 添加 `snap_term`（4阶差分）作为 minimum-snap 平滑目标
- 动态权重：smooth_weight 前期更强，obstacle_weight 后期更强

**效果**：2-3 周，解决轨迹折线飞行问题。

### 8.6 验收指标

| 指标 | 当前值 | 目标值 | 测量方法 | 完成状态 |
|------|--------|--------|----------|----------|
| `repeated_coverage_ratio` | 0.95-0.99 | < 0.30 | `EpisodeMetrics.repeated_coverage_ratio` | P0/P1 已实施，待验证 |
| `trajectory_smoothness_cost` | 高（折线） | 降低 50% | `ContinuousTrajectory.metrics()["smoothness_cost"]` | P2 已实施，待验证 |
| `coverage_ratio` | ~0.90 | >= 0.90 | 最终 `coverage_ratio` 不降低 | 已有基线 |
| `success_rate` | 有 | 不降低 | 100次评估的成功率 | 已有基线 |

### 8.7 下一阶段计划

完成状态一览：
- P0 快速修复 [已完成]：配置参数调整（w_repeat, w_overlap, reservation_radius）
- P1 核心改动 [已完成]：frontier_graph.py 已实现并集成；graph_attention.py 已创建但未集成
- P2 轨迹平滑 [已完成]：minimum-snap 和动态权重已实现
- P3 增强图注意力集成 [待集成]：graph_attention.py 已创建，需集成到 QNetwork.forward
- P4 显式分工层 [未实现]
- P5 课程学习 [未实现]

如果 P0/P1/P2 实施后 `repeated_coverage_ratio` 仍 > 0.30，考虑：
- P4：显式分工层 - 软 Voronoi region + frontier cluster penalty
- P5：课程学习 - 小地图低目标 → 大地图高目标

### 8.8 2026-05-28 迭代优化记录

**问题重新诊断：**
用户指出4000 episodes的"成功"实际上是"假成功"——靠重复探索硬凑覆盖率，效率极低。真正目标是：多无人机高效分工、一次覆盖到位、降低重复探索。

**新策略：分阶段、保守调整**

#### 阶段1（2026-05-28）：配置回退
修改 `qmix_ego_large.yaml`：
```yaml
reward:
  w_repeat: 0.3      # 0.8 → 0.3
  w_overlap: 0.5     # 1.0 → 0.5
  w_info: 2.0        # 1.5 → 2.0

environment:
  reservation_radius: 2   # 4 → 2
```

#### 阶段2（2026-05-28）：验证训练稳定性
- 训练：200 episodes
- 结果：**eval_success_rate=1.0, eval_coverage=0.901** ✅
- 但 repeated_coverage_ratio=0.975，仍在高位

#### 阶段3（2026-05-28）：软分工机制
修改 `frontier_detector.py`：
- 新增 `_compute_responsibility_penalty()` 方法
- 在 candidate 评分中加入 `w_division * responsibility_penalty`
- 新增配置 `w_division: 1.5`

#### 阶段4/4b/4c（2026-05-28）：responsibility_penalty调参迭代

| 阶段 | w_division | success_rate | coverage | 说明 |
|------|------------|--------------|----------|------|
| 阶段4 | 1.5 | 0.700 | 0.894 | 过重，无限重复 |
| 阶段4b | 0.5 | 0.400 | 0.888 | 仍过重 |
| 阶段4c | 0.0 | 0.800 | 0.901 | 完全移除后恢复 |

**根因分析（review agent）：**
- responsibility_penalty是冗余的强制分散机制
- 已有机制（reserved_penalty, neighbor_overlap, late_reassignment）已足够
- 强制分散反而干扰了协作覆盖效率

#### 阶段5：确认最终配置
- **结论：移除responsibility_penalty，使用阶段2配置**
- 配置：
  - w_repeat: 0.3
  - w_overlap: 0.5
  - w_info: 2.0
  - reservation_radius: 2
  - w_division: 0.0

#### 阶段6：最终长时训练
- 训练：1000 episodes
- 使用配置：阶段2基础配置（无responsibility_penalty）
- 验收：success_rate >= 0.95, coverage >= 0.90

**当前核心发现：**
- responsibility_penalty（强制分散）不能解决问题
- 协作分工应由已有机制（late_reassignment等）处理
- 后续改进方向：增强late_reassignment参数，而非强制分工

## 9. 2026-05-29 改进验证结果

### 9.1 代码改进内容

**轨迹平滑性增强（ego_planner.py）**：
- `__init__` 添加 `min_segment_length=2.0` 和 `max_jerk=2.0` 参数
- `optimize_iterations` 从80提升到150
- `_compress_path` 添加 `min_segment_length` 检查，避免过短段被保留
- `_optimize_control_points` 添加速度/加速度/jerk约束投影

**候选多样性增强（frontier_detector.py）**：
- `CandidateFeatureLayout` 添加 `spatial_exclusivity` 属性（feature size 11 + max_neighbors）
- `candidate_score` 增加 `spatial_exclusivity` 权重 +0.5
- `generate_candidates` 添加空间分块预过滤（`_grid_filter_candidates`）
- `compute_features` 添加 `spatial_exclusivity` 特征计算

**配置文件更新（qmix_ego_large.yaml）**：
| 参数 | 原值 | 新值 |
|------|------|------|
| `ego_optimize_iterations` | 80 | 150 |
| `ego_smooth_weight` | 0.25 | 0.30 |
| `num_frontier_candidates` | 10 | 12 |
| `candidate_min_separation` | 2.0 | 2.5 |
| `late_reassign_min_coverage` | 0.50 | 0.60 |
| `late_reassign_zero_gain_streak` | 3 | 2 |

### 9.2 训练结果（3000 episodes）

**训练命令**：
```bash
cd /home/jude/Safe_ego_planner/Safe-CTDE-MACE
export PYTHONPATH=/home/jude/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH
/home/jude/anaconda3/envs/uav_rl/bin/python -m safe_ctde_mace.scripts.train_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --episodes 3000 \
  --device cuda \
  --num-envs 4 \
  --artifact-dir artifacts/qmix_large_train_3000 \
  --output checkpoints/qmix_large_3000.pt
```

**训练输出摘要**：
- 总episodes: 3000
- 最终checkpoint: `checkpoints/qmix_large_3000.pt`
- 训练历史: `artifacts/qmix_large_train_3000/train_history.csv`

**评估结果（10 episodes）**：
```
episodes=10 coverage_mean=0.900 success_rate=0.800 episode_length_mean=86.2
last_trace plateau_step=81 first_hover_step=21 first_collision_step=21
first_planner_failure_step=None max_zero_gain_streak=1 planner_failures=0
physical_links_mean=2.94 effective_links_mean=2.94
```

| 指标 | 值 | 说明 |
|------|------|------|
| `coverage_mean` | 0.900 | 达到90%覆盖率目标 |
| `success_rate` | 0.800 | 10次中8次成功 |
| `episode_length_mean` | 86.2 | 平均步数 |
| `repeated_coverage_ratio` | 0.94-0.99 | 仍然较高（未显著改善） |
| `planner_failures` | 0 | 规划器无失败 |

### 9.3 问题回答：泛化性

**使用固定场景训练出来的策略在新场景下能否保持成功率？**

答案：**目前训练不具有泛化性**，原因：
1. 当前训练采用固定地图 `[20,20,8]`、固定初始位置、固定障碍物分布
2. QMIX 策略网络学习的是从**特定起点**到**特定障碍物布局**的最优动作映射
3. 网络权重编码了对该特定地图的覆盖路径偏好
4. 如果障碍物位置完全改变，原本的路径可能不再可达或不再最优

**提升泛化性的方法（未来工作）**：
- 课程学习：从小地图到大地图，逐渐增加难度
- 域随机化：训练时随机化障碍物位置、地图大小、无人机数量
- 更通用的状态表示：使用遮挡地图而非绝对坐标

### 9.4 结论与下一轮建议

**本次改进效果评估**：
- ✅ 轨迹平滑性参数已增强（迭代次数、约束投影）
- ✅ 候选多样性机制已添加（空间分块、独占性评分）
- ⚠️ `repeated_coverage_ratio` 仍然较高（0.94-0.99），未能显著降低
- ⚠️ `success_rate=0.800` 未达到 >= 0.90 目标

**下一轮修改建议**：
1. 进一步调整 `w_repeat` 和 `w_overlap` 权重，尝试更激进的重复惩罚
2. 考虑增加 `spatial_block_grid` 的分块数量（从5x5x5减少到更细粒度）
3. 尝试降低 `late_reassign_min_coverage` 到 0.50，使其更早触发重分配
4. 考虑添加 agent 之间的直接通信信息交换机制

## 10. 2026-05-29 GAT通信机制 + 域随机化改进

### 10.1 本次改进内容

**GAT通信机制修复（graph_attention.py）**：
- `GraphAttentionLayer.forward`：使用 edge_index 构建稀疏注意力，正确融合相对位置编码
- `AgentIntentPredictor.forward`：调用 `build_communication_edges` 构建通信边
- `frontier_detector.py`：`neighbor_intent_penalty` 方法正确接收 `candidate_idx` 参数（第260-262行）

**域随机化增强（voxel_world.py）**：
- 新增 `randomize_obstacles_domain` 方法：对所有障碍物应用统一随机平移，保持结构增加多样性

**配置文件更新（qmix_ego_large.yaml）**：
```yaml
environment:
  randomize_obstacles: true
  obstacle_jitter_mode: "global_shift"
  obstacle_shift_range: 5
  gat_hidden_dim: 64
  gat_num_heads: 2
```

### 10.2 2500 episodes训练结果

**训练命令**：
```bash
/home/jude/anaconda3/envs/uav_rl/bin/python -m safe_ctde_mace.scripts.train_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --episodes 2500 \
  --device cuda \
  --num-envs 4 \
  --artifact-dir artifacts/qmix_large_train_2500 \
  --output checkpoints/qmix_large_2500.pt
```

**训练结果**：
- 总episodes: 2500
- 最终checkpoint: `checkpoints/qmix_large_2500.pt`
- 训练历史: `artifacts/qmix_large_train_2500/train_history.csv`

**评估结果（10 episodes）**：
```
episodes=10 coverage_mean=0.891 success_rate=0.600 episode_length_mean=94.5
last_trace plateau_step=100 first_hover_step=35 first_collision_step=35
first_planner_failure_step=None max_zero_gain_streak=2 planner_failures=0
physical_links_mean=2.95 effective_links_mean=2.95
```

| 指标 | 值 | 说明 |
|------|------|------|
| `coverage_mean` | 0.891 | 接近90%目标 |
| `success_rate` | 0.600 | 10次中6次成功 |
| `episode_length_mean` | 94.5 | 平均步数 |
| `repeated_coverage_ratio` | ~0.95-0.99 | 仍然较高 |
| `planner_failures` | 0 | 规划器无失败 |

### 10.3 课程学习设计方案

详见 `Safe-CTDE-MACE/course.md`：

| 阶段 | 地图尺寸 | 障碍物 | 目标覆盖率 | 预期episodes |
|------|----------|--------|------------|--------------|
| 1 Easy | 10×10×4 | 无 | 0.60 | 500 |
| 2 Medium | 15×15×6 | 少 | 0.75 | 800 |
| 3 Hard | 20×20×8 | 当前 | 0.85 | 1000 |
| 4 Generalization | 20×20×8 | 域随机化 | 0.90 | 1500 |

**注意**：当前课程学习设计存在维度不兼容问题（见11.3节），建议暂不启用课程学习。

## 11. 2026-05-30 改进记录

### 11.1 本次改进内容

**问题诊断**：
- 2500 episodes训练后 `success_rate=0.600`（目标>=0.90）
- `repeated_coverage_ratio=0.95-0.99`，仍然较高
- 域随机化和GAT通信机制增加了训练不稳定性

**改进措施**：

#### 1. 禁用域随机化
**文件**: `safe_ctde_mace/configs/qmix_ego_large.yaml`

```yaml
# 第46行已修改
randomize_obstacles: false  # true → false
# obstacle_jitter_mode: "global_shift"  # 已注释
# obstacle_shift_range: 5  # 已注释
```

**原因**：域随机化导致训练不稳定，每次reset时障碍物位置随机偏移，使得观测空间不稳定。

#### 2. GAT正确集成到QNetwork
**文件**: `safe_ctde_mace/marl/networks.py`

- QNetwork新增GAT支持：
  - `use_gat_attention=True`（默认启用）
  - `gat_hidden_dim=64`, `gat_num_heads=2`
- forward方法接收 `neighbor_features` 和 `edge_index` 参数
- GAT输出通过GraphAttentionLayer聚合邻居信息

**文件**: `safe_ctde_mace/marl/qmix.py`

- QMIXAgent从training_config读取GAT配置
- train_step方法处理neighbor_obs，传递给QNetwork

#### 3. 修复_predict_neighbor_intents中的Bug
**文件**: `safe_ctde_mace/envs/multi_uav_env.py`（第617-628行）

**原错误代码**：
```python
for agent in self.agents:
    # ...
    feats = self.current_candidates[0].features  # 始终使用第一个agent的特征！
```

**修复后**：
```python
for i, agent in enumerate(self.agents):
    # ...
    if i < len(self.current_candidates):
        feats = self.current_candidates[i].features  # 使用当前agent自己的特征
```

### 11.2 课程学习评估

**问题发现**：`course.md` 中的课程学习设计存在以下问题：

1. **文档自相矛盾**：`course.md` 中存在两个互相冲突的课程设计版本（章节2.1和2.2），且配置示例中 `num_frontier_candidates` 不一致（Stage 1 用 8，其他阶段用 12）

2. **维度不兼容问题未彻底解决**：
   - Stage 1/2 使用 10×10×4 地图
   - Stage 3 使用 15×15×6 地图
   - Stage 4 使用 20×20×8 地图
   - 每个阶段的观测维度不同，网络权重无法直接迁移

3. **建议**：暂不启用课程学习，继续使用当前的单一配置（20×20×8）进行训练，直到GAT和协调机制稳定后再考虑课程学习。

### 11.3 改进效果预期

| 改进项 | 预期效果 |
|--------|----------|
| 禁用域随机化 | 消除训练不稳定性，成功率应提升 |
| GAT正确集成 | 减少重复覆盖，协同更高效 |
| Bug修复 | 每个agent使用自己的候选特征，意图预测更准确 |

### 11.4 验收标准

训练后应达到：
- `success_rate >= 0.90`
- `coverage >= 0.90`
- `repeated_coverage_ratio < 0.50`（最终目标 < 0.30）