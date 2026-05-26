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
