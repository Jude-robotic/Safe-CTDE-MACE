# Safe-Ego-Planner

## Legal Disclaimer

**This software is provided for academic research and educational purposes only.**

- **NON-COMMERCIAL USE ONLY:** This project is open-sourced under a non-commercial license. You may not use this software for commercial purposes, profit-generating activities, or any use that generates revenue.
- **NO WARRANTY:** This software is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.
- **LIABILITY:** In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
- **ATTRIBUTION REQUIRED:** If you use this software or adapt any part of the codebase in your research, you must acknowledge and credit the authors appropriately in your publications.

If you are interested in commercial licensing, please contact the authors separately.

---

## 1. Project Overview

**Safe-Ego-Planner** is a multi-UAV 3D coverage exploration system that combines centralized training with distributed execution (CTDE). It uses reinforcement learning (QMIX) with an EGO-style continuous trajectory planner to coordinate multiple drones in unknown 3D environments, with safety constraints, limited sensing ranges, and communication constraints.

The system consists of two major components:

| Component | Description |
|-----------|-------------|
| **Safe-CTDE-MACE** | Python simulation framework: RL training, coverage exploration, EGO-style trajectory optimization, and visualization |
| **Safe-Ego-Planner (ROS)** | ROS Noetic workspace: exports Python exploration results to ROS, converts trajectories to B-splines, and executes real quadrotor dynamics with SO(3) control in RVIZ |

### Key Features

- **Centralized Training, Distributed Execution (CTDE):** Uses QMIX for multi-agent RL training with a global state mixer, while execution relies only on local observation.
- **EGO-Style Continuous Trajectory Planning:** Generates smooth, minimum-snap trajectories with collision avoidance in continuous 3D space.
- **Safety Shield:** Filters unreachable or conflicting candidate frontiers before trajectory planning.
- **Cooperative Exploration with Limited Communication:** Neighbors share maps and coverage state; a late-stage reassignment mechanism reduces redundant exploration.
- **Mine Corridor Environment:** Supports fixed 3D mine tunnel scenes with line-of-sight sensor occlusion.
- **Full ROS Integration:** Exports trained policies to ROS for real-dynamics simulation and RVIZ visualization.

### Target Performance

- Success rate ≥ 90% (achieving 90% coverage in a 20×20×8 voxel environment)
- Coverage ratio ≥ 90%
- Collision count near zero

---

## 2. Architecture

### 2.1 Python Training Pipeline (Safe-CTDE-MACE)

```
QMIX Agent (Centralized Training, Distributed Execution)
  -> Candidate Frontier Selection (with candidate features)
  -> SafetyShield Filtering
  -> Late-Stage Reassignment & Goal Deconfliction
  -> EGO-Style Planner (A* seed path + continuous trajectory optimization)
  -> TrajectoryTracker (executes one step)
  -> Update local maps, fuse neighbor maps, update coverage statistics
  -> Compute reward and termination
```

**Key modules:**

| Module | File | Role |
|--------|------|------|
| Environment | `envs/multi_uav_env.py` | Main environment loop, reward, termination, coordination |
| Voxel World | `envs/voxel_world.py` | 3D voxel world, obstacle generation |
| Coverage Map | `mapping/coverage_map.py` | Voxel states, sensing updates, coverage statistics |
| Frontier Detector | `mapping/frontier_detector.py` | Frontier detection, candidate generation, feature computation |
| QMIX Agent | `marl/qmix.py` | QMIX agent, target network, mixer, checkpoint |
| EGO Planner | `planning/ego_planner.py` | A* seed path + minimum-snap trajectory optimization |
| Communication | `communication/comm_graph.py`, `map_fusion.py` | Communication graph and neighbor map fusion |
| Safety Shield | `planning/safety_shield.py` | Filter unsafe/unreachable/conflicting goals |

### 2.2 ROS Execution Pipeline

```
Python export_ros_eval.py (export episode as JSON)
  -> ros_eval_episode.json (map, obstacles, trajectories)
  -> map_generator (build point cloud from JSON boxes)
  -> ros_eval_playback.py (publish /uav{i}/python_traj as nav_msgs/Path)
  -> traj_bridge_node (convert Path to ego_planner/Bspline)
  -> traj_server (publish position commands)
  -> so3_control + quadrotor_simulator_so3 (real SO(3) dynamics)
  -> /uav{i}/sim/odom (RVIZ visualization and metrics logging)
```

**Key ROS nodes:**

| Node | Package | Role |
|------|---------|------|
| `map_generator` | `map_generator` | Read JSON obstacle boxes, publish global point cloud |
| `ros_eval_playback` | `traj_bridge` | Publish Python trajectories, markers, coverage ratio |
| `traj_bridge_node` | `traj_bridge` | Convert `nav_msgs/Path` to `ego_planner/Bspline` |
| `traj_server` | `ego_planner` | Publish position commands from B-spline |
| `quadrotor_simulator_so3` | `uav_simulator` | SO(3) quadrotor dynamics simulator |
| `so3_control` | `uav_simulator` | SO(3) attitude controller |

---

## 3. Installation

### 3.1 Python Environment (Safe-CTDE-MACE)

```bash
# Create and activate the conda environment
conda create -n uav_rl -y scipy pyyaml gymnasium tqdm matplotlib pillow
conda activate uav_rl

# Install PyTorch (CPU or CUDA depending on your hardware)
# CPU:  pip install torch
# CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy>=1.26 scipy>=1.13 pyyaml>=6.0 gymnasium>=1.1 matplotlib>=3.8 pillow>=10.0 tqdm>=4.66 pytest>=7.4 torch>=2.5

# Set PYTHONPATH
cd /path/to/Safe_ego_planner/Safe-CTDE-MACE
export PYTHONPATH=/path/to/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH
```

### 3.2 ROS Environment (Safe-Ego-Planner)

Requires **ROS Noetic**.

```bash
cd /path/to/Safe_ego_planner
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

**Key ROS packages:**

| Package | Path | Description |
|---------|------|-------------|
| `ego_planner` | `src/planner/plan_manage/` | Main planning node, launch files |
| `traj_bridge` | `src/uav_simulator/traj_bridge/` | Trajectory bridge between Python and ROS |
| `map_generator` | `src/uav_simulator/map_generator/` | Voxel obstacle to point cloud generator |
| `uav_simulator` | `src/uav_simulator/` | SO(3) simulator, control, visualization |

---

## 4. Quick Start

### 4.1 Health Check

```bash
cd /path/to/Safe_ego_planner/Safe-CTDE-MACE
conda activate uav_rl
export PYTHONPATH=/path/to/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH

# Check CUDA availability
python -m safe_ctde_mace.scripts.check_cuda --device cuda

# Run core unit tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_config.py tests/test_ros_export.py -q
```

### 4.2 Training (Large Scene QMIX + EGO)

```bash
cd /path/to/Safe_ego_planner/Safe-CTDE-MACE
conda activate uav_rl
export PYTHONPATH=/path/to/Safe_ego_planner/Safe-CTDE-MACE:$PYTHONPATH

python -m safe_ctde_mace.scripts.train_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --episodes 2000 \
  --device cuda \
  --num-envs 4 \
  --artifact-dir artifacts/qmix_large_train \
  --output checkpoints/qmix_large_final.pt
```

### 4.3 Evaluation

```bash
python -m safe_ctde_mace.scripts.evaluate_qmix \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --checkpoint checkpoints/qmix_large_final.pt \
  --device cuda \
  --seed-count 10 \
  --artifact-dir artifacts/qmix_large_eval
```

### 4.4 ROS Export & Visualization

```bash
# 1. Export episode to ROS JSON
python -m safe_ctde_mace.scripts.export_ros_eval \
  --config safe_ctde_mace/configs/qmix_ego_large.yaml \
  --checkpoint checkpoints/qmix_large_final.pt \
  --device cuda \
  --seed 7 \
  --artifact-dir result/ros_eval

# 2. Build ROS workspace
cd /path/to/Safe_ego_planner
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

# 3. Launch with RVIZ
roslaunch ego_planner simple_run.launch \
  episode_json:=/path/to/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval/ros_eval_episode.json \
  metrics_dir:=/path/to/Safe_ego_planner/Safe-CTDE-MACE/result/ros_eval \
  use_rviz:=true
```

---

## 5. Project Structure

```
Safe_ego_planner/
├── Safe-CTDE-MACE/                    # Python simulation framework
│   ├── safe_ctde_mace/
│   │   ├── agents/                     # UAV agent: position, velocity, local map
│   │   ├── communication/              # Comm graph and neighbor map fusion
│   │   ├── configs/                    # YAML configs (qmix_ego_large.yaml, etc.)
│   │   ├── envs/                       # Voxel world, multi-UAV environment
│   │   ├── mapping/                    # Coverage map, frontier detector
│   │   ├── marl/                       # QMIX, Shared DQN, networks, trainer, replay buffer
│   │   ├── planning/                   # Safety shield, A*, EGO planner, trajectory tracker
│   │   ├── scripts/                    # train_qmix, evaluate_qmix, export_ros_eval, etc.
│   │   └── utils/                      # Visualization, reporting, ROS export
│   ├── checkpoints/                    # Trained model checkpoints
│   ├── artifacts/                      # Training artifacts, evaluation results
│   ├── result/                         # ROS export results
│   └── tests/                          # Unit tests
├── src/                                # ROS Noetic workspace
│   ├── planner/plan_manage/
│   │   ├── launch/                     # Launch files (simple_run.launch, etc.)
│   │   └── rviz/                       # RVIZ configuration
│   └── uav_simulator/
│       ├── map_generator/               # Point cloud from JSON obstacle boxes
│       ├── traj_bridge/                 # ROS trajectory bridge (C++ node + Python playback)
│       └── so3_quadrotor_simulator/     # SO(3) quadrotor simulator and controller
├── end_for_paper.md                    # Reproduction guide for paper experiments
├── Guide_ego.md                        # ROS verification guide
└── Guide.md                            # Safe-CTDE-MACE Chinese guide
```

---

## 6. Key Configuration

### Large Scene (20×20×8, 3 UAVs)

File: `Safe-CTDE-MACE/safe_ctde_mace/configs/qmix_ego_large.yaml`

| Parameter | Value |
|-----------|-------|
| Grid size | 20 × 20 × 8 |
| Voxel resolution | 1 m |
| UAV count | 3 |
| Initial positions | [1,1,1], [1,18,1], [18,1,1] |
| Sensor range | 3.5 |
| Communication range | 20.0 |
| Target coverage | 0.90 |
| Max steps | 100 |
| Planner | ego |
| EGO optimize iterations | 150 |
| Global sync interval | 0 (pure physical comm) |
| Domain randomization | false |

### Mine Corridor (50×50×20, 3 UAVs)

File: `Safe-CTDE-MACE/safe_ctde_mace/configs/qmix_ego_mine_50x50x20.yaml`

- 3D mine tunnel with horizontal corridors and sloped connecting tunnels
- Line-of-sight sensor occlusion enabled
- Communication range: 75.0

---

## 7. ROS Data Flow

```
ros_eval_episode.json
  |
  v
map_generator / random_forest_sensing.cpp
  -> /map_generator/global_cloud           (voxel boxes -> point cloud)
  |
  v
ros_eval_playback.py
  -> /uav{i}/python_traj                   (nav_msgs/Path, meter units)
  -> /safe_ctde_mace/uav{i}/python_path_marker
  -> /safe_ctde_mace/coverage_ratio
  |
  v
traj_bridge_node
  -> /uav{i}/planning/bspline             (ego_planner/Bspline)
  |
  v
traj_server
  -> /uav{i}/planning/pos_cmd             (geometry_msgs/PoseStamped)
  |
  v
so3_control + quadrotor_simulator_so3
  -> /uav{i}/sim/odom                      (nav_msgs/Odometry)
  -> /uav{i}/odom_visualization/path       (real executed trajectory)
  -> /uav{i}/odom_visualization/robot       (UAV mesh)
  |
  v
ros_execution_metrics.csv                   (written on shutdown)
```

---

## 8. Coordinate Conventions

- Python voxel center → ROS meter:
  ```
  ros_x = (voxel_x + 0.5) * voxel_resolution
  ros_y = (voxel_y + 0.5) * voxel_resolution
  ros_z = (voxel_z + 0.5) * voxel_resolution
  ```
- Python trajectory points are published in ROS meters via `nav_msgs/Path`.
- Obstacle boxes use closed-interval voxel semantics (inclusive both ends).

---

## 9. Citation

If you use this software in your research, please cite the original paper and acknowledge this repository in your publication.

---

## 10. References

- QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
- EGO-Planner: An ESDF-Free Online Planning Framework for Autonomous Flight
- SO(3) Quadrotor Control: Geometric tracking control of a quadrotor UAV on SE(3)

For detailed experimental procedures and reproduction steps, see `end_for_paper.md`.