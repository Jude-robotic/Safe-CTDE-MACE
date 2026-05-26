# Vibe: Multi-UAV ROS Coverage Exploration Verification

## Mission

Bridge Python QMIX-trained multi-uav coverage exploration (Safe-CTDE-MACE) to ROS, enabling RVIZ visualization and real-robot validation.

## The Problem We're Solving

1. **Planner Mismatch**: Python trains with EGO-style planner (A* seed + gradient descent), ROS executes with BsplineOptimizer (A* + L-BFGS). Same state → different trajectories. QMIX策略无法直接在ROS验证.

2. **Environment Mismatch**: Python uses voxel world (20×20×8, 0.5m resolution), ROS uses continuous space with random cylindrical obstacles. Training env ≠ testing env.

## Core Insight

Don't try to run QMIX in ROS. Instead:
- **Python side**: Runs QMIX policy, makes decisions, generates trajectories via EGO-style planner
- **ROS side**: Executes trajectories, provides RVIZ visualization, safety backup

This preserves training→execution fidelity while enabling real-world validation.

## Architecture

```
Python (Safe-CTDE-MACE)                    ROS (ego_planner)
┌─────────────────────────┐                ┌─────────────────────────┐
│ QMIX → EGO-planner      │──traj bridge──▶│ BsplineExecutor          │
│ MultiUAVCoverageEnv    │   (topic)      │ + RVIZ viz               │
│ MapFusion (comm)       │                │ + SafetyMonitor          │
└─────────────────────────┘                └─────────────────────────┘
         │                                           ▲
         │  same obstacles                          │
         └───────────────────────────────────────────┘
                    (voxel_world ≡ random_forest_sensing)
```

## File Changes Required

### 1. `src/uav_simulator/map_generator/src/random_forest_sensing.cpp`

Replace cylindrical obstacle generation with voxel box obstacles matching `voxel_world.py`:
- Parse obstacle centers/sizes from ROS params (same as Python yaml)
- Generate axis-aligned box point clouds (not cylinders/ellipses)
- Add coverage tracker publishing voxel coverage state to ROS topic

### 2. New: `src/uav_simulator/traj_bridge/src/traj_bridge_node.cpp`

ROS node that:
- Subscribes to `/uav{i}/python_traj` (nav_msgs::Path from Python)
- Converts Python trajectory points → UniformBspline
- Publishes to `/planning/bspline` for execution
- Monitors trajectory safety, falls back to native planner if invalid

### 3. `src/planner/plan_manage/launch/simple_run.launch`

Multi-uav configuration:
- 3x traj_server instances with namespace `/uav1`, `/uav2`, `/uav3`
- 3x ego_replan_fsm instances
- Shared obstacle map (all uavs see same environment)
- Python traj bridge subscribers

### 4. Python side (Safe-CTDE-MACE)

In `multi_uav_env.py` step():
- After trajectory generation, publish to ROS topic per agent
- Use `rospy.Publisher` with `nav_msgs/Path` message
- Topics: `/uav1/python_traj`, `/uav2/python_traj`, `/uav3/python_traj`

## Environment Alignment Checklist

- Map size: 20×20×8 voxels (Python) → 10m×10m×4m continuous (ROS, 0.5m/voxel scale)
- Resolution: 0.5m/voxel
- Obstacle count: 8 random boxes + 1 center box (from Python config)
- Sensor range: 3.5 units (matches Python `sensor_range=3.5`)
- Initial positions: (-5,-5), (5,-5), (0,5) in continuous space

## Success Criteria

1. RVIZ shows 3 UAVs exploring 3D space with obstacles
2. Python-side coverage% matches ROS-side coverage%
3. Trajectory smoothness:jerk < threshold
4. ROS execution time < 2x Python simulation time

## Key Principle

**Verify, don't retrain.** The QMIX policy is already trained. This project validates it runs correctly in ROS. No fine-tuning, no on-robot sim2real transfer—just faithful reproduction of trained behavior in a new visualization layer.

## When Stuck

Ask: "Am I trying to make ROS think like Python, or just display Python's decisions?" Prefer the latter.