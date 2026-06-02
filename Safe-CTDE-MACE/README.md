# Safe-CTDE-MACE

Safe-CTDE-MACE is a runnable research framework for multi-UAV 3D coverage exploration in voxel worlds.

The repository now keeps two experiment tracks side by side:

- **Shared DQN + A\***: verified baseline for reproducibility and debugging
- **QMIX + Python EGO-style planner**: centralized training, decentralized execution, and continuous local trajectories

## Features

- Static 3D voxel worlds with manual and random box obstacles
- Per-UAV local knowledge maps with limited spherical sensing
- Distance-limited communication graph and neighbor-only map fusion
- Frontier clustering with fixed candidate sets and invalid-action masks
- Engineering safety shield with reserved CBF-QP extension point
- Baseline 3D A* planning and continuous EGO-style local trajectories
- Shared DQN and QMIX training branches
- Readable 3D visualization for obstacle voxels, covered shells, frontier markers, UAV trajectories, and GIF replays
- Planner comparison tooling with trajectory smoothness diagnostics

## Setup

```powershell
conda activate Safe-CTDE-MACE
pip install -r requirement.txt
```

## Quick Start

Run all tests:

```powershell
python -m pytest -q
```

Train the verified DQN baseline:

```powershell
python -m safe_ctde_mace.scripts.train `
  --config safe_ctde_mace/configs/verified_baseline.yaml `
  --episodes 30 `
  --artifact-dir artifacts/verified_train `
  --output checkpoints/shared_dqn_final.pt
```

Evaluate the DQN baseline:

```powershell
python -m safe_ctde_mace.scripts.evaluate `
  --config safe_ctde_mace/configs/verified_baseline.yaml `
  --checkpoint checkpoints/shared_dqn_final.pt `
  --episodes 5 `
  --artifact-dir artifacts/verified_eval
```

Train QMIX with the continuous planner:

```powershell
python -m safe_ctde_mace.scripts.train_qmix `
  --config safe_ctde_mace/configs/qmix_ego.yaml `
  --episodes 30 `
  --artifact-dir artifacts/qmix_train `
  --output checkpoints/qmix_final.pt
```

Evaluate QMIX:

```powershell
python -m safe_ctde_mace.scripts.evaluate_qmix `
  --config safe_ctde_mace/configs/qmix_ego.yaml `
  --checkpoint checkpoints/qmix_final.pt `
  --episodes 5 `
  --artifact-dir artifacts/qmix_eval
```

Train the larger three-UAV QMIX experiment:

```powershell
python -m safe_ctde_mace.scripts.train_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --episodes 500 `
  --artifact-dir artifacts/qmix_large_train `
  --output checkpoints/qmix_large_final.pt
```

Evaluate the larger experiment:

```powershell
python -m safe_ctde_mace.scripts.evaluate_qmix `
  --config safe_ctde_mace/configs/qmix_ego_large.yaml `
  --checkpoint checkpoints/qmix_large_final.pt `
  --episodes 5 `
  --artifact-dir artifacts/qmix_large_eval
```

Compare A* and EGO-style planning:

```powershell
python -m safe_ctde_mace.scripts.compare_planners `
  --config safe_ctde_mace/configs/verified_baseline.yaml `
  --steps 20 `
  --artifact-dir artifacts/verified_planner_comparison
```

## Main Configurations

- `verified_baseline.yaml`
  - smaller reproducible DQN+A* benchmark
- `qmix_ego.yaml`
  - compact QMIX + continuous planner regression experiment
- `qmix_ego_large.yaml`
  - larger 3-UAV QMIX + EGO-style experiment for longer training runs
- `default_config.yaml`
  - larger research-oriented environment

## Core Interfaces

- `MultiUAVCoverageEnv.reset()`
- `MultiUAVCoverageEnv.step(actions)`
- `MultiUAVCoverageEnv.get_global_state()`
- `QMIXAgent.select_actions(...)`
- `QMIXAgent.train_step(...)`
- `EGOStylePlanner.plan(...)`
- `ContinuousTrajectory.sample(...)`

Per-UAV observations contain:

- `local_voxel_map`
- `self_state`
- `neighbor_states`
- `coverage_ratio`
- `candidate_features`
- `action_mask`

`info` now also reports:

- `trajectory_lengths`
- `mean_acceleration`
- `max_acceleration`
- `smoothness_cost`
- `planner_type`

Visualization artifacts now include:

- `evaluation_episode.png` / `last_evaluation_episode.png`
- `evaluation_replay.gif` / `last_evaluation_replay.gif`
- planner comparison `*_episode.png` and `*_episode.gif`

## Current Scope

Implemented:

- Shared DQN baseline
- QMIX training branch
- Deterministic communication
- Static obstacles
- Spherical sensing
- Engineering safety screening
- 3D A* baseline planner
- Python EGO-style continuous planner
- Static 3D visualization, GIF replay rendering, and planner comparison outputs

Still reserved for later work:

- CBF-QP optimization
- Probabilistic communication
- Dynamic obstacles
- 3D CNN observation encoder
- MADDPG / MASAC
- More faithful kinodynamic optimization

For a Chinese walkthrough of the architecture, workflows, diagnostics, and recommended experiment order, see `Guide.md`.
