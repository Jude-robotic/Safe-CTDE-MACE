import numpy as np

from safe_ctde_mace.envs.voxel_world import VoxelWorld
from safe_ctde_mace.mapping.coverage_map import CoverageMap
from safe_ctde_mace.mapping.voxel_map import VoxelState


def test_voxel_world_manual_obstacle_and_patch() -> None:
    world = VoxelWorld((6, 6, 4), seed=1)
    world.add_box((2, 2, 1), (3, 3, 2))

    assert world.is_obstacle((2, 2, 1))
    assert not world.is_obstacle((0, 0, 0))
    patch = world.get_local_patch((0, 0, 0), radius=1)
    assert patch.shape == (3, 3, 3)
    assert patch[0, 0, 0] == int(VoxelState.OBSTACLE)


def test_sensor_updates_coverage_and_obstacles() -> None:
    world = VoxelWorld((5, 5, 5), seed=1)
    world.add_box((2, 2, 2), (2, 2, 2))
    coverage = CoverageMap(world.grid_size)

    first = coverage.update_from_sensor(world, (2, 2, 1), sensor_range=1.5)
    second = coverage.update_from_sensor(world, (2, 2, 1), sensor_range=1.5)

    assert first.new_covered > 0
    assert first.unknown_reduction > 0
    assert coverage.base_states[2, 2, 2] == int(VoxelState.OBSTACLE)
    assert second.repeated_covered > 0
    assert coverage.coverage_ratio(world) > 0.0


def test_reserved_overlay_does_not_override_covered_or_obstacles() -> None:
    coverage = CoverageMap((4, 4, 4))
    coverage.mark_covered([(1, 1, 1)])
    coverage.mark_obstacle([(2, 2, 2)])
    coverage.mark_free([(0, 0, 0)])
    coverage.reserve([(1, 1, 1), (2, 2, 2), (0, 0, 0)])
    states = coverage.as_array()

    assert states[1, 1, 1] == int(VoxelState.COVERED)
    assert states[2, 2, 2] == int(VoxelState.OBSTACLE)
    assert states[0, 0, 0] == int(VoxelState.RESERVED)


def test_random_obstacles_respect_forbidden_positions() -> None:
    world = VoxelWorld((8, 8, 4), seed=3)
    forbidden = [(0, 0, 0), (1, 1, 1)]
    world.add_random_obstacles(
        count=3,
        min_box_size=(1, 1, 1),
        max_box_size=(2, 2, 2),
        forbidden_positions=forbidden,
    )
    assert all(not world.is_obstacle(position) for position in forbidden)
    assert np.count_nonzero(world.grid == int(VoxelState.OBSTACLE)) > 0

