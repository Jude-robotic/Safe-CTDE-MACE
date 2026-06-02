from safe_ctde_mace import __version__
from safe_ctde_mace.utils.config import load_config


def test_package_import_and_version() -> None:
    assert __version__ == "0.1.0"


def test_default_config_loads() -> None:
    config = load_config()
    assert config["environment"]["grid_size"] == [20, 20, 8]
    assert config["training"]["batch_size"] == 64


def test_verified_baseline_config_loads() -> None:
    config = load_config("safe_ctde_mace/configs/verified_baseline.yaml")
    assert config["environment"]["grid_size"] == [10, 10, 4]
    assert config["environment"]["target_coverage_ratio"] == 0.90


def test_qmix_ego_config_loads() -> None:
    config = load_config("safe_ctde_mace/configs/qmix_ego.yaml")
    assert config["environment"]["num_uavs"] == 2
    assert config["training"]["mixer_hidden_dim"] == 64


def test_qmix_ego_large_config_loads() -> None:
    config = load_config("safe_ctde_mace/configs/qmix_ego_large.yaml")
    assert config["environment"]["num_uavs"] == 3
    assert config["environment"]["planner_type"] == "ego"
    assert config["environment"]["global_sync_interval"] == 0
    assert config["environment"]["max_neighbors"] == 2
    assert config["training"]["num_episodes"] == 500
    assert config["training"]["device"] == "cuda"
    assert config["training"]["num_envs"] == 4
    assert config["training"]["hidden_dim"] == 512
    assert config["training"]["mixer_hidden_dim"] == 128
    assert config["training"]["hypernet_hidden_dim"] == 256
