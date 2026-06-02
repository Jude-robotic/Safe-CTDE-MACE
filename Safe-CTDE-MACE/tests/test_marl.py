from copy import deepcopy

import numpy as np
import pytest
import torch

from safe_ctde_mace.marl.device import resolve_device
from safe_ctde_mace.marl.networks import flatten_observation, observation_dim
from safe_ctde_mace.marl.networks import QMixer
from safe_ctde_mace.marl.qmix import QMIXAgent
from safe_ctde_mace.marl.replay_buffer import JointReplayBuffer, ReplayBuffer
from safe_ctde_mace.marl.shared_dqn import SharedDQNAgent
from safe_ctde_mace.marl.trainer import QMIXTrainer, Trainer
from safe_ctde_mace.utils.config import load_config


def _fake_observation() -> dict[str, np.ndarray]:
    return {
        "local_voxel_map": np.zeros((7, 3, 3, 3), dtype=np.float32),
        "self_state": np.zeros(9, dtype=np.float32),
        "neighbor_states": np.zeros((1, 9), dtype=np.float32),
        "coverage_ratio": np.zeros(1, dtype=np.float32),
        "candidate_features": np.zeros((3, 10), dtype=np.float32),
        "action_mask": np.asarray([False, True, False]),
    }


def _tiny_training_config() -> dict:
    config = deepcopy(load_config())
    config["environment"]["grid_size"] = [8, 8, 4]
    config["environment"]["num_uavs"] = 2
    config["environment"]["initial_positions"] = [[1, 1, 1], [1, 6, 1]]
    config["environment"]["max_neighbors"] = 1
    config["environment"]["sensor_range"] = 1.8
    config["environment"]["comm_range"] = 10.0
    config["environment"]["local_patch_radius"] = 2
    config["environment"]["num_frontier_candidates"] = 4
    config["environment"]["max_steps"] = 2
    config["environment"]["target_coverage_ratio"] = 0.99
    config["environment"]["obstacle_generation"] = {
        "random_boxes": 0,
        "min_box_size": [1, 1, 1],
        "max_box_size": [1, 1, 1],
        "manual_boxes": [],
    }
    config["training"]["batch_size"] = 2
    config["training"]["warmup_steps"] = 2
    config["training"]["replay_capacity"] = 32
    config["training"]["target_update_interval"] = 1
    config["training"]["save_interval"] = 0
    config["training"]["hidden_dim"] = 32
    return config


def test_flatten_observation_dimension() -> None:
    observation = _fake_observation()
    assert observation_dim(observation) == flatten_observation(observation).shape[0]


def test_replay_buffer_sampling_shapes() -> None:
    buffer = ReplayBuffer(capacity=4)
    obs = flatten_observation(_fake_observation())
    mask = np.asarray([False, True, False])
    buffer.add(obs, 1, 1.0, obs, False, mask, mask)
    batch = buffer.sample(1)
    assert batch["obs"].shape == (1, obs.shape[0])
    assert batch["actions"].tolist() == [1]


def test_shared_dqn_respects_single_valid_action_mask() -> None:
    observation = _fake_observation()
    agent = SharedDQNAgent(
        obs_dim=observation_dim(observation),
        num_actions=3,
        training_config={
            "learning_rate": 0.001,
            "gamma": 0.99,
            "target_update_interval": 1,
            "epsilon_start": 1.0,
            "epsilon_end": 1.0,
            "epsilon_decay_steps": 1,
            "hidden_dim": 16,
        },
        device="cpu",
    )
    assert {agent.select_action(observation, explore=True) for _ in range(5)} == {1}


def test_shared_dqn_train_step_runs() -> None:
    observation = _fake_observation()
    obs = flatten_observation(observation)
    mask = observation["action_mask"]
    agent = SharedDQNAgent(
        obs_dim=obs.shape[0],
        num_actions=3,
        training_config={
            "learning_rate": 0.001,
            "gamma": 0.99,
            "target_update_interval": 1,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 1,
            "hidden_dim": 16,
        },
        device="cpu",
    )
    batch = {
        "obs": np.stack([obs, obs]),
        "actions": np.asarray([1, 1]),
        "rewards": np.asarray([1.0, 0.5], dtype=np.float32),
        "next_obs": np.stack([obs, obs]),
        "dones": np.asarray([0.0, 1.0], dtype=np.float32),
        "action_masks": np.stack([mask, mask]),
        "next_action_masks": np.stack([mask, mask]),
    }
    assert agent.train_step(batch) >= 0.0


def test_trainer_short_episode_runs() -> None:
    trainer = Trainer(_tiny_training_config())
    history = trainer.train(num_episodes=1)
    assert len(history) == 1
    assert history[0].episode_length <= 2
    assert trainer.last_trace is not None
    assert history[0].termination_reason in {"coverage_target", "all_failed", "max_steps"}


def test_joint_replay_buffer_sampling_shapes() -> None:
    buffer = JointReplayBuffer(capacity=4)
    obs = np.zeros((2, 5), dtype=np.float32)
    state = np.zeros(7, dtype=np.float32)
    masks = np.ones((2, 3), dtype=bool)
    buffer.add(obs, np.asarray([0, 1]), state, 1.0, obs, state, np.zeros(2), False, masks, masks)
    batch = buffer.sample(1)
    assert batch["obs"].shape == (1, 2, 5)
    assert batch["states"].shape == (1, 7)
    assert batch["actions"].shape == (1, 2)


def test_qmixer_outputs_batch_total_q() -> None:
    mixer = QMixer(num_agents=2, state_dim=5, hidden_dim=8, hypernet_hidden_dim=8)
    total_q = mixer(
        agent_qs=torch.tensor([[1.0, 2.0], [0.5, -0.5]], dtype=torch.float32),
        states=torch.zeros((2, 5), dtype=torch.float32),
    )
    assert tuple(total_q.shape) == (2,)


def test_qmix_agent_respects_single_valid_action_mask() -> None:
    observation = _fake_observation()
    agent = QMIXAgent(
        obs_dim=observation_dim(observation),
        state_dim=6,
        num_agents=2,
        num_actions=3,
        training_config={
            "learning_rate": 0.001,
            "gamma": 0.99,
            "target_update_interval": 1,
            "epsilon_start": 1.0,
            "epsilon_end": 1.0,
            "epsilon_decay_steps": 1,
            "hidden_dim": 16,
            "mixer_hidden_dim": 8,
            "hypernet_hidden_dim": 8,
        },
        device="cpu",
    )
    actions = agent.select_actions([observation, observation], explore=True)
    assert actions == [1, 1]


def test_qmix_agent_batches_greedy_action_selection() -> None:
    observation = _fake_observation()
    agent = QMIXAgent(
        obs_dim=observation_dim(observation),
        state_dim=6,
        num_agents=2,
        num_actions=3,
        training_config={
            "learning_rate": 0.001,
            "gamma": 0.99,
            "target_update_interval": 1,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 1,
            "hidden_dim": 16,
            "mixer_hidden_dim": 8,
            "hypernet_hidden_dim": 8,
        },
        device="cpu",
    )
    calls = 0
    original_forward = agent.q_network.forward

    def counted_forward(inputs: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_forward(inputs)

    agent.q_network.forward = counted_forward  # type: ignore[method-assign]
    actions = agent.select_actions([observation, observation], explore=False)

    assert actions == [1, 1]
    assert calls == 1


def test_qmix_agent_batches_multiple_environments() -> None:
    observation = _fake_observation()
    agent = QMIXAgent(
        obs_dim=observation_dim(observation),
        state_dim=6,
        num_agents=2,
        num_actions=3,
        training_config={
            "learning_rate": 0.001,
            "gamma": 0.99,
            "target_update_interval": 1,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 1,
            "hidden_dim": 16,
            "mixer_hidden_dim": 8,
            "hypernet_hidden_dim": 8,
        },
        device="cpu",
    )
    actions = agent.select_actions_batch(
        [[observation, observation], [observation, observation]],
        explore=False,
    )

    assert actions == [[1, 1], [1, 1]]


def test_qmix_train_step_runs() -> None:
    observation = _fake_observation()
    obs = flatten_observation(observation)
    mask = observation["action_mask"]
    agent = QMIXAgent(
        obs_dim=obs.shape[0],
        state_dim=6,
        num_agents=2,
        num_actions=3,
        training_config={
            "learning_rate": 0.001,
            "gamma": 0.99,
            "target_update_interval": 1,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 1,
            "hidden_dim": 16,
            "mixer_hidden_dim": 8,
            "hypernet_hidden_dim": 8,
        },
        device="cpu",
    )
    batch = {
        "obs": np.stack([[obs, obs], [obs, obs]]),
        "actions": np.asarray([[1, 1], [1, 1]]),
        "states": np.zeros((2, 6), dtype=np.float32),
        "rewards": np.asarray([1.0, 0.5], dtype=np.float32),
        "next_obs": np.stack([[obs, obs], [obs, obs]]),
        "next_states": np.zeros((2, 6), dtype=np.float32),
        "agent_dones": np.zeros((2, 2), dtype=np.float32),
        "dones": np.asarray([0.0, 1.0], dtype=np.float32),
        "action_masks": np.stack([[mask, mask], [mask, mask]]),
        "next_action_masks": np.stack([[mask, mask], [mask, mask]]),
    }
    assert agent.train_step(batch) >= 0.0


def test_qmix_checkpoint_round_trip_preserves_matching_architecture(tmp_path) -> None:
    observation = _fake_observation()
    config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "target_update_interval": 1,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 1,
        "hidden_dim": 16,
        "mixer_hidden_dim": 8,
        "hypernet_hidden_dim": 8,
    }
    source = QMIXAgent(observation_dim(observation), 6, 2, 3, config, device="cpu")
    destination = QMIXAgent(observation_dim(observation), 6, 2, 3, config, device="cpu")
    checkpoint = tmp_path / "qmix.pt"

    source.save(checkpoint)
    destination.load(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["metadata"] == {
        "obs_dim": observation_dim(observation),
        "state_dim": 6,
        "num_agents": 2,
        "num_actions": 3,
        "feature_schema_version": 2,
    }


def test_qmix_checkpoint_rejects_mismatched_architecture(tmp_path) -> None:
    observation = _fake_observation()
    config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "target_update_interval": 1,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 1,
        "hidden_dim": 16,
        "mixer_hidden_dim": 8,
        "hypernet_hidden_dim": 8,
    }
    source = QMIXAgent(observation_dim(observation), 6, 2, 3, config, device="cpu")
    destination = QMIXAgent(observation_dim(observation) + 1, 7, 3, 4, config, device="cpu")
    checkpoint = tmp_path / "qmix.pt"
    source.save(checkpoint)

    with pytest.raises(ValueError, match="obs_dim.*state_dim.*num_agents.*num_actions"):
        destination.load(checkpoint)


def test_qmix_checkpoint_rejects_legacy_feature_schema(tmp_path) -> None:
    observation = _fake_observation()
    config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "target_update_interval": 1,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 1,
        "hidden_dim": 16,
        "mixer_hidden_dim": 8,
        "hypernet_hidden_dim": 8,
    }
    source = QMIXAgent(observation_dim(observation), 6, 2, 3, config, device="cpu")
    destination = QMIXAgent(observation_dim(observation), 6, 2, 3, config, device="cpu")
    checkpoint = tmp_path / "qmix.pt"
    source.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    payload["metadata"].pop("feature_schema_version")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="feature_schema_version"):
        destination.load(checkpoint)


def test_qmix_trainer_short_episode_runs() -> None:
    trainer = QMIXTrainer(_tiny_training_config())
    history = trainer.train(num_episodes=1)
    assert len(history) == 1
    assert history[0].episode_length <= 2


def test_qmix_trainer_advances_training_seed_per_serial_episode() -> None:
    config = _tiny_training_config()
    config["training"]["num_envs"] = 1
    trainer = QMIXTrainer(config)
    trainer.train(num_episodes=2)

    assert trainer.env.seed == int(config["seed"]) + 1


def test_qmix_trainer_evaluates_explicit_seed_list() -> None:
    trainer = QMIXTrainer(_tiny_training_config())
    history = trainer.evaluate(seeds=[11, 12], capture_replay=False)

    assert len(history) == 2
    assert trainer.env.seed == 12


def test_trainer_replay_capture_is_optional() -> None:
    trainer = Trainer(_tiny_training_config())
    trainer.run_episode(training=False, capture_replay=False)
    assert trainer.last_replay is None

    trainer.run_episode(training=False, capture_replay=True)
    assert trainer.last_replay is not None
    assert trainer.last_trace is not None
    assert len(trainer.last_replay.frames) == len(trainer.last_trace.coverage_curve)


def test_qmix_trainer_collects_replays_for_evaluation() -> None:
    trainer = QMIXTrainer(_tiny_training_config())
    trainer.evaluate(episodes=1, capture_replay=True)

    assert len(trainer.evaluation_replays) == 1
    assert trainer.evaluation_replays[0].frames


def test_qmix_trainer_parallel_rollout_runs() -> None:
    config = _tiny_training_config()
    config["training"]["num_envs"] = 2
    history = QMIXTrainer(config).train(num_episodes=3)

    assert len(history) == 3
    assert all(item.episode_length <= 2 for item in history)


def test_resolve_device_cpu_and_auto() -> None:
    assert str(resolve_device("cpu")) == "cpu"
    assert str(resolve_device("auto")) in {"cpu", "cuda"}


def test_resolve_device_rejects_invalid_value() -> None:
    with np.testing.assert_raises(ValueError):
        resolve_device("tpu")
