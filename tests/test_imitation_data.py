"""Tests for the Stage A imitation dataset + action-space mapping."""
from __future__ import annotations

import pytest
import torch

from src.training.imitation_data import (
    GAME_ACTION_TO_GLOBAL,
    GLOBAL_ACTION_NAMES,
    N_GLOBAL_ACTIONS,
    legal_action_mask,
    ImitationDataset,
)


def test_global_action_space_is_18():
    assert len(GLOBAL_ACTION_NAMES) == N_GLOBAL_ACTIONS == 18


def test_mspacman_mapping_skips_fire():
    # MsPacman doesn't have FIRE in its action set; mapping must skip global index 1.
    mapping = GAME_ACTION_TO_GLOBAL["MsPacman"]
    assert mapping[0] == 0   # NOOP -> NOOP
    assert mapping[1] == 2   # local UP -> global UP (skipped FIRE)
    assert 1 not in mapping  # FIRE never legal in MsPacman


def test_frostbite_uses_full_18_space():
    mapping = GAME_ACTION_TO_GLOBAL["Frostbite"]
    assert mapping == tuple(range(18))


def test_legal_mask_shape_and_content():
    m = legal_action_mask("MsPacman")
    assert m.dtype == torch.bool
    assert m.shape == (18,)
    assert m.sum().item() == 9  # MsPacman has 9 legal actions
    assert m[0].item() is True
    assert m[1].item() is False   # FIRE not legal


@pytest.fixture(scope="module")
def trajectory_file(tmp_path_factory):
    """Run a tiny trajectory collection to produce a real .pt file for the dataset test."""
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("ale_py")
    from src.env.atari_wrapper import AtariEnv

    out = tmp_path_factory.mktemp("traj") / "ms.pt"
    episodes = []
    for ep_seed in range(2):
        env = AtariEnv(game_name="MsPacman", seed=ep_seed)
        obs, _ = env.reset()
        obses, actions, rewards = [], [], []
        import numpy as np
        rng = np.random.default_rng(ep_seed)
        for _ in range(20):
            a = int(rng.integers(0, env.action_space_size))
            next_obs, r, term, trunc, _ = env.step(a)
            obses.append(obs)
            actions.append(a)
            rewards.append(r)
            obs = next_obs
            if term or trunc:
                break
        env.close()
        episodes.append({
            "obs": np.stack(obses, axis=0).astype(np.uint8),
            "action": np.array(actions, dtype=np.int8),
            "reward": np.array(rewards, dtype=np.float32),
            "done": np.zeros(len(actions), dtype=bool),
            "text_idx": np.array([0], dtype=np.int32),
            "text_state": [None],
        })
    torch.save({"game": "MsPacman", "epsilon": 1.0, "episodes": episodes}, str(out))
    return str(out)


def test_dataset_loads_and_remaps_actions(trajectory_file):
    ds = ImitationDataset([trajectory_file])
    assert len(ds) == 40  # 2 episodes × 20 ticks
    sample = ds[0]
    assert set(sample.keys()) == {"frame", "action_global", "action_legal_mask", "reward", "game"}
    assert sample["frame"].dtype == torch.uint8
    assert sample["frame"].shape == (210, 160, 3)
    assert sample["action_global"].dtype == torch.long
    # MsPacman has 9 legal actions, so the global action must be one of those.
    assert sample["action_global"].item() in GAME_ACTION_TO_GLOBAL["MsPacman"]
    assert sample["action_legal_mask"].shape == (18,)
    assert sample["action_legal_mask"][sample["action_global"]].item() is True


def test_dataset_dataloader_compatibility(trajectory_file):
    from torch.utils.data import DataLoader
    ds = ImitationDataset([trajectory_file])
    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=lambda batch: {
                            k: torch.stack([b[k] for b in batch]) if isinstance(batch[0][k], torch.Tensor)
                            else [b[k] for b in batch]
                            for k in batch[0]
                        })
    batch = next(iter(loader))
    assert batch["frame"].shape == (4, 210, 160, 3)
    assert batch["action_global"].shape == (4,)
    assert batch["action_legal_mask"].shape == (4, 18)
