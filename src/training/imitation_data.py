"""Stage A imitation-learning dataset.

Loads trajectory .pt files emitted by `scripts/collect_trajectories.py` and yields
(frame, action_global, action_legal_mask) tuples suitable for training the fast model's
18-way classification head.

Design notes:
- All games share a single 18-way ALE-canonical action space. Per-game action indices
  (which differ — MsPacman has 9, Pong has 6, Pitfall/Frostbite have 18) are remapped
  to the global space using `GAME_ACTION_TO_GLOBAL[game]`.
- An `action_legal_mask` is emitted so the fast model can mask logits during sampling
  to the game's legal subset (necessary at eval, optional during training — masking
  out illegal actions during cross-entropy is equivalent because the labels are always
  legal).
- Frames are loaded as uint8 HWC at native ALE resolution (210×160). Resizing to the
  fast model's expected input size is done by the model wrapper, not the dataset, so
  the same .pt files can serve multiple model variants.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


# Global 18-way ALE canonical action space (the ordering used by Frostbite, Pitfall,
# PrivateEye — i.e., the full action space).
GLOBAL_ACTION_NAMES = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
)
N_GLOBAL_ACTIONS = 18

# Per-game action mapping: local_idx -> global_idx. Derived from gymnasium ALE
# `get_action_meanings()` at registration time; recorded here so the dataset does not
# require an ALE roundtrip per sample.
_GAME_ACTION_NAMES = {
    "MsPacman": ("NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"),
    "Frostbite": GLOBAL_ACTION_NAMES,
    "Seaquest": GLOBAL_ACTION_NAMES,
    "Pong": ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"),
    "Breakout": ("NOOP", "FIRE", "RIGHT", "LEFT"),
    "BeamRider": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "UPRIGHT", "UPLEFT", "RIGHTFIRE", "LEFTFIRE"),
    "SpaceInvaders": ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"),
    "Qbert": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"),
    "Pitfall": GLOBAL_ACTION_NAMES,
    "PrivateEye": GLOBAL_ACTION_NAMES,
    "Riverraid": GLOBAL_ACTION_NAMES,    # 18-action ALE-canonical space
    "RiverRaid": GLOBAL_ACTION_NAMES,
    "Berzerk": GLOBAL_ACTION_NAMES,       # 18-action ALE-canonical space
    "RoadRunner": GLOBAL_ACTION_NAMES,    # 18-action ALE-canonical space
    "Roadrunner": GLOBAL_ACTION_NAMES,
}

_NAME_TO_GLOBAL = {name: i for i, name in enumerate(GLOBAL_ACTION_NAMES)}

GAME_ACTION_TO_GLOBAL: dict[str, tuple[int, ...]] = {
    game: tuple(_NAME_TO_GLOBAL[name] for name in names)
    for game, names in _GAME_ACTION_NAMES.items()
}

# Inverse mapping: global_idx -> local_idx for each game, with -1 for illegal globals.
GLOBAL_ACTION_TO_GAME: dict[str, tuple[int, ...]] = {}
for _game, _g2l in GAME_ACTION_TO_GLOBAL.items():
    _inv = [-1] * N_GLOBAL_ACTIONS
    for _local, _global in enumerate(_g2l):
        _inv[_global] = _local
    GLOBAL_ACTION_TO_GAME[_game] = tuple(_inv)


def global_to_local_action(game: str, global_action: int) -> int:
    """Map a 0-17 global ALE-canonical action index to the game's local action index.

    Raises ValueError if the global action is illegal for this game.
    """
    local = GLOBAL_ACTION_TO_GAME[game][global_action]
    if local < 0:
        raise ValueError(f"global action {global_action} is illegal for {game!r}")
    return local


def legal_action_mask(game: str) -> torch.Tensor:
    """Return a [18] bool mask: True at global indices that are legal for `game`."""
    if game not in GAME_ACTION_TO_GLOBAL:
        raise ValueError(f"no action mapping for game {game!r}")
    mask = torch.zeros(N_GLOBAL_ACTIONS, dtype=torch.bool)
    for global_idx in GAME_ACTION_TO_GLOBAL[game]:
        mask[global_idx] = True
    return mask


class ImitationDataset(Dataset):
    """Yields (frame_uint8_HWC, action_global, action_legal_mask, reward) tuples.

    `frame_paths` is a list of trajectory .pt files (output of collect_trajectories.py).
    The dataset flat-indexes across all episodes in all files.
    """

    def __init__(self, traj_paths: Iterable[str | os.PathLike]):
        self.traj_paths = [str(p) for p in traj_paths]
        if not self.traj_paths:
            raise ValueError("no trajectory paths supplied")
        # Lazy-load: build (file_idx, ep_idx, tick_idx) flat index without loading
        # all frame data into memory.
        self._flat_index: list[tuple[int, int, int]] = []
        self._file_meta: list[dict] = []
        for fi, p in enumerate(self.traj_paths):
            blob = torch.load(p, weights_only=False)
            game = blob["game"]
            if game not in GAME_ACTION_TO_GLOBAL:
                raise ValueError(f"unknown game {game!r} in {p}")
            self._file_meta.append({"path": p, "game": game, "blob": blob})
            for ei, ep in enumerate(blob["episodes"]):
                T = len(ep["action"])
                for ti in range(T):
                    self._flat_index.append((fi, ei, ti))

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> dict:
        fi, ei, ti = self._flat_index[idx]
        meta = self._file_meta[fi]
        game = meta["game"]
        ep = meta["blob"]["episodes"][ei]
        local_a = int(ep["action"][ti])
        global_a = GAME_ACTION_TO_GLOBAL[game][local_a]
        frame = ep["obs"][ti]  # uint8 HWC
        return {
            "frame": torch.from_numpy(frame),         # uint8 [H, W, 3]
            "action_global": torch.tensor(global_a, dtype=torch.long),
            "action_legal_mask": legal_action_mask(game),
            "reward": torch.tensor(float(ep["reward"][ti]), dtype=torch.float32),
            "game": game,
        }


def load_dataset_from_glob(pattern: str) -> ImitationDataset:
    """Convenience: load all trajectory .pt files matching `pattern`."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r}")
    return ImitationDataset(paths)
