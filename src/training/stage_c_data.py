"""Stage C bridge-curriculum dataset.

Loads T-condition trajectory files (output of `scripts/run_text_bridge_baseline.py`)
and yields per-tick training tuples for the COCONUT-style latent-bridge curriculum.

Each `Stage C` sample answers the question: "given this fast-model context, can the
*latent* representation that the slow model produced replace the *text* it emitted?"

Schema yielded per `__getitem__`:
  - frame:                  uint8 [H, W, 3]    Atari frame the fast model sees
  - game:                   str
  - action:                 int                global 18-way action that was taken
  - legal_action_mask:      bool [18]          game's legal-action set
  - slow_text:              Optional[str]      the text emission active at this tick
                                                (None if no emission yet at this tick)
  - slow_vecs:              Optional[Tensor]   [L, D_bridge] latent vectors for this
                                                emission (None if no emission)
  - emission_age_seconds:   float              wall-clock age of the active emission
                                                (0 if no emission)

Important: trajectory files are produced once on GPU and cached on disk. Stage C
training reloads only the (frame, text, vecs) tuples — no slow-model inference
required during Stage C, only the fast model is in the training loop.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.imitation_data import legal_action_mask


class StageCDataset(Dataset):
    """A flat-indexed view over T-condition trace files.

    Each sample corresponds to one fast-tick. The slow emission active at that tick
    is the most-recent prior emission (i.e., the same one used in `run_text_bridge_baseline.py`'s
    `latest_slow_text` state).
    """

    def __init__(self, trace_paths: Iterable[str | os.PathLike]):
        self.trace_paths = [str(p) for p in trace_paths]
        if not self.trace_paths:
            raise ValueError("no trace paths supplied")
        self._files: list[dict] = []  # full loaded trace blobs
        self._flat_index: list[tuple[int, int]] = []  # (file_idx, tick_idx_in_trace)
        for fi, p in enumerate(self.trace_paths):
            blob = torch.load(p, weights_only=False)
            self._files.append(blob)
            for ti in range(len(blob["trace"])):
                self._flat_index.append((fi, ti))

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> dict:
        fi, ti = self._flat_index[idx]
        blob = self._files[fi]
        game: str = blob["game"]
        trace = blob["trace"]
        item = trace[ti]

        # Find the active emission: walk backward to find the most-recent slow_text
        active_text = None
        active_vecs: Optional[torch.Tensor] = None
        active_age_ticks = 0
        for back in range(ti, -1, -1):
            if trace[back]["slow_text"] is not None:
                active_text = trace[back]["slow_text"]
                active_vecs = trace[back]["slow_vecs"]
                active_age_ticks = ti - back
                break

        return {
            "frame": torch.from_numpy(np.asarray(item["obs"])),
            "game": game,
            "action_global": torch.tensor(int(item["action"]), dtype=torch.long),
            "legal_action_mask": legal_action_mask(game),
            "slow_text": active_text,
            "slow_vecs": active_vecs,  # [L, D_bridge] float32 on CPU, or None
            "emission_age_seconds": float(active_age_ticks) / 15.0,
        }


def load_from_glob(pattern: str) -> StageCDataset:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r}")
    return StageCDataset(paths)
