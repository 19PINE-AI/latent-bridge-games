"""CPU-side tests for Stage C dataset + curriculum mechanics."""
from __future__ import annotations

import pytest
import numpy as np
import torch

from src.training.stage_c_data import StageCDataset
from src.training.stage_c_bridge import _build_thought_buffer, _truncated_text


def _make_fake_trace(tmp_path, n_ticks=30, emit_every=10, n_slow_tokens=50,
                    bridge_dim=256, game="MsPacman"):
    """Build a minimal trace blob mimicking run_text_bridge_baseline.py output."""
    trace = []
    for t in range(n_ticks):
        slow_text = None
        slow_vecs = None
        if (t + 1) % emit_every == 0:
            slow_text = f"head down-left, dodge ghost (tick {t})"
            slow_vecs = torch.randn(n_slow_tokens, bridge_dim, dtype=torch.float32)
        trace.append({
            "tick": t,
            "action": 2,  # UP
            "reward": 0.0,
            "obs": np.zeros((210, 160, 3), dtype=np.uint8),
            "text_state": None,
            "slow_text": slow_text,
            "slow_vecs": slow_vecs,
        })
    blob = {"game": game, "seed": 0, "trace": trace, "cumulative_reward": 0.0}
    p = tmp_path / "fake_trace.pt"
    torch.save(blob, str(p))
    return str(p)


def test_dataset_loads_and_indexes_ticks(tmp_path):
    p = _make_fake_trace(tmp_path, n_ticks=30, emit_every=10)
    ds = StageCDataset([p])
    assert len(ds) == 30
    sample = ds[0]
    assert set(sample.keys()) >= {
        "frame", "game", "action_global", "legal_action_mask",
        "slow_text", "slow_vecs", "emission_age_seconds",
    }
    assert sample["game"] == "MsPacman"
    assert sample["action_global"].item() == 2
    assert sample["frame"].shape == (210, 160, 3)
    assert sample["legal_action_mask"].shape == (18,)


def test_dataset_finds_most_recent_emission(tmp_path):
    p = _make_fake_trace(tmp_path, n_ticks=30, emit_every=10)
    ds = StageCDataset([p])
    # Tick 0: no emission yet (first emission is at tick 9)
    assert ds[0]["slow_text"] is None
    assert ds[0]["slow_vecs"] is None
    # Tick 9: emission landed at this tick
    assert ds[9]["slow_text"] is not None
    assert ds[9]["emission_age_seconds"] == 0.0
    # Tick 15: 6 ticks after the tick-9 emission
    assert ds[15]["slow_text"] == ds[9]["slow_text"]
    assert abs(ds[15]["emission_age_seconds"] - 6/15.0) < 1e-6
    # Tick 19: just after second emission at tick 19
    assert ds[19]["slow_text"] is not None
    assert ds[19]["slow_text"] != ds[9]["slow_text"]


def test_build_thought_buffer_c2_takes_all():
    vecs = torch.randn(50, 256)
    tb = _build_thought_buffer(vecs, stage="C2", capacity=16, dim=256, device="cpu")
    assert tb.shape == (1, 16, 256)
    # Last 16 of the 50 vectors (most-recent wins)
    expected = vecs[-16:].to(dtype=torch.bfloat16)
    assert torch.allclose(tb[0].float(), expected.float(), atol=1e-3)


def test_build_thought_buffer_c1_takes_second_half():
    vecs = torch.randn(40, 256)
    tb = _build_thought_buffer(vecs, stage="C1", capacity=16, dim=256, device="cpu")
    assert tb.shape == (1, 16, 256)
    # C1 = second half = last 20 vecs, then truncated to last 16
    expected = vecs[40 // 2:][-16:].to(dtype=torch.bfloat16)
    assert torch.allclose(tb[0].float(), expected.float(), atol=1e-3)


def test_build_thought_buffer_pads_short_emission():
    # 5 vecs but capacity 16 → first 11 slots should be zero pad
    vecs = torch.ones(5, 256)
    tb = _build_thought_buffer(vecs, stage="C2", capacity=16, dim=256, device="cpu")
    assert tb.shape == (1, 16, 256)
    # First 11 slots padded with zeros
    assert torch.all(tb[0, :11] == 0)
    assert torch.allclose(tb[0, 11:].float(), torch.ones(5, 256), atol=1e-3)


def test_truncated_text_per_stage():
    text = "head down-left to dodge the orange ghost approaching from upper-right"
    assert _truncated_text(text, "C0") == text
    assert _truncated_text(text, "C2") == ""
    truncated = _truncated_text(text, "C1")
    # C1 keeps first half of words
    assert truncated == "head down-left to dodge the"


def test_truncated_text_handles_none():
    # Short-circuit: None passes through for any stage. The train loop skips
    # samples with no active emission, so this branch is never the loss target.
    assert _truncated_text(None, "C0") is None
    assert _truncated_text(None, "C1") is None
    assert _truncated_text(None, "C2") is None
