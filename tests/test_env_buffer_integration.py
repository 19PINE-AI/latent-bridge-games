"""CPU-only integration test: AtariEnv + ring buffers wired together as the eval loop will.

Verifies:
  - AtariEnv steps at 15Hz and emits a text_state every 15 ticks (1Hz).
  - ThoughtBuffer can be appended to from the env's text-state cadence and read at
    every fast tick.
  - PerceptionBuffer accepts a write every 7 ticks without crashing.

Runs without CUDA and finishes in <5s.
"""
from __future__ import annotations

import pytest
import torch

# Skip cleanly if ALE isn't installed on this machine.
gym = pytest.importorskip("gymnasium")
pytest.importorskip("ale_py")

from src.env.atari_wrapper import AtariEnv
from src.bridge import ThoughtBuffer, PerceptionBuffer


BRIDGE_DIM = 256


@pytest.fixture(scope="module")
def cpu_buffers():
    return (
        ThoughtBuffer(capacity=16, dim=BRIDGE_DIM, device="cpu", dtype=torch.float32),
        PerceptionBuffer(capacity=8, dim=BRIDGE_DIM, device="cpu", dtype=torch.float32),
    )


def test_env_text_state_cadence():
    env = AtariEnv(game_name="MsPacman", seed=0)
    obs, text0 = env.reset()
    assert text0 is not None, "reset should return an initial text state"
    text_indices = []
    for t in range(40):
        _, _, _, _, text = env.step(0)
        if text is not None:
            text_indices.append(t)
    env.close()
    assert text_indices == [14, 29], (
        "text_state should fire at fast-tick indices 14, 29 (every 15 ticks, 0-based)"
        f"; got {text_indices}"
    )


def test_buffers_integrate_with_env_loop(cpu_buffers):
    thought_buf, perception_buf = cpu_buffers

    env = AtariEnv(game_name="MsPacman", seed=1)
    obs, text0 = env.reset()
    # Simulate the slow-emission write
    thought_buf.append(
        torch.randn(BRIDGE_DIM, dtype=torch.float32), timestamp=0.0
    )

    tick_dt = 1.0 / 15.0
    for t in range(45):
        _, _, _, _, text = env.step(0)
        wall_time = (t + 1) * tick_dt

        # On every text-state emission, write to thought buffer
        if text is not None:
            v = torch.randn(BRIDGE_DIM, dtype=torch.float32)
            thought_buf.append(v, timestamp=wall_time)

        # Every 7 ticks, write a perception summary (cadence from configs/bridge.yaml)
        if (t + 1) % 7 == 0:
            p = torch.randn(BRIDGE_DIM, dtype=torch.float32)
            perception_buf.append(p, timestamp=wall_time)

        # Every tick: fast model would read both buffers
        entries, ages = thought_buf.read(current_time=wall_time)
        assert entries.shape == (16, BRIDGE_DIM)
        assert ages.shape == (16,)

    env.close()

    # After 45 ticks we have: initial reset write + emissions at t=14, t=29, t=44 = 4
    assert thought_buf.size == 4
    # And 6 perception entries at t=7,14,21,28,35,42 (every 7 ticks, 1-based)
    assert perception_buf.size == 6


def test_age_encoding_runs_on_cpu(cpu_buffers):
    thought_buf, _ = cpu_buffers
    _, ages = thought_buf.read(current_time=100.0)
    enc = thought_buf.age_encoding(ages, n_freq=8)
    assert enc.shape == (16, BRIDGE_DIM)
    assert enc.dtype == torch.float32
