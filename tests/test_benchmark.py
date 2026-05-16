"""CPU-side tests for the eval harness helpers (the GPU-bound _run_episode is
exercised in integration runs, not unit-tested)."""
from __future__ import annotations

import pytest
import yaml

from src.eval.benchmark import _aggregate, _percentile, TICK_BUDGET_MS


def test_percentile_handles_empty():
    assert _percentile([], 50) == 0.0
    assert _percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_aggregate_groups_by_strategy_game():
    cells = [
        {"strategy": "F", "game": "MsPacman", "seed": 0, "score": 100.0,
         "ticks": 500, "mean_action_latency_ms": 200.0, "frac_on_clock": 0.0},
        {"strategy": "F", "game": "MsPacman", "seed": 1, "score": 200.0,
         "ticks": 600, "mean_action_latency_ms": 210.0, "frac_on_clock": 0.0},
        {"strategy": "T", "game": "MsPacman", "seed": 0, "score": 50.0,
         "ticks": 400, "mean_action_latency_ms": 250.0, "frac_on_clock": 0.0},
        {"strategy": "F", "game": "Frostbite", "seed": 0, "score": 30.0,
         "ticks": 300, "mean_action_latency_ms": 190.0, "frac_on_clock": 0.0},
    ]
    summary = _aggregate(cells)
    assert set(summary.keys()) == {"F/MsPacman", "T/MsPacman", "F/Frostbite"}
    assert summary["F/MsPacman"]["n_episodes"] == 2
    assert summary["F/MsPacman"]["mean_score"] == 150.0
    assert summary["T/MsPacman"]["n_episodes"] == 1
    assert summary["T/MsPacman"]["mean_score"] == 50.0


def test_tick_budget_is_15hz_target():
    # 1000 / 15 ≈ 66.67 ms
    assert abs(TICK_BUDGET_MS - 1000/15) < 1.0


def test_eval_config_loads_and_has_expected_keys():
    import pathlib
    cfg_path = pathlib.Path("configs/eval.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    assert "games" in cfg
    assert "strategies" in cfg
    assert "seeds" in cfg
    assert "episodes_per_cell" in cfg
    # Tier structure
    assert set(cfg["games"].keys()) == {"tier_1_low", "tier_2_medium", "tier_3_high"}
    for tier_games in cfg["games"].values():
        assert isinstance(tier_games, list)
        assert len(tier_games) >= 2
