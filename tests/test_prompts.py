"""Tests for the slow-model prompt templates (T-condition)."""
from __future__ import annotations

import pytest

from src.env.atari_wrapper import TextState
from src.training.prompts import (
    SYSTEM_PROMPT,
    build_slow_model_messages,
    build_fast_model_context_suffix,
)


def _mspacman_state() -> TextState:
    return TextState(
        frame_idx=300,
        score=30,
        lives=3,
        level=0,
        entities={
            "pacman_xy": (88, 98),
            "ghost_sue_xy": (104, 50),
            "ghost_inky_xy": (94, 78),
            "ghost_pinky_xy": (94, 68),
            "ghost_blinky_xy": (42, 27),
            "fruit_xy": (0, 0),
            "ghosts_on_board": 3,
            "dots_eaten": 3,
            "player_direction": 3,
        },
    )


def _frostbite_state() -> TextState:
    return TextState(
        frame_idx=120,
        score=20,
        lives=3,
        level=0,
        entities={
            "player_xy": (42, 47),
            "igloo_progress": 1,
            "temperature": 65,
            "ice_floe_xs_top_to_bottom": [43, 53, 43, 53],
            "hazards": [
                {"kind": "bird", "row_y": 174, "x": 7, "double": False},
                {"kind": "crab", "row_y": 148, "x": 39, "double": True},
            ],
            "polar_bear_x": None,
            "sinking": False,
        },
    )


def test_mspacman_prompt_includes_all_entities():
    ts = _mspacman_state()
    msgs = build_slow_model_messages("MsPacman", ts)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    user = msgs[-1]["content"]
    assert "Ms. Pac-Man" in user
    assert "Pac-Man at (88, 98)" in user
    assert "Sue (orange) at (104, 50)" in user
    # Delta from Pac-Man for the closest ghost
    assert "(+16, -48)" in user  # Sue Δ = (104-88, 50-98)
    assert "Score: 30" in user
    assert "Dots eaten this board: 3" in user


def test_frostbite_prompt_lists_hazards_and_omits_absent_polar_bear():
    ts = _frostbite_state()
    msgs = build_slow_model_messages("Frostbite", ts)
    user = msgs[-1]["content"]
    assert "Frostbite" in user
    assert "Bailey at (42, 47)" in user
    assert "Temperature: 65" in user
    assert "Igloo: 1/15 blocks" in user
    assert "bird at x=7, row_y=174" in user
    assert "crab at x=39, row_y=148 (double instance)" in user
    assert "POLAR BEAR" not in user
    assert "sinking" not in user.lower()


def test_frostbite_prompt_calls_out_polar_bear_and_sinking():
    ts = _frostbite_state()
    ts.entities["polar_bear_x"] = 50
    ts.entities["sinking"] = True
    user = build_slow_model_messages("Frostbite", ts)[-1]["content"]
    assert "POLAR BEAR on top shore at x=50" in user
    assert "WARNING: Bailey is sinking" in user


def test_prior_thought_threads_in_as_assistant_turn():
    ts = _mspacman_state()
    msgs = build_slow_model_messages("MsPacman", ts, prior_thought="head down-left, dodge Sue")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "assistant", "user"]
    assert msgs[1]["content"] == "head down-left, dodge Sue"


def test_unknown_game_raises():
    ts = _mspacman_state()
    with pytest.raises(ValueError):
        build_slow_model_messages("Pong", ts)


def test_fast_model_context_suffix_is_delimited():
    s = build_fast_model_context_suffix("  head down-left ")
    assert s.startswith("\n[strategic-guidance]:")
    assert "head down-left" in s
    assert not s.endswith(" ")  # trimmed
