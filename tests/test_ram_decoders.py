"""Sanity tests for the RAM decoders.

These tests run ALE for a few real frames and check that the decoded text-state
is structurally well-formed and contains plausible values. Exact byte values
depend on the ROM-side state machine so we test invariants rather than literals.
"""
from __future__ import annotations

import pytest
import numpy as np

gym = pytest.importorskip("gymnasium")
pytest.importorskip("ale_py")

from src.env.atari_wrapper import AtariEnv, _decode_mspacman_ram, _decode_frostbite_ram


def _ram_after_n_steps(game: str, n: int, action: int = 0) -> np.ndarray:
    env = AtariEnv(game_name=game, seed=42)
    env.reset()
    for _ in range(n):
        env.step(action)
    ram = env.env.unwrapped.ale.getRAM()
    env.close()
    return np.array(ram, dtype=np.uint8)


def test_mspacman_decoder_structure():
    ram = _ram_after_n_steps("MsPacman", n=20)
    decoded = _decode_mspacman_ram(ram)

    assert set(decoded.keys()) == {"score", "lives", "level", "entities"}
    assert decoded["score"] % 10 == 0, "Ms. Pac-Man scores are multiples of 10"
    assert 0 <= decoded["score"] <= 999_990
    # Lives at game start: usually 2-3 lives in standard ALE config
    assert 1 <= decoded["lives"] <= 8

    e = decoded["entities"]
    assert e["pacman_xy"] != (0, 0), "player should not be at origin after 20 ticks"
    for ghost in ("ghost_sue_xy", "ghost_inky_xy", "ghost_pinky_xy", "ghost_blinky_xy"):
        x, y = e[ghost]
        assert 0 <= x < 256 and 0 <= y < 256
    assert 0 <= e["ghosts_on_board"] <= 4
    assert 0 <= e["dots_eaten"] <= 220


def test_frostbite_decoder_structure():
    ram = _ram_after_n_steps("Frostbite", n=20)
    decoded = _decode_frostbite_ram(ram)

    assert set(decoded.keys()) == {"score", "lives", "level", "entities"}
    assert 0 <= decoded["score"] <= 999_999

    e = decoded["entities"]
    px, py = e["player_xy"]
    assert 0 <= px < 256 and 0 <= py < 256
    assert 0 <= e["igloo_progress"] <= 15
    assert 0 <= e["temperature"] < 256
    assert len(e["ice_floe_xs_top_to_bottom"]) == 4
    for floe_x in e["ice_floe_xs_top_to_bottom"]:
        assert 0 <= floe_x < 256
    # Hazards list may be empty at boot; just check it's a list of well-formed dicts.
    for h in e["hazards"]:
        assert h["kind"] in ("bird", "crab", "clam", "green_fish")
        assert h["row_y"] in (96, 122, 148, 174)
        assert 0 <= h["x"] < 256


def test_text_state_uses_decoder():
    env = AtariEnv(game_name="MsPacman", seed=0)
    env.reset()
    # Step until first 1Hz text emission
    text = None
    for _ in range(40):
        _, _, _, _, t = env.step(0)
        if t is not None:
            text = t
            break
    env.close()
    assert text is not None
    assert "pacman_xy" in text.entities
    assert isinstance(text.score, int)


def test_mspacman_score_matches_reward():
    """Score reported in text-state must equal cumulative reward — guards against the
    LSB/MSB byte-order regression we hit during initial decoder development."""
    env = AtariEnv(game_name="MsPacman", seed=0)
    env.reset()
    cumulative_reward = 0.0
    last_text_score = 0
    for t in range(300):
        _, reward, term, trunc, text = env.step(0)
        cumulative_reward += reward
        if text is not None:
            last_text_score = text.score
        if term or trunc:
            break
    env.close()
    # If RAM-decoded score and ALE cumulative reward disagree, our decoder is wrong.
    assert last_text_score == int(cumulative_reward), (
        f"decoded score {last_text_score} != ALE cumulative reward {cumulative_reward}"
    )


def test_frostbite_score_matches_reward():
    import numpy as np
    env = AtariEnv(game_name="Frostbite", seed=0)
    env.reset()
    rng = np.random.default_rng(1)
    cumulative_reward = 0.0
    last_text_score = 0
    n_actions = env.action_space_size
    for t in range(400):
        a = int(rng.integers(0, n_actions))
        _, reward, term, trunc, text = env.step(a)
        cumulative_reward += reward
        if text is not None:
            last_text_score = text.score
        if term or trunc:
            break
    env.close()
    assert last_text_score == int(cumulative_reward), (
        f"decoded score {last_text_score} != ALE cumulative reward {cumulative_reward}"
    )


def test_seaquest_score_matches_reward():
    import numpy as np
    env = AtariEnv(game_name="Seaquest", seed=42)
    env.reset()
    rng = np.random.default_rng(0)
    cumulative_reward = 0.0
    last_text_score = 0
    n_actions = env.action_space_size
    for t in range(400):
        a = int(rng.integers(0, n_actions))
        _, reward, term, trunc, text = env.step(a)
        cumulative_reward += reward
        if text is not None:
            last_text_score = text.score
        if term or trunc:
            break
    env.close()
    assert last_text_score == int(cumulative_reward), (
        f"Seaquest decoded score {last_text_score} != ALE cumulative reward {cumulative_reward}"
    )


def test_seaquest_decoder_structure():
    env = AtariEnv(game_name="Seaquest", seed=0)
    env.reset()
    text = None
    for _ in range(40):
        _, _, _, _, t = env.step(0)
        if t is not None:
            text = t
            break
    env.close()
    assert text is not None
    e = text.entities
    for k in ("oxygen", "divers_collected", "player_xy", "enemy_lane_xs", "slot_xs",
              "alive", "submarine_full", "oxygen_critical"):
        assert k in e, f"missing key {k}"
    assert 0 <= e["oxygen"] < 256
    assert 0 <= e["divers_collected"] <= 6
    assert len(e["enemy_lane_xs"]) == 4
    assert len(e["slot_xs"]) == 4


def test_spaceinvaders_score_matches_reward():
    import numpy as np
    env = AtariEnv(game_name="SpaceInvaders", seed=42)
    env.reset()
    rng = np.random.default_rng(0)
    cumulative_reward = 0.0
    last_text_score = 0
    n_actions = env.action_space_size
    for t in range(600):
        a = int(rng.integers(0, n_actions))
        _, reward, term, trunc, text = env.step(a)
        cumulative_reward += reward
        if text is not None:
            last_text_score = text.score
        if term or trunc:
            break
    env.close()
    assert last_text_score == int(cumulative_reward), (
        f"SpaceInvaders decoded score {last_text_score} != ALE cum reward {cumulative_reward}"
    )


def test_pong_decoder_structure():
    env = AtariEnv(game_name="Pong", seed=0)
    env.reset()
    text = None
    for _ in range(60):
        _, _, _, _, t = env.step(0)
        if t is not None:
            text = t
            break
    env.close()
    assert text is not None
    e = text.entities
    for k in ("player_paddle_y", "enemy_paddle_y", "ball_xy",
              "player_score", "enemy_score",
              "ball_relative_to_player", "ball_relative_to_enemy"):
        assert k in e, f"missing key {k}"
    bx, by = e["ball_xy"]
    assert 0 <= bx < 256 and 0 <= by < 256
    assert 0 <= e["player_paddle_y"] < 256
    assert 0 <= e["enemy_paddle_y"] < 256


def test_pong_score_matches_reward():
    """Pong score = player_score - enemy_score must equal cumulative ALE reward.
    ALE reward is +1 / -1 per point so the delta is tracked exactly."""
    env = AtariEnv(game_name="Pong", seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    cumulative_reward = 0.0
    last_text_score = 0
    n_actions = env.action_space_size
    for t in range(2000):
        a = int(rng.integers(0, n_actions))
        _, reward, term, trunc, text = env.step(a)
        cumulative_reward += reward
        if text is not None:
            last_text_score = text.score
        if term or trunc:
            break
    env.close()
    assert last_text_score == int(cumulative_reward), (
        f"Pong decoded score {last_text_score} != ALE cum reward {cumulative_reward}"
    )


def test_unknown_game_returns_empty_entities():
    # Breakout has no registered decoder (yet)
    env = AtariEnv(game_name="Breakout", seed=0)
    env.reset()
    text = None
    for _ in range(40):
        _, _, _, _, t = env.step(0)
        if t is not None:
            text = t
            break
    env.close()
    assert text is not None
    assert text.entities == {}
    assert text.score == 0
