"""Atari (ALE) environment wrapper with tick-rate downsampling and text-state extraction.

Provides:
- 15Hz fast-model observations (every 4th 60Hz frame)
- 1Hz slow-model text snapshots (entity positions + score + level), decoded from ALE RAM
- Standard Gymnasium step interface

RAM-decoder sources:
- AtariARI (Anand et al., NeurIPS 2019, mila-iqia/atari-representation-learning)
- OCAtari (Delfosse et al., RLC 2024, k4ntz/OC_Atari)
- ALE source (Farama-Foundation/Arcade-Learning-Environment, src/ale/games/supported/*.cpp)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextState:
    """Compact textual snapshot of the game state for the slow model."""
    frame_idx: int
    score: int
    lives: int
    level: int
    # Game-specific entity positions populated per-game
    entities: dict


def _bcd(byte: int) -> int:
    """Decode a packed-BCD byte: high nibble = tens, low nibble = units."""
    return ((byte >> 4) & 0xF) * 10 + (byte & 0xF)


def _decode_mspacman_ram(ram: np.ndarray) -> dict:
    """Return (score, lives, entities) extracted from Ms. Pac-Man RAM.

    Sources: AtariARI ram_annotations.py (mspacman dict), ALE MsPacman.cpp.
    Score is BCD across bytes 120-122 stored LSB-first (verified empirically against
    the dots-eaten counter at ram[119]): ram[120] is units/tens, ram[121] is
    hundreds/thousands, ram[122] is ten-thousands/hundred-thousands. The AtariARI README
    documents these as `player_score=(120,121,122)` without specifying byte order; ALE's
    own MsPacman.cpp calls `getDecimalScore(0xF8, 0xF9, 0xFA)` which iterates LSB→MSB.
    Each dot = 10 points; the BCD digits store the displayed score directly (no ×10).
    Lives encoded as (byte_123 & 0x7) + 1 in ALE source.
    Entity (x,y) positions in sprite-grid units, not raw pixels (AtariARI issue #64 caveat).
    """
    score = (
        _bcd(int(ram[122])) * 10000
        + _bcd(int(ram[121])) * 100
        + _bcd(int(ram[120]))
    )
    lives = (int(ram[123]) & 0x7) + 1
    return {
        "score": score,
        "lives": lives,
        "level": 0,  # not exposed in RAM; maintained by caller via dots-reset counter
        "entities": {
            "pacman_xy": (int(ram[10]), int(ram[16])),
            "ghost_sue_xy":    (int(ram[6]),  int(ram[12])),
            "ghost_inky_xy":   (int(ram[7]),  int(ram[13])),
            "ghost_pinky_xy":  (int(ram[8]),  int(ram[14])),
            "ghost_blinky_xy": (int(ram[9]),  int(ram[15])),
            "fruit_xy": (int(ram[11]), int(ram[17])),
            "ghosts_on_board": int(ram[19]),  # decrements as ghosts are eaten in scared phase
            "dots_eaten": int(ram[119]),
            "player_direction": int(ram[56]),
        },
    }


# Ice-floe row → screen-y mapping for Frostbite, per OCAtari (top y first).
_FROSTBITE_FLOE_ROW_Y = (96, 122, 148, 174)
# Hazard slot type codes from OCAtari ocatari/ram/frostbite.py.
_FROSTBITE_HAZARD_CODE = {
    range(18, 19): "bird", range(26, 27): "bird",
    range(34, 39): "crab", range(50, 55): "crab",
    range(66, 71): "clam", range(94, 99): "clam",
    range(109, 114): "green_fish", range(124, 129): "green_fish",
}


def _hazard_kind(code: int) -> Optional[str]:
    for code_range, name in _FROSTBITE_HAZARD_CODE.items():
        if code in code_range:
            return name
    return None


def _decode_frostbite_ram(ram: np.ndarray) -> dict:
    """Return decoded Frostbite text-state.

    Sources: AtariARI, OCAtari frostbite.py. Score BCD at bytes 72/73/74, LSB-first as
    verified empirically (a +10 reward increments ram[74] from 0x00→0x10, ram[73]/72
    only increment on rollover). Lives at 76.
    Igloo build progress 0-15 at byte 77 (255 = reset). Temperature at byte 101.
    Player (x,y) at (102, 100). Ice-floe row x-positions at bytes 31-34 (bottom→top is
    OCAtari ordering; we expose top→bottom by reversing). Hazard slot i: type at 35+i,
    x at 84+i; row index i. Polar bear x at byte 104 (140 = off-screen).
    Day/night not exposed in RAM — caller may infer from frame or igloo-completion counter.
    """
    score = (
        _bcd(int(ram[72])) * 10000
        + _bcd(int(ram[73])) * 100
        + _bcd(int(ram[74]))
    )
    lives = int(ram[76])
    igloo_raw = int(ram[77])
    igloo = 0 if igloo_raw == 255 else min(igloo_raw, 15)
    temperature = int(ram[101])
    player_xy = (int(ram[102]), int(ram[100]))
    # OCAtari stores rows bottom-up at bytes 31..34. Reverse for top→bottom presentation.
    floe_xs_top_to_bottom = [int(ram[34]), int(ram[33]), int(ram[32]), int(ram[31])]
    hazards = []
    for i in range(4):
        kind = _hazard_kind(int(ram[35 + i]))
        if kind is None:
            continue
        hazards.append({
            "kind": kind,
            "row_y": _FROSTBITE_FLOE_ROW_Y[3 - i],  # row i is bottom-up; flip to top-down y
            "x": int(ram[84 + i]),
            "double": bool(int(ram[88 + i])),
        })
    bear_x = int(ram[104])
    return {
        "score": score,
        "lives": lives,
        "level": 0,  # day/night not in RAM; left at 0
        "entities": {
            "player_xy": player_xy,
            "igloo_progress": igloo,        # 0-15; 15 = complete
            "temperature": temperature,
            "ice_floe_xs_top_to_bottom": floe_xs_top_to_bottom,
            "hazards": hazards,
            "polar_bear_x": None if bear_x == 140 else bear_x,
            "sinking": int(ram[106]) == 26,
        },
    }


def _decode_seaquest_ram(ram: np.ndarray) -> dict:
    """Return decoded Seaquest text-state.

    Sources: AtariARI, OCAtari seaquest.py, ALE Seaquest.cpp.

    Score bytes are at 56/57/58. **Unlike MsPacman/Frostbite, Seaquest is BCD-MSB-first**:
    ram[56] is the most-significant byte (10000s+1000s digits packed as BCD),
    ram[57] is hundreds+tens, ram[58] is the ones digit. So:
        score = bcd(ram[56]) * 1000 + bcd(ram[57]) * 10 + bcd(ram[58])
    ALE source uses `getDecimalScore(0xBA, 0xB9, 0xB8)` with MSB-first arg ordering.
    OCAtari computes `ram[56]*100 + ram[57]*100 + ram[58]` (note: their formula contains
    a typo of *100 instead of *10000/*100/*1 — we use the corrected formula above).

    Lives = ram[59] + 1 (per ALE source).

    Strategically critical fields (this is why Seaquest is a Tier-3 game):
      - **oxygen**: ram[102] = pixel-width 0..~64 of the oxygen bar. <~16 = critical.
      - **divers_collected**: ram[62] = integer 0..6. Surfacing with 6 gives bonus.
      - **depth**: ram[97] = player y (raw). Surface when small.
    """
    score = _bcd(int(ram[56])) * 1000 + _bcd(int(ram[57])) * 10 + _bcd(int(ram[58]))
    lives = int(ram[59]) + 1
    oxygen = int(ram[102])
    divers = int(ram[62])
    player_x = int(ram[70])
    player_y = int(ram[97])
    facing_east = (int(ram[86]) == 0)
    own_torpedo_x = int(ram[103])
    crash_state = int(ram[105])  # 0 < x < 15 means alive
    surface_enemy_active = int(ram[60]) >= 2

    # Enemy/diver lanes: 4 lanes at y-rows beneath the player.
    enemy_lane_xs = [int(ram[30 + i]) for i in range(4)]
    # Slot table 71..74 holds either divers or enemy missiles (shared positions).
    slot_xs = [int(ram[71 + i]) for i in range(4)]

    return {
        "score": score,
        "lives": lives,
        "level": int(ram[61]),
        "entities": {
            "oxygen": oxygen,
            "oxygen_critical": oxygen < 16,
            "divers_collected": divers,
            "submarine_full": divers >= 6,
            "player_xy": (player_x, player_y),
            "facing_east": facing_east,
            "own_torpedo_x": None if own_torpedo_x == 0 else own_torpedo_x,
            "enemy_lane_xs": enemy_lane_xs,   # bottom-up
            "slot_xs": slot_xs,               # divers OR enemy missiles
            "alive": 0 < crash_state < 15,
            "surface_enemy_x": int(ram[118]) if surface_enemy_active else None,
        },
    }


def _decode_spaceinvaders_ram(ram: np.ndarray) -> dict:
    """Return decoded SpaceInvaders text-state.

    Sources: AtariARI, OCAtari spaceinvaders.py, ALE SpaceInvaders.cpp.

    Score encoding (verified via ALE `getDecimalScore(0xE8, 0xE6, &system)`):
      ram[104] = low BCD pair (ones/tens digits)
      ram[102] = high BCD pair (hundreds/thousands)
      → score = bcd(ram[102]) * 100 + bcd(ram[104])   (max 9999, wraps)

    Lives encoded raw at ram[73] (per ALE `readRam(0xC9)`).

    Strategic fields:
      - invaders_left ram[17]: 0..36 invaders alive
      - per-row alive bitfields ram[18..23]: which columns survive in each row
      - formation x/y at ram[26], ram[24]
      - 2 enemy bombs at (ram[83], ram[81]) and (ram[84], ram[82])
      - player missile at (ram[87], ram[85])
    """
    score = _bcd(int(ram[102])) * 100 + _bcd(int(ram[104]))
    lives = int(ram[73])
    cannon_x = int(ram[28])
    form_x, form_y = int(ram[26]), int(ram[24])
    invaders_left = int(ram[17])
    row_bitfields = [int(ram[18 + i]) & 0x3F for i in range(6)]
    saucer_x = int(ram[30])
    saucer_active = saucer_x != 0 and saucer_x < 160
    player_missile_x = int(ram[87])
    player_missile_y = int(ram[85])
    player_missile_active = player_missile_y != 0
    bombs = []
    for slot in range(2):
        by = int(ram[81 + slot])
        bx = int(ram[83 + slot])
        if by != 0:
            bombs.append({"x": bx, "y": by, "slot": slot})
    return {
        "score": score,
        "lives": lives,
        "level": 0,  # not exposed
        "entities": {
            "cannon_x": cannon_x,
            "invaders_left": invaders_left,
            "row_bitfields_top_to_bottom": row_bitfields,
            "formation_xy": (form_x, form_y),
            "player_missile_xy": (player_missile_x, player_missile_y) if player_missile_active else None,
            "enemy_bombs": bombs,
            "saucer_x": saucer_x if saucer_active else None,
        },
    }


def _decode_pong_ram(ram: np.ndarray) -> dict:
    """Pong RAM decoder.

    Sources: AtariARI ram_annotations.py (pong dict) and OCAtari/ocatari/ram/pong.py.
    Pong is the simplest decoder: two paddles + one ball + two scores. Used as the
    Tier-1 control game (predicted: L ~ T because fast model alone saturates).

    Bytes (from AtariARI):
      player_y    = ram[51]   (our paddle, right side)
      enemy_y     = ram[50]   (CPU paddle, left side)
      ball_x      = ram[49]
      ball_y      = ram[54]
      player_score= ram[14]   (single byte, 0..21)
      enemy_score = ram[13]
    """
    player_y = int(ram[51])
    enemy_y = int(ram[50])
    ball_x = int(ram[49])
    ball_y = int(ram[54])
    player_score = int(ram[14])
    enemy_score = int(ram[13])
    # Net score (player − enemy); used by the score-vs-reward regression test
    return {
        "score": player_score - enemy_score,
        "lives": 1,  # Pong has no lives
        "level": 0,
        "entities": {
            "player_paddle_y": player_y,
            "enemy_paddle_y": enemy_y,
            "ball_xy": (ball_x, ball_y),
            "player_score": player_score,
            "enemy_score": enemy_score,
            # Convenience derived fields
            "ball_relative_to_player": ball_y - player_y,
            "ball_relative_to_enemy": ball_y - enemy_y,
        },
    }


_RAM_DECODERS = {
    "MsPacman": _decode_mspacman_ram,
    "Frostbite": _decode_frostbite_ram,
    "Seaquest": _decode_seaquest_ram,
    "SpaceInvaders": _decode_spaceinvaders_ram,
    "Pong": _decode_pong_ram,
}


class AtariEnv:
    """ALE wrapper. Falls back gracefully if gymnasium[atari] isn't installed."""

    FAST_TICK_EVERY_N_FRAMES = 4   # 60Hz → 15Hz
    TEXT_STATE_EVERY_N_TICKS = 15  # 15Hz → 1Hz

    def __init__(self, game_name: str = "MsPacman", render: bool = False, seed: int = 0):
        try:
            import gymnasium as gym
            import ale_py  # noqa: F401  -- registers ALE namespace
        except ImportError as e:
            raise ImportError(
                "gymnasium[atari] and ale-py are required. "
                "Install with `pip install 'gymnasium[atari]' ale-py`."
            ) from e

        self.env = gym.make(
            f"ALE/{game_name}-v5",
            render_mode="rgb_array" if render else None,
            frameskip=1,  # we handle our own downsampling
        )
        self.env.reset(seed=seed)
        self._frame_idx = 0
        self._tick_idx = 0
        self._last_obs = None
        self.game_name = game_name

    def reset(self) -> tuple[np.ndarray, TextState]:
        obs, _ = self.env.reset()
        self._frame_idx = 0
        self._tick_idx = 0
        self._last_obs = obs
        return obs, self._text_state(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, TextState]:
        """Steps the env by 4 frames, returning the 4th frame as the next fast-model obs."""
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self.FAST_TICK_EVERY_N_FRAMES):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += float(reward)
            self._frame_idx += 1
            if terminated or truncated:
                break
        self._tick_idx += 1
        self._last_obs = obs
        text_state = self._text_state(obs) if self._tick_idx % self.TEXT_STATE_EVERY_N_TICKS == 0 else None
        return obs, total_reward, terminated, truncated, text_state

    def _text_state(self, obs: np.ndarray) -> TextState:
        """Extract a compact text snapshot via ALE RAM.

        Decoders are registered in `_RAM_DECODERS` by game name; games without a
        registered decoder get an empty entities dict (score/lives still 0).
        """
        decoder = _RAM_DECODERS.get(self.game_name)
        if decoder is None:
            return TextState(
                frame_idx=self._frame_idx, score=0, lives=0, level=0, entities={}
            )
        ram = self.env.unwrapped.ale.getRAM()
        decoded = decoder(ram)
        return TextState(
            frame_idx=self._frame_idx,
            score=decoded["score"],
            lives=decoded["lives"],
            level=decoded["level"],
            entities=decoded["entities"],
        )

    @property
    def action_space_size(self) -> int:
        return int(self.env.action_space.n)

    def close(self):
        self.env.close()
