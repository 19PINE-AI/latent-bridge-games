"""MiniGrid environment wrapper — drop-in analog of AtariEnv for the non-Atari
cross-domain experiment.

Why MiniGrid (vs Procgen): MiniGrid exposes the *symbolic* grid directly
(`grid.get(i,j)`, `agent_pos`, `agent_dir`), so (a) the expert is a provably
optimal A* planner — no fragile pixel-parsing, no weak-Stage-A risk — and (b)
the slow model's structured-state prompt is decoded for free, exactly like the
Atari RAM decoders.

Interface mirrors AtariEnv:
  reset(seed) -> (obs_rgb_HWC_uint8, text_state_or_None)
  step(local_action) -> (obs, reward, terminated, truncated, text_state_or_None)

`text_state` is a TextState (see atari_wrapper) emitted at slow cadence
(every `_emit_every` agent steps) carrying the symbolic spatial state.

Action space: MiniGrid native {0:turn_left, 1:turn_right, 2:forward}. We only use
these three; pickup/drop/toggle/done are unused for pure navigation. They map to
global indices [2,3,4] (see imitation_data.GAME_ACTION_TO_GLOBAL["MiniGrid"]).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.env.atari_wrapper import TextState


# MiniGrid agent_dir convention: 0=east(right), 1=south(down), 2=west(left), 3=north(up)
_DIR_NAME = {0: "east", 1: "south", 2: "west", 3: "north"}
_DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _find_goal(grid, w, h):
    for i in range(w):
        for j in range(h):
            c = grid.get(i, j)
            if c is not None and c.type == "goal":
                return (i, j)
    return None


def _doorway_positions(grid, w, h):
    """Open passages through interior walls. Returns (x,y) of explicit door
    objects (MultiRoom) plus wall-gap cells (FourRooms): an open cell that sits
    on an interior wall line, i.e. a hole the agent can pass through."""

    def is_wall(x, y):
        if x < 0 or y < 0 or x >= w or y >= h:
            return True
        c = grid.get(x, y)
        return c is not None and c.type == "wall"

    def is_open(x, y):
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        c = grid.get(x, y)
        return c is None or c.type not in ("wall", "lava")

    doors = []
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            c = grid.get(i, j)
            if c is not None and c.type == "door":
                doors.append((i, j))
                continue
            if not is_open(i, j):
                continue
            # Horizontal corridor through a vertical wall: walls left+right, open up/down
            vgap = is_wall(i - 1, j) and is_wall(i + 1, j) and is_open(i, j - 1) and is_open(i, j + 1)
            # Vertical corridor through a horizontal wall: walls up+down, open left/right
            hgap = is_wall(i, j - 1) and is_wall(i, j + 1) and is_open(i - 1, j) and is_open(i + 1, j)
            if vgap or hgap:
                doors.append((i, j))
    return doors


def decode_minigrid_state(unwrapped, frame_idx: int) -> TextState:
    """Build a TextState from the live MiniGrid symbolic grid."""
    w, h = unwrapped.width, unwrapped.height
    grid = unwrapped.grid
    ax, ay = int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1])
    adir = int(unwrapped.agent_dir)
    goal = _find_goal(grid, w, h)
    gx, gy = (goal if goal is not None else (-1, -1))

    # Walls in the 4 cardinal directions adjacent to the agent
    def is_wall(x, y):
        if x < 0 or y < 0 or x >= w or y >= h:
            return True
        c = grid.get(x, y)
        return c is not None and c.type == "wall"

    walls = {
        "east": is_wall(ax + 1, ay),
        "south": is_wall(ax, ay + 1),
        "west": is_wall(ax - 1, ay),
        "north": is_wall(ax, ay - 1),
    }
    doors = _doorway_positions(grid, w, h)

    entities = {
        "agent_x": ax, "agent_y": ay,
        "agent_dir": _DIR_NAME.get(adir, str(adir)),
        "goal_x": int(gx), "goal_y": int(gy),
        "dx": int(gx - ax), "dy": int(gy - ay),
        "walls": walls,
        "doors": doors,
        "grid_w": w, "grid_h": h,
    }
    return TextState(frame_idx=frame_idx, score=0, lives=1, level=0, entities=entities)


class MiniGridEnv:
    """Thin wrapper over a MiniGrid gymnasium env emitting a TextState per slow tick."""

    def __init__(self, game_name: str = "MiniGrid-FourRooms-v0", seed: int = 0,
                 emit_every: int = 5):
        import gymnasium as gym
        import minigrid  # noqa: F401  (registers envs)
        # MiniGrid env id may be passed without the "MiniGrid-" prefix for brevity
        env_id = game_name if game_name.startswith("MiniGrid") else f"MiniGrid-{game_name}-v0"
        self.game_name = game_name
        self._gym = gym.make(env_id, render_mode="rgb_array")
        self._emit_every = emit_every
        self._tick = 0
        self._seed = seed

    # Number of native nav actions used (turn_left, turn_right, forward).
    action_space_size = 3

    @property
    def unwrapped(self):
        return self._gym.unwrapped

    def _render(self) -> np.ndarray:
        # Full top-down symbolic render (RGB HWC uint8) for the VL models.
        return self.unwrapped.get_frame(highlight=False)

    def reset(self, seed: Optional[int] = None):
        s = seed if seed is not None else self._seed
        obs, info = self._gym.reset(seed=s)
        self._tick = 0
        frame = self._render()
        # Emit an initial state so the slow model has context from tick 0
        text_state = decode_minigrid_state(self.unwrapped, self._tick)
        self._last_text_state = text_state
        return frame, text_state

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._gym.step(int(action))
        self._tick += 1
        frame = self._render()
        text_state = None
        if self._tick % self._emit_every == 0:
            text_state = decode_minigrid_state(self.unwrapped, self._tick)
        self._last_text_state = text_state
        return frame, float(reward), bool(terminated), bool(truncated), text_state

    def close(self):
        self._gym.close()
