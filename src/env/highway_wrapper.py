"""highway-env wrapper for the latent-bridge fast/slow setup.

Design that makes the latency premise REAL (unlike MiniGrid):
  - policy_frequency = 15 Hz, ContinuousAction [throttle, steering] -> the fast
    model does genuine ~67 ms reactive control; a ~1.5 s slow model CANNOT sit in
    this loop, forcing async fast/slow decoupling exactly like Atari.
  - slow model emits strategic guidance at ~1 Hz (every 15 fast ticks).
  - dense reward built in (speed-maintenance + right-lane + collision penalty),
    so F/T/L do not collapse to all-zero.

Action discretization for the fast model's discrete head:
  9 actions = {throttle: brake -0.5 / coast 0 / accel +0.5}
            x {steer:    left -0.25 / straight 0 / right +0.25}
  mapped to global slots GAME_ACTION_TO_GLOBAL["Highway"].

Interface mirrors AtariEnv:
  reset(seed) -> (rgb_obs HWC uint8, text_state)
  step(local_action_idx) -> (rgb, reward, terminated, truncated, text_state_or_None)
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from src.env.atari_wrapper import TextState

# 9 discrete (throttle, steering) combos
THROTTLES = [-0.5, 0.0, 0.5]
STEERS = [-0.25, 0.0, 0.25]
DISCRETE_ACTIONS = [(t, s) for t in THROTTLES for s in STEERS]  # len 9

HIGHWAY_CONFIG = {
    "policy_frequency": 15,
    "simulation_frequency": 15,
    "duration": 40,
    "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
    "observation": {"type": "Kinematics", "vehicles_count": 8,
                    "features": ["presence", "x", "y", "vx", "vy"], "absolute": False},
    # HARDENED for a real strategic gap (remove ceiling): denser traffic + higher
    # target speed forces frequent overtaking; a speed-only/no-strategy policy now
    # gets stuck behind slow cars and scores far below an overtaking expert.
    "lanes_count": 4,
    "vehicles_count": 50,
    "vehicles_density": 2.0,
    "collision_reward": -1.0,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.05,
    "reward_speed_range": [25, 35],
    "normalize_reward": True,
}

TEXT_EVERY = 15  # 1 Hz slow emission at 15 Hz control


def decode_highway_state(u, frame_idx: int) -> TextState:
    """Structured state for the slow model's prompt, from ego + neighbours."""
    v = u.vehicle
    px, py = float(v.position[0]), float(v.position[1])
    speed = float(v.speed)
    lane = v.lane_index[2] if v.lane_index is not None else -1
    n_lanes = u.config.get("lanes_count", 4)
    # lead + neighbour gaps
    nbrs = u.road.neighbour_vehicles(v)
    lead = nbrs[0] if nbrs and nbrs[0] is not None else None
    lead_gap = float(lead.position[0] - px) if lead is not None else 999.0
    lead_speed = float(lead.speed) if lead is not None else 0.0
    others = []
    for ov in u.road.vehicles:
        if ov is v:
            continue
        dx = float(ov.position[0] - px); dy = float(ov.position[1] - py)
        if abs(dx) < 60:
            others.append((round(dx, 1), round(dy, 1), round(float(ov.speed), 1)))
    others = sorted(others, key=lambda o: abs(o[0]))[:5]
    entities = {
        "x": round(px, 1), "y": round(py, 1), "speed": round(speed, 1),
        "lane": int(lane), "n_lanes": int(n_lanes),
        "lead_gap": round(lead_gap, 1), "lead_speed": round(lead_speed, 1),
        "neighbors": others,
    }
    return TextState(frame_idx=frame_idx, score=0, lives=1, level=0, entities=entities)


class HighwayEnv:
    action_space_size = len(DISCRETE_ACTIONS)  # 9

    def __init__(self, game_name: str = "highway-v0", seed: int = 0, emit_every: int = TEXT_EVERY):
        import gymnasium as gym
        import highway_env  # noqa: F401
        self.game_name = "Highway"
        self._gym = gym.make("highway-v0", render_mode="rgb_array", config=HIGHWAY_CONFIG)
        self._emit_every = emit_every
        self._tick = 0
        self._seed = seed

    @property
    def unwrapped(self):
        return self._gym.unwrapped

    def _render(self) -> np.ndarray:
        return np.asarray(self._gym.render(), dtype=np.uint8)

    def reset(self, seed: Optional[int] = None):
        s = seed if seed is not None else self._seed
        obs, info = self._gym.reset(seed=s)
        self._tick = 0
        frame = self._render()
        ts = decode_highway_state(self.unwrapped, self._tick)
        return frame, ts

    def step(self, action_idx: int):
        thr, steer = DISCRETE_ACTIONS[int(action_idx)]
        obs, reward, terminated, truncated, info = self._gym.step(np.array([thr, steer], dtype=np.float32))
        self._tick += 1
        frame = self._render()
        text_state = None
        if self._tick % self._emit_every == 0:
            text_state = decode_highway_state(self.unwrapped, self._tick)
        return frame, float(reward), bool(terminated), bool(truncated), text_state

    def close(self):
        self._gym.close()


def _gap_in_lane(u, v, lane_idx, px):
    """Return (front_gap, back_gap) to nearest cars in lane `lane_idx` (a full
    lane_index tuple). Large sentinel if none. Used for overtaking decisions."""
    front, back = 999.0, 999.0
    for ov in u.road.vehicles:
        if ov is v:
            continue
        if ov.lane_index != lane_idx:
            continue
        dx = float(ov.position[0] - px)
        if dx >= 0:
            front = min(front, dx)
        else:
            back = min(back, -dx)
    return front, back


def scripted_expert_action(u) -> int:
    """Rule-based OVERTAKING driving expert -> discrete action index.

    Strategy (this is the slow/strategic layer that makes the domain non-trivial):
      - If blocked by a slower lead within a time-gap, look for an adjacent lane
        whose front+back gaps are safe and CHANGE LANES to overtake (prefer the
        faster/left lane, then right). This is a genuine multi-second decision.
      - Otherwise keep lane and regulate speed toward target.
    Longitudinal: target ~28 m/s; brake if lead too close and no lane to move to.
    Returns index into DISCRETE_ACTIONS (t_i*3 + s_i)."""
    v = u.vehicle
    speed = float(v.speed)
    px = float(v.position[0])
    nbrs = u.road.neighbour_vehicles(v)
    lead = nbrs[0] if nbrs and nbrs[0] is not None else None
    lead_gap = float(lead.position[0] - px) if lead is not None else 999.0
    lead_speed = float(lead.speed) if lead is not None else 99.0

    target = 28.0
    follow_gap = max(15.0, speed * 1.4)        # want to overtake within this gap
    blocked = lead_gap < follow_gap and lead_speed < speed - 1.0

    # current lane index tuple + side lanes
    li = v.lane_index
    side_indices = u.road.network.side_lanes(li) if hasattr(u.road.network, "side_lanes") else []

    steer_choice = 0.0
    thr_choice = 0.0
    did_lane_change = False

    if blocked and side_indices:
        # rank candidate lanes by free front gap; require safe back gap
        best = None
        for cand in side_indices:
            fg, bg = _gap_in_lane(u, v, cand, px)
            if fg > follow_gap + 5.0 and bg > 12.0:
                # prefer the lane with the larger front gap
                if best is None or fg > best[1]:
                    best = (cand, fg)
        if best is not None:
            cand = best[0]
            # cand lane number is cand[2]; steer toward it (lower index = left = -y)
            steer_choice = -0.25 if cand[2] < li[2] else 0.25
            thr_choice = 0.0   # maintain speed through the change
            did_lane_change = True

    if not did_lane_change:
        # longitudinal regulation + stay centered in lane
        if blocked or lead_gap < max(12.0, speed * 1.0):
            thr_choice = -0.5
        elif speed < target - 1:
            thr_choice = 0.5
        else:
            thr_choice = 0.0
        # keep lane center
        try:
            lane_obj = u.road.network.get_lane(li)
            _, lat_ = lane_obj.local_coordinates(v.position)
            if lat_ > 0.5:
                steer_choice = -0.25
            elif lat_ < -0.5:
                steer_choice = 0.25
        except Exception:
            steer_choice = 0.0

    t_i = THROTTLES.index(thr_choice)
    s_i = STEERS.index(steer_choice)
    return t_i * len(STEERS) + s_i
