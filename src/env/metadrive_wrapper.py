"""MetaDrive wrapper for the latent-bridge fast/slow setup.

Real-time premise (like Atari, unlike MiniGrid): MetaDrive steps at a fixed
control frequency (physics_dt=0.02 * decision_repeat=5 => 10 Hz control). A
~1.5s slow model cannot sit in the per-step loop => async fast/slow decoupling.
Dense reward built in (driving-progress + speed - crash), so F/T/L don't collapse
to all-zero.

Rendering: we use MetaDrive's TOP-DOWN (pygame/numpy) observation via
`TopDownMetaDrive`, NOT the Panda3D RGBCamera. Top-down rendering is pure CPU
surface blitting (no GL context, no simplepbr shader, no driver dependency) and
runs at ~300 steps/sec headless with `SDL_VIDEODRIVER=dummy`. This sidesteps the
NVIDIA GL/driver-skew wall that made the 3D RGBCamera path unusable on this box.
The fast (VL) model sees a 3-channel top-down image; the slow model still gets the
structured TextState decoded from env.agent / engine (unchanged).

Interface mirrors AtariEnv:
  reset(seed) -> (rgb HWC uint8, text_state)
  step(local_action_idx) -> (rgb, reward, terminated, truncated, text_state_or_None)
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np

# Top-down rendering uses pygame; force its offscreen ("dummy") video driver so the
# whole thing runs headless with no X / no GL. Set before MetaDrive/pygame import.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.env.atari_wrapper import TextState

# 9 discrete (steer, throttle) combos. MetaDrive action = [steer, throttle] in [-1,1];
# throttle<0 = brake/reverse.
STEERS = [-0.4, 0.0, 0.4]
THROTTLES = [-0.4, 0.1, 0.5]   # brake, light-coast, accelerate
DISCRETE_ACTIONS = [(s, t) for s in STEERS for t in THROTTLES]  # len 9

TEXT_EVERY = 10  # ~1 Hz slow emission at 10 Hz control

# Map layout. LB_MD_MAP env var overrides: e.g. "SXSXSX" forces a planning-heavy route
# (straights punctuated by X-intersections) where survival depends on taking the correct
# exit -- a SLOW sub-problem the fast reactive policy can't solve from the current frame
# alone. Default 3 = the original near-reactive lane-keeping layout.
import os as _os
def _map_spec():
    m = _os.environ.get("LB_MD_MAP", "3")
    try:
        return int(m)
    except ValueError:
        return m


def _md_config(seed: int):
    planning = isinstance(_map_spec(), str)  # explicit block string => planning route
    return dict(
        use_render=False,
        num_scenarios=200,
        start_seed=seed,
        horizon=600 if planning else 500,
        traffic_density=0.3,
        map=_map_spec(),
        out_of_route_done=planning,   # taking the wrong exit ENDS the episode (route matters)
        # reward shaping (defaults are already dense; keep explicit for the record)
        success_reward=20.0 if planning else 10.0,
        out_of_road_penalty=5.0,
        crash_vehicle_penalty=5.0,
        driving_reward=1.0,
        speed_reward=0.1,
    )


def decode_metadrive_state(env, frame_idx: int) -> TextState:
    try:
        return _decode_metadrive_state_inner(env, frame_idx)
    except Exception:
        # Any malformed lane/nav/traffic state -> minimal valid TextState rather
        # than crash Stage B. Speed is the one field that's always available.
        try:
            spd = round(float(env.agent.speed), 1)
        except Exception:
            spd = 0.0
        return TextState(frame_idx=frame_idx, score=0, lives=1, level=0,
                         entities={"speed": spd, "heading": 0.0, "lane_offset": 0.0,
                                   "lane_width": 3.5, "nav_left": 0.0, "nav_right": 0.0,
                                   "neighbors": []})


def _decode_metadrive_state_inner(env, frame_idx: int) -> TextState:
    v = env.agent
    speed = float(v.speed)
    heading = float(v.heading_theta)
    # lane lateral offset + width
    try:
        lane = v.lane
        long_, lat_ = lane.local_coordinates(v.position)
        lane_w = float(lane.width)
    except Exception:
        lat_, lane_w = 0.0, 3.5
    # upcoming route: MetaDrive's navi_arrow_dir gives the next maneuver direction.
    # It is [0,0] on a straight stretch and becomes non-zero (≈±pi/2) as the car nears
    # a junction where the route turns. This is the SLOW planning cue: it tells the slow
    # model "a turn is coming and which way" before the turn is reachable from the frame.
    nav = v.navigation
    nav_left = nav_right = 0.0
    turn_hint = "straight"
    try:
        arrow = getattr(nav, "navi_arrow_dir", None)
        if arrow is not None and len(arrow) >= 2:
            ax, ay = float(arrow[0]), float(arrow[1])
            # ay > 0 => upcoming right turn; ay < 0 => left; magnitude ~ pi/2 at a junction
            nav_right = round(max(0.0, ay), 2)
            nav_left = round(max(0.0, -ay), 2)
            if ay > 0.5:
                turn_hint = "turn_right_ahead"
            elif ay < -0.5:
                turn_hint = "turn_left_ahead"
    except Exception:
        pass
    try:
        route_completion = round(float(nav.route_completion), 2)
    except Exception:
        route_completion = 0.0
    # nearby vehicles (relative long/lat, speed)
    others = []
    try:
        px, py = float(v.position[0]), float(v.position[1])
        for ov in env.engine.traffic_manager.vehicles:
            if ov is v:
                continue
            dx = float(ov.position[0] - px); dy = float(ov.position[1] - py)
            d = (dx*dx + dy*dy) ** 0.5
            if d < 40:
                others.append((round(dx, 1), round(dy, 1), round(float(ov.speed), 1)))
        others = sorted(others, key=lambda o: o[0]**2 + o[1]**2)[:5]
    except Exception:
        pass
    entities = {
        "speed": round(speed, 1), "heading": round(heading, 2),
        "lane_offset": round(lat_, 2), "lane_width": round(lane_w, 1),
        "nav_left": round(nav_left, 2), "nav_right": round(nav_right, 2),
        "turn_hint": turn_hint, "route_completion": route_completion,
        "neighbors": others,
    }
    return TextState(frame_idx=frame_idx, score=0, lives=1, level=0, entities=entities)


class MetaDriveWrapper:
    action_space_size = len(DISCRETE_ACTIONS)  # 9

    def __init__(self, game_name: str = "MetaDrive", seed: int = 0, emit_every: int = TEXT_EVERY):
        from metadrive.envs.top_down_env import TopDownMetaDrive
        self.game_name = "MetaDrive"
        self._env = TopDownMetaDrive(_md_config(seed))
        self._emit_every = emit_every
        self._tick = 0
        self._seed = seed

    @property
    def agent(self):
        return self._env.agent

    @property
    def engine(self):
        return self._env.engine

    def _frame(self, obs) -> np.ndarray:
        """TopDownMultiChannel obs is (84,84,C) float in [0,1]. Compose a 3-channel
        uint8 image the VL model can read: B=road network, G=navigation route,
        R=other vehicles (max over the remaining/traffic channels). Consistent and
        informative regardless of the exact channel count."""
        a = np.asarray(obs, dtype=np.float32)
        if a.ndim == 2:
            a = a[..., None]
        c = a.shape[-1]
        road = a[..., 0]
        nav = a[..., 1] if c > 1 else road
        traffic = a[..., 2:].max(axis=-1) if c > 2 else np.zeros_like(road)
        rgb = np.stack([traffic, nav, road], axis=-1)  # R,G,B
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def reset(self, seed: Optional[int] = None):
        s = seed if seed is not None else self._seed
        obs, info = self._env.reset(seed=s)
        self._tick = 0
        ts = decode_metadrive_state(self._env, 0)
        return self._frame(obs), ts

    def step(self, action_idx: int):
        steer, thr = DISCRETE_ACTIONS[int(action_idx)]
        obs, reward, terminated, truncated, info = self._env.step(np.array([steer, thr], dtype=np.float32))
        self._tick += 1
        frame = self._frame(obs)
        text_state = None
        if self._tick % self._emit_every == 0:
            text_state = decode_metadrive_state(self._env, self._tick)
        return frame, float(reward), bool(terminated), bool(truncated), text_state

    def close(self):
        self._env.close()


def expert_action_idx(env_wrapper) -> int:
    """Map MetaDrive's pretrained PPO expert (continuous [steer,throttle]) to the
    nearest discrete action index. Used as the Stage A teacher."""
    from metadrive.examples import expert
    a = expert(env_wrapper.agent)            # np array [steer, throttle] in [-1,1]
    steer, thr = float(a[0]), float(a[1])
    s_i = int(np.argmin([abs(steer - s) for s in STEERS]))
    t_i = int(np.argmin([abs(thr - t) for t in THROTTLES]))
    return s_i * len(THROTTLES) + t_i
