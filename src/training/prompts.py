"""Prompt templates that turn decoded TextState into slow-model-friendly prose.

Used by the T (text-bridge) baseline and by Stage C0 of the latent-bridge curriculum.
The slow model (Qwen3-VL-8B-Thinking) ingests:
  - System prompt with task framing
  - Per-game user prompt assembled from the most recent TextState(s)
  - (Optionally) an attached 1Hz frame snapshot

Design notes:
  - Keep the prose dense: every token costs a slow-model emission cycle.
  - Coordinates are reported in sprite-grid units (the AtariARI convention) rather than
    pixels — they're stable across ROM rendering quirks and easier for the slow model to
    reason over because grid steps are uniform.
  - We don't ask for a final action — the slow model emits strategic guidance, and the
    fast model selects the action. This matches the T baseline protocol.
"""
from __future__ import annotations

from typing import Sequence

from src.env.atari_wrapper import TextState


SYSTEM_PROMPT = """You are the strategic-reasoning module of a fast/slow agent playing an Atari game in real time.

You receive a compact text snapshot of the game state every ~1 second. The fast model handles per-frame reactions; your job is to plan over a 5-30 second horizon and emit *strategic guidance* — not individual actions.

Your output should be 1-3 sentences identifying:
  1. The current threat or opportunity (e.g., a ghost is closing in, a power-pellet is reachable, the temperature is dropping).
  2. The recommended directional/positional intent (e.g., "head to the bottom-left maze quadrant", "prioritize the closest ice floe row").
  3. Any contingency (e.g., "if Blinky reverses, switch to the upper corridor").

Be concise. The fast model has its own perception of pixels — you do not need to describe what it can see directly. Focus on what *changes* the reactive policy."""


def _mspacman_user_prompt(ts: TextState) -> str:
    e = ts.entities
    px, py = e["pacman_xy"]
    ghosts = [
        ("Sue (orange)", e["ghost_sue_xy"]),
        ("Inky (cyan)", e["ghost_inky_xy"]),
        ("Pinky (pink)", e["ghost_pinky_xy"]),
        ("Blinky (red)", e["ghost_blinky_xy"]),
    ]
    ghost_lines = "\n".join(
        f"  - {name} at ({gx}, {gy})  (Δ = ({gx - px:+d}, {gy - py:+d}))"
        for name, (gx, gy) in ghosts
    )
    fruit_x, fruit_y = e["fruit_xy"]
    fruit_line = (
        f"\n- Bonus fruit visible at ({fruit_x}, {fruit_y})." if (fruit_x or fruit_y) else ""
    )
    return f"""[Ms. Pac-Man — game state at frame {ts.frame_idx}]
- Score: {ts.score}    Lives: {ts.lives}    Dots eaten this board: {e['dots_eaten']}
- Pac-Man at ({px}, {py}), facing direction code {e['player_direction']}
- Ghosts on board: {e['ghosts_on_board']}/4
{ghost_lines}{fruit_line}

Provide strategic guidance for the next ~10 seconds."""


def _frostbite_user_prompt(ts: TextState) -> str:
    e = ts.entities
    px, py = e["player_xy"]
    floes = e["ice_floe_xs_top_to_bottom"]
    floe_line = ", ".join(
        f"row{i} (top→bottom) x={x}" for i, x in enumerate(floes)
    )
    hazards = e["hazards"]
    if hazards:
        hazard_lines = "\n".join(
            f"  - {h['kind']} at x={h['x']}, row_y={h['row_y']}"
            + (" (double instance)" if h["double"] else "")
            for h in hazards
        )
    else:
        hazard_lines = "  - none detected"
    bear = e["polar_bear_x"]
    bear_line = (
        f"\n- POLAR BEAR on top shore at x={bear} (chases Bailey on igloo platform)"
        if bear is not None else ""
    )
    sinking = "\n- WARNING: Bailey is sinking (water hit)." if e["sinking"] else ""
    return f"""[Frostbite — game state at frame {ts.frame_idx}]
- Score: {ts.score}    Lives: {ts.lives}    Temperature: {e['temperature']}    Igloo: {e['igloo_progress']}/15 blocks
- Bailey at ({px}, {py})
- Ice floe row x-positions: {floe_line}
- Hazards on floe rows:
{hazard_lines}{bear_line}{sinking}

Provide strategic guidance for the next ~10 seconds."""


def _seaquest_user_prompt(ts: TextState) -> str:
    e = ts.entities
    px, py = e["player_xy"]
    oxygen = e["oxygen"]
    divers = e["divers_collected"]
    facing = "right" if e["facing_east"] else "left"
    own_torpedo = (f"\n- Own torpedo in flight at x={e['own_torpedo_x']}"
                   if e["own_torpedo_x"] is not None else "")
    surface_enemy = (f"\n- SURFACE ENEMY at x={e['surface_enemy_x']} (must shoot before surfacing)"
                     if e["surface_enemy_x"] is not None else "")
    crit_oxy = "\n- ⚠ OXYGEN CRITICAL — must surface" if e["oxygen_critical"] else ""
    sub_full = "\n- ⚠ SUBMARINE FULL (6 divers) — surface to claim bonus" if e["submarine_full"] else ""
    lane_lines = "\n".join(
        f"  - lane{i} (bottom→top) base_x={x}" for i, x in enumerate(e["enemy_lane_xs"])
    )
    slot_lines = "\n".join(
        f"  - slot{i} at x={x}" for i, x in enumerate(e["slot_xs"]) if x != 0
    ) or "  - all slots empty"
    return f"""[Seaquest — game state at frame {ts.frame_idx}]
- Score: {ts.score}    Lives: {ts.lives}
- Oxygen: {oxygen}/64    Divers collected: {divers}/6
- Player at ({px}, {py}), facing {facing}{crit_oxy}{sub_full}{own_torpedo}{surface_enemy}
- Enemy lanes:
{lane_lines}
- Diver/missile slots:
{slot_lines}

Strategic considerations: oxygen depletion forces surfacing; surfacing with 6 divers
gives a big bonus; surfacing with 0 divers in later levels costs a life; shoot enemies
for points but the kill-economy is secondary to diver-collection. Provide guidance
for the next ~10 seconds."""


def _spaceinvaders_user_prompt(ts: TextState) -> str:
    e = ts.entities
    cannon_x = e["cannon_x"]
    form_x, form_y = e["formation_xy"]
    rows_alive = e["row_bitfields_top_to_bottom"]
    rows_lines = "\n".join(
        f"  - row{i} (top→bottom): alive_columns={bin(b)[2:].zfill(6)[::-1]} ({bin(b).count('1')}/6)"
        for i, b in enumerate(rows_alive)
    )
    bomb_lines = "\n".join(
        f"  - bomb at ({b['x']}, {b['y']})" for b in e["enemy_bombs"]
    ) or "  - none in flight"
    own_missile = (f"\n- Own missile in flight at {e['player_missile_xy']}"
                   if e["player_missile_xy"] is not None else "")
    saucer = (f"\n- Bonus saucer at x={e['saucer_x']} — high reward!"
              if e["saucer_x"] is not None else "")
    return f"""[Space Invaders — game state at frame {ts.frame_idx}]
- Score: {ts.score}    Lives: {ts.lives}
- Cannon at x={cannon_x}; invaders left: {e['invaders_left']}/36
- Invader formation centered at ({form_x}, {form_y}){own_missile}{saucer}
- Surviving invaders per row:
{rows_lines}
- Enemy bombs:
{bomb_lines}

CRITICAL: the only way to score is to FIRE at invaders. Every fired missile that
hits is +5 to +30 points; every tick spent not firing is a tick the enemy advances.
Bombs are RARE and a 1-tile dodge is enough — never abandon firing to dodge unless
a bomb is directly above the cannon. Saucers (when present) are worth up to 200
points — drop everything to shoot them.

Recommend: keep firing constantly (FIRE / RIGHTFIRE / LEFTFIRE actions). Choose
the directional component to align the cannon under the densest surviving column.
Provide guidance for the next ~10 seconds; lead with the column to target."""


def _pong_user_prompt(ts: TextState) -> str:
    e = ts.entities
    player_y = e["player_paddle_y"]
    enemy_y = e["enemy_paddle_y"]
    bx, by = e["ball_xy"]
    rel_to_player = e["ball_relative_to_player"]
    cue = ""
    if abs(rel_to_player) > 20:
        cue = "  (move toward ball)"
    elif abs(rel_to_player) < 6:
        cue = "  (paddle aligned with ball)"
    return f"""[Pong — game state at frame {ts.frame_idx}]
- Score: player {e['player_score']} - enemy {e['enemy_score']}
- Player paddle (right) y={player_y}; enemy paddle (left) y={enemy_y}
- Ball at ({bx}, {by}); ball is {rel_to_player:+d} relative to player paddle{cue}

Strategic considerations: Pong rewards consistent paddle alignment with the ball
(positive vertical-delta = ball below = move DOWN; negative = ball above = move UP).
Score-difference matters more than per-rally win rate. Provide guidance for the
next ~5 seconds."""


def _riverraid_user_prompt(ts: TextState) -> str:
    e = ts.entities
    fuel = e["fuel"]
    fuel_status = ("CRITICAL" if e["fuel_critical"] else
                   ("LOW" if e["fuel_low"] else "OK"))
    enemies_str = "\n".join(
        f"  - craft at ({x}, {y})" for x, y in zip(e["enemy_xs"], e["enemy_ys"])
        if (x, y) != (0, 0)
    ) or "  - no enemies visible"
    return f"""[River Raid — game state at frame {ts.frame_idx}]
- Score: {ts.score}    Lives: {ts.lives}
- Player x={e['player_x']} (jet position, scrolling river)
- Fuel: {fuel}/255 — status: {fuel_status}
- Enemy craft visible:
{enemies_str}

Strategic considerations: fuel depletion ends the run — refueling stations (fuel
gauge ramps) are scattered along the river; prioritize reaching one when fuel is
LOW or CRITICAL even at the cost of skipping enemies. Bridges block forward
progress until shot. Enemies score points (jets > helicopters > ships) but the
kill economy is secondary to fuel survival. Provide guidance for the next ~10
seconds; lead with fuel priority if relevant."""


def _berzerk_user_prompt(ts: TextState) -> str:
    e = ts.entities
    px, py = e["player_xy"]
    robots_str = "\n".join(
        f"  - robot {i} at ({r['x']}, {r['y']})  Δ=({r['x']-px:+d}, {r['y']-py:+d})"
        for i, r in enumerate(e["robots"])
    ) or "  - no robots in this room"
    otto = ""
    if e["evil_otto_active"]:
        ox, oy = e["evil_otto_xy"]
        otto = (f"\n- ⚠ EVIL OTTO active at ({ox}, {oy}); Δ=({ox-px:+d}, {oy-py:+d}) "
                f"— bouncing toward player, can pass through walls and enemies")
    return f"""[Berzerk — game state at frame {ts.frame_idx}]
- Score: {ts.score}    Lives: {ts.lives}    Room: {e['level']}
- Player at ({px}, {py})
- Robots in room ({e['n_robots']}):
{robots_str}{otto}

Strategic considerations: rooms are bounded mazes with exits on each side.
Killing all robots clears the room for bonus points but is risky; staying still
draws Evil Otto. Lead with: which exit to head for, which robots to shoot first
vs which to dodge, and whether the time budget allows full clearing before Otto
arrives. The map of the current room and the relative positions of multiple
robots is the kind of joint spatial state that demands rich strategic guidance.
Provide guidance for the next ~10 seconds."""


def _roadrunner_user_prompt(ts: TextState) -> str:
    e = ts.entities
    rx, ry = e["roadrunner_x"], e["roadrunner_y"]
    cx, cy = e["coyote_x"], e["coyote_y"]
    dx = e["coyote_distance_x"]
    coyote_warn = ("⚠ COYOTE close!" if abs(dx) < 30 else "")
    obs_lines = "\n".join(
        f"  - obstacle at ({x}, {y})" for x, y in zip(e["obstacle_xs"], e["obstacle_ys"])
        if (x, y) != (0, 0)
    ) or "  - none visible"
    return f"""[Road Runner — game state at frame {ts.frame_idx}]
- Score: {ts.score}    Lives: {ts.lives}
- Road Runner (player) at ({rx}, {ry}); Coyote at ({cx}, {cy}); Δ_x={dx:+d} {coyote_warn}
- Nearest birdseed pellet at x={e['pellet_x']}
- Obstacles (trucks/landmines) ahead:
{obs_lines}

Strategic considerations: collect birdseed pellets for points (~100 each); stay
ahead of Coyote; dodge trucks (above road) and landmines (on road). Speed and
position trade-off: pellets reward but slow you; obstacles require quick lateral
dodges. Provide guidance for the next ~10 seconds."""


def _enduro_user_prompt(ts: TextState) -> str:
    e = ts.entities
    enemy_str = "\n".join(
        f"  - car at ({x}, {y})" for x, y in zip(e["enemy_car_xs"], e["enemy_car_ys"])
        if (x, y) != (0, 0)
    ) or "  - clear road"
    quota_warn = "⚠ behind quota!" if e["cars_to_pass"] > 100 else ""
    return f"""[Enduro — game state at frame {ts.frame_idx}]
- Day: {e['day']}    Score: {ts.score}    Speed: {e['speed']}
- Player car x={e['player_x']} (lateral position)
- Cars left to pass today: {e['cars_to_pass']}  {quota_warn}
- Visible opposing cars:
{enemy_str}

Strategic considerations: pass the daily car quota before time runs out;
keep speed up (RIGHT to accelerate) but steer (LEFT/RIGHT) to dodge cars;
weather/visibility changes throughout the day. Lead with: steer direction
+ accelerate/brake decision for the next ~5 seconds."""


def _qbert_user_prompt(ts: TextState) -> str:
    e = ts.entities
    return f"""[Q*bert — game state at frame {ts.frame_idx}]
- Score: {ts.score}    Lives: {ts.lives}    Level: {e['level']}
- Q*bert on tile row={e['qbert_on_tile_row']}, col={e['qbert_on_tile_col']}
- Q*bert pixel position: ({e['qbert_x']}, {e['qbert_y']})
- Coily (snake) at ({e['coily_x']}, {e['coily_y']})
- Purple ball y={e['purple_ball_y']}, green enemy y={e['green_enemy_y']}

Strategic considerations: jump on all tiles to change their color (level
clears). Avoid Coily and falling balls. Coily can be tricked off the edge.
Plan a tile-jump sequence; lead with direction (UP/RIGHT/LEFT/DOWN) for
the next ~5 seconds."""


_USER_PROMPT_BUILDERS = {
    "MsPacman": _mspacman_user_prompt,
    "Frostbite": _frostbite_user_prompt,
    "Seaquest": _seaquest_user_prompt,
    "SpaceInvaders": _spaceinvaders_user_prompt,
    "Pong": _pong_user_prompt,
    "Riverraid": _riverraid_user_prompt,
    "RiverRaid": _riverraid_user_prompt,
    "Berzerk": _berzerk_user_prompt,
    "RoadRunner": _roadrunner_user_prompt,
    "Roadrunner": _roadrunner_user_prompt,
    "Enduro": _enduro_user_prompt,
    "Qbert": _qbert_user_prompt,
}


def build_slow_model_messages(game: str, text_state: TextState,
                              prior_thought: str | None = None) -> list[dict]:
    """Assemble a chat-format message list for the slow model's T-condition emission.

    Returns OpenAI-style messages: [system, user] (optionally with a prior-thought
    assistant turn for short-context continuity).

    `prior_thought` is the slow model's *previous* emission, threaded back in as a
    single turn so the slow model has a memory of its last decision. This is the
    cheapest form of continuity that costs no LoRA training; bridge-mediated continuity
    is what Stage C trains.
    """
    builder = _USER_PROMPT_BUILDERS.get(game)
    if builder is None:
        raise ValueError(f"no prompt template registered for game {game!r}")
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prior_thought:
        msgs.append({"role": "assistant", "content": prior_thought})
    msgs.append({"role": "user", "content": builder(text_state)})
    return msgs


def build_fast_model_context_suffix(latest_slow_emission: str) -> str:
    """Format the slow-model emission as a context suffix injected into the fast
    model's prompt under the T condition. Kept short and clearly delimited so the fast
    model can learn to attend to it without parsing ambiguity."""
    return f"\n[strategic-guidance]: {latest_slow_emission.strip()}"
