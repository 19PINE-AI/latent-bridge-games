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


_USER_PROMPT_BUILDERS = {
    "MsPacman": _mspacman_user_prompt,
    "Frostbite": _frostbite_user_prompt,
    "Seaquest": _seaquest_user_prompt,
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
