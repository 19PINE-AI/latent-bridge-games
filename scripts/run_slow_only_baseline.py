"""S-condition: slow-only baseline.

The slow model directly chooses actions at its own emission rate (1-2 Hz). The fast
model is not used. Between slow decisions, the action repeats (this is the only
sensible thing to do — anything else would cheat by introducing a second policy).

This is the control showing "just use the big model in real time" is the wrong design.
Atari runs at 60 Hz (15 Hz for our fast-tick), so choosing an action only every ~15
ticks loses far too much information for reactive games.

Usage:
    HF_HUB_OFFLINE=1 python scripts/run_slow_only_baseline.py \
        --game MsPacman --episodes 3 --max-ticks 500 \
        --out results/eval_slow_only_mspacman.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.atari_wrapper import AtariEnv
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.imitation_data import (
    legal_action_mask, global_to_local_action, GAME_ACTION_TO_GLOBAL,
)
from src.training.prompts import build_slow_model_messages, SYSTEM_PROMPT


# Reuse the existing user-prompt builders but append an explicit action request.
_ACTION_REQUEST = """\

Respond with EXACTLY one of the legal action indices below, prefixed by 'ACTION: '.
Choose the action that best matches your strategic recommendation above.

Legal actions: {legal_str}

Example response format: 'ACTION: 3'"""


def _format_legal(game: str) -> str:
    """Human-readable list of legal action indices for the game."""
    legal = legal_action_mask(game)
    indices = legal.nonzero(as_tuple=True)[0].tolist()
    names = [
        "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
        "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
        "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
        "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
    ]
    return ", ".join(f"{i}={names[i]}" for i in indices)


def _parse_action(text: str, legal_indices: list[int]) -> int:
    """Extract action index from slow's response. Falls back to NOOP if parse fails."""
    if not text:
        return 0 if 0 in legal_indices else legal_indices[0]
    # Match "ACTION: N" anywhere in the response
    m = re.search(r"ACTION[:\s]+(\d+)", text, re.IGNORECASE)
    if m:
        try:
            a = int(m.group(1))
            if a in legal_indices:
                return a
        except ValueError:
            pass
    # Fallback: last bare integer in the text
    nums = re.findall(r"\b(\d+)\b", text)
    for n in reversed(nums):
        try:
            a = int(n)
            if a in legal_indices:
                return a
        except ValueError:
            pass
    return 0 if 0 in legal_indices else legal_indices[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-ticks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slow-max-tokens", type=int, default=128,
                    help="max tokens for the slow's action emission. Higher = more "
                         "thinking budget but slower. 128 is enough for short CoT.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Loading SlowModel...")
    slow = SlowModel(SlowModelConfig()).load_pretrained()
    print(f"  done. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    legal = legal_action_mask(args.game)
    legal_indices = legal.nonzero(as_tuple=True)[0].tolist()
    legal_str = _format_legal(args.game)

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        env = AtariEnv(game_name=args.game, seed=seed)
        obs, text_state0 = env.reset()
        text_state = text_state0  # the env returns initial text state at reset

        cumulative_reward = 0.0
        n_ticks = 0
        n_slow_calls = 0
        slow_latencies_s = []
        latest_action = 0 if 0 in legal_indices else legal_indices[0]
        prior_thought = None
        t_ep_start = time.time()

        print(f"\n[ep {ep} seed {seed}] starting...", flush=True)
        for t in range(args.max_ticks):
            # Slow choice every ~15 fast ticks (matches the 1Hz emission cadence).
            # The slow.emit() runs at its own wall-clock speed; we record that.
            if text_state is not None:
                msgs = build_slow_model_messages(args.game, text_state,
                                                 prior_thought=prior_thought)
                # Append the action-request to the user turn
                msgs[-1] = dict(msgs[-1])
                msgs[-1]["content"] = msgs[-1]["content"] + _ACTION_REQUEST.format(
                    legal_str=legal_str)
                t_slow = time.time()
                emit_text, _emit_bridge = slow.emit(
                    msgs, frame=obs, max_new_tokens=args.slow_max_tokens,
                    return_raw_residuals=False,
                )
                slow_latencies_s.append(time.time() - t_slow)
                latest_action = _parse_action(emit_text, legal_indices)
                prior_thought = emit_text
                n_slow_calls += 1
                if n_slow_calls <= 3 or n_slow_calls % 10 == 0:
                    print(f"  tick {t:3d}  slow={slow_latencies_s[-1]:.1f}s  "
                          f"chose action={latest_action}  text={emit_text[-80:]!r}",
                          flush=True)

            # Step env with the latest_action; same action repeats between slow calls
            local_action = global_to_local_action(args.game, latest_action)
            obs, reward, terminated, truncated, text_state = env.step(local_action)
            cumulative_reward += float(reward)
            n_ticks += 1

            if terminated or truncated:
                break

        env.close()
        ep_time = time.time() - t_ep_start
        ep_result = {
            "ep": ep, "seed": seed, "game": args.game,
            "score": cumulative_reward, "ticks": n_ticks,
            "n_slow_calls": n_slow_calls,
            "mean_slow_latency_s": float(np.mean(slow_latencies_s))
                                  if slow_latencies_s else 0.0,
            "wall_clock_sec": ep_time,
        }
        results.append(ep_result)
        print(f"  [ep {ep} done] score={cumulative_reward:.0f} ticks={n_ticks} "
              f"slow_calls={n_slow_calls} wall={ep_time:.0f}s "
              f"slow_lat={ep_result['mean_slow_latency_s']:.1f}s", flush=True)

    scores = [r["score"] for r in results]
    summary = {
        "game": args.game, "n_episodes": len(results),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "mean_slow_latency_s": float(np.mean([r["mean_slow_latency_s"] for r in results])),
        "episodes": results,
    }
    print(f"\nSummary: S/{args.game}: mean={summary['mean_score']:.1f}"
          f" ± {summary['std_score']:.1f}  median={summary['median_score']:.0f}"
          f"  slow_latency={summary['mean_slow_latency_s']:.1f}s")

    if args.out:
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
