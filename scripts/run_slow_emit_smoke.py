"""End-to-end smoke test of SlowModel.emit() on a real RAM-decoded game state.

Plays a few seconds of MsPacman, grabs a decoded TextState + the latest frame, builds
the prompt via the Stage-B template, and runs one slow-model emission. Prints:
  - The decoded TextState
  - The slow model's text emission (strategic guidance)
  - Shape + statistics of the projected thought vectors

This validates the slow-side data path end-to-end: env -> RAM decode -> prompt template
-> slow model emit -> thought-vector projection. The fast-side cross-attention consumer
is the next milestone.

Usage:
    python scripts/run_slow_emit_smoke.py --game MsPacman --warmup-ticks 90
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pprint import pformat

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.atari_wrapper import AtariEnv
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.prompts import build_slow_model_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--warmup-ticks", type=int, default=90,
                    help="random-action ticks before the snapshot (15Hz, so 90 = 6s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--no-frame", action="store_true",
                    help="skip attaching the frame snapshot (text-only emission)")
    args = ap.parse_args()

    # ---- Step 1: get a decoded text-state from real gameplay ----
    rng = np.random.default_rng(args.seed)
    env = AtariEnv(game_name=args.game, seed=args.seed)
    obs, _ = env.reset()
    text_state = None
    n_actions = env.action_space_size
    for _ in range(args.warmup_ticks):
        a = int(rng.integers(0, n_actions))
        obs, _, term, trunc, t = env.step(a)
        if t is not None:
            text_state = t
        if term or trunc:
            break
    env.close()
    if text_state is None:
        raise RuntimeError("no text-state emitted; increase --warmup-ticks")

    print("=" * 60)
    print(f"Decoded TextState for {args.game} (after {args.warmup_ticks} ticks):")
    print(pformat(text_state.__dict__, width=100, sort_dicts=False))
    print()

    # ---- Step 2: build the slow-model prompt ----
    messages = build_slow_model_messages(args.game, text_state)
    print("Prompt (last user turn):")
    print(messages[-1]["content"])
    print()

    # ---- Step 3: load slow model and run one emission ----
    print("Loading Qwen3-VL-8B-Thinking...")
    t0 = time.time()
    slow = SlowModel(SlowModelConfig()).load_pretrained()
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"Emitting (max_new_tokens={args.max_new_tokens})...")
    t0 = time.time()
    text, thought_vecs = slow.emit(
        messages,
        frame=None if args.no_frame else obs,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t0
    print(f"  emit completed in {elapsed:.2f}s")
    print()

    print("=" * 60)
    print("Slow-model text emission:")
    print(text)
    print("=" * 60)
    print(f"Thought vectors: shape={tuple(thought_vecs.shape)}  "
          f"dtype={thought_vecs.dtype}  device={thought_vecs.device}")
    print(f"  per-vector L2 norm range: "
          f"[{thought_vecs.float().norm(dim=-1).min().item():.3f}, "
          f"{thought_vecs.float().norm(dim=-1).max().item():.3f}]")
    print(f"  per-dim std across tokens: "
          f"{thought_vecs.float().std(dim=0).mean().item():.4f} "
          f"(near 0 would indicate a collapsed bridge)")


if __name__ == "__main__":
    main()
