"""T-condition runtime: slow + fast models running together via the text bridge.

End-to-end Stage B baseline:
  - Env loop at 15Hz on a single Atari game
  - Every 15 fast ticks (~1Hz), feed the decoded text-state (and current frame) to
    Qwen3-VL-8B-Thinking. The slow model emits 1-3 sentences of strategic guidance
    plus a stream of thought vectors.
  - The text emission is threaded into the fast model's user prompt as
    "[strategic-guidance]: ..." for the subsequent ticks.
  - The thought vectors are appended to a `ThoughtBuffer` for the L-condition that
    would consume them. In T mode, the buffer is recorded but ignored by the fast
    model's bridge (we pass `thought_buffer=None`).

Outputs per episode:
  - Final score / reward
  - List of (tick, action, slow_text_emission_or_None) for debugging
  - Trajectory file with all the data needed for Stage C bridge supervision:
      (obs, action, text_state, slow_text, thought_vectors, reward)

Usage:
    HF_HUB_OFFLINE=1 python scripts/run_text_bridge_baseline.py --game MsPacman \\
        --ticks 75 --seed 0 --out /tmp/t_episode.pt

Caveats:
  - Slow.emit blocks the env loop (~3-8s) every 15 ticks. For real-time inference this
    would need to be async on a separate stream/thread; for offline Stage-B data
    generation it's fine. We bake the staleness assumption into the protocol: the
    *next-tick* prompt uses the *previous* slow emission, so the fast model always
    sees stale-but-coherent guidance.
  - At init, the FastModel's action head is zero. For the demo we sample randomly
    from the legal action mask so the episode shows action variation; trained-head
    Stage A is needed before this is a real T-condition policy.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.atari_wrapper import AtariEnv
from src.bridge.ring_buffer import ThoughtBuffer
from src.models.fast_model import FastModel, FastModelConfig
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.imitation_data import legal_action_mask, global_to_local_action
from src.training.prompts import build_slow_model_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--ticks", type=int, default=75,
                    help="number of 15Hz fast ticks to run (default 75 = 5s game time)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slow-max-tokens", type=int, default=128,
                    help="max tokens per slow emission")
    ap.add_argument("--out", default=None, help="trajectory .pt output path")
    ap.add_argument("--action-policy", choices=("zero-head", "random-legal"),
                    default="random-legal",
                    help="At init the fast action_head outputs zeros; 'random-legal' "
                         "samples from legal actions for varied demo data.")
    ap.add_argument("--load-slow", action="store_true", default=True,
                    help="Load Qwen3-VL-8B-Thinking and emit real strategic guidance.")
    ap.add_argument("--no-slow", dest="load_slow", action="store_false",
                    help="Skip slow model (fast-only baseline; F condition).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    legal = legal_action_mask(args.game)
    legal_indices = legal.nonzero(as_tuple=True)[0].tolist()

    # ---- Load models ----
    print("Loading FastModel (MiniCPM-o)...", flush=True)
    t0 = time.time()
    fast = FastModel(FastModelConfig()).load_pretrained()
    print(f"  done in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    slow = None
    if args.load_slow:
        print("Loading SlowModel (Qwen3-VL-8B-Thinking)...", flush=True)
        t0 = time.time()
        slow = SlowModel(SlowModelConfig()).load_pretrained()
        print(f"  done in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ---- Bridge state ----
    bridge_dim = 256
    thought_buffer = ThoughtBuffer(
        capacity=16, dim=bridge_dim, device="cuda", dtype=torch.bfloat16,
    )

    # ---- Env loop ----
    env = AtariEnv(game_name=args.game, seed=args.seed)
    obs, _ = env.reset()

    trace = []  # per-tick: dict(tick, action, reward, slow_emit?, text_state?)
    latest_slow_text: str | None = None
    cumulative_reward = 0.0
    t_loop_start = time.time()

    for t in range(args.ticks):
        # 1) Predict fast-model action logits (frame-conditioned)
        with torch.no_grad():
            logits = fast.predict_action(
                obs,
                thought_buffer=None,  # T-mode: bridge present-but-silent (zero-init no-op anyway)
                legal_action_mask=legal,
                slow_text_suffix=latest_slow_text,
            )
        # 2) Choose action (in the global 18-way space)
        if args.action_policy == "zero-head":
            global_action = int(logits.argmax(dim=-1).item())
        else:
            global_action = int(rng.choice(legal_indices))

        # 3) Map global -> local for ALE, then step
        local_action = global_to_local_action(args.game, global_action)
        obs, reward, terminated, truncated, text_state = env.step(local_action)
        action = global_action  # what we record in trace
        cumulative_reward += float(reward)
        slow_emit_text = None
        slow_emit_vecs = None

        # 4) Slow emission on text-state ticks
        if text_state is not None and slow is not None:
            messages = build_slow_model_messages(args.game, text_state,
                                                 prior_thought=latest_slow_text)
            t_slow = time.time()
            slow_emit_text, slow_emit_vecs = slow.emit(
                messages, frame=obs, max_new_tokens=args.slow_max_tokens,
            )
            slow_elapsed = time.time() - t_slow
            # Write a single summary vector (last token's projection) to the thought buffer
            if slow_emit_vecs.numel() > 0:
                last_vec = slow_emit_vecs[-1].to(thought_buffer.device, dtype=thought_buffer.dtype)
                thought_buffer.append(last_vec, timestamp=(t + 1) / 15.0)
            latest_slow_text = slow_emit_text
            print(f"  [tick {t+1:3d}] slow emission ({slow_elapsed:.1f}s, "
                  f"{slow_emit_vecs.shape[0]} tokens): "
                  f"{slow_emit_text[:120]!r}", flush=True)

        trace.append({
            "tick": t,
            "action": action,
            "reward": float(reward),
            "text_state": vars(text_state) if text_state is not None else None,
            "slow_text": slow_emit_text,
            "slow_vecs_shape": list(slow_emit_vecs.shape) if slow_emit_vecs is not None else None,
        })

        if terminated or truncated:
            print(f"  [tick {t+1}] episode ended", flush=True)
            break

    env.close()
    elapsed = time.time() - t_loop_start
    n_ticks = len(trace)
    n_slow = sum(1 for x in trace if x["slow_text"])

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"T-condition episode summary ({args.game}):")
    print(f"  total ticks: {n_ticks}")
    print(f"  cumulative reward (score): {cumulative_reward:.0f}")
    print(f"  slow emissions: {n_slow}")
    print(f"  wall-clock: {elapsed:.1f}s  (game time: {n_ticks/15:.1f}s)")
    print(f"  slow-time fraction: "
          f"{(elapsed - n_ticks * (elapsed-n_slow*5)/max(1,n_ticks))/elapsed*100 if n_slow else 0:.0f}% est.")
    if n_slow > 0:
        print(f"\n  Sample slow emission:")
        first_slow = next(x for x in trace if x["slow_text"])
        print(f"    {first_slow['slow_text'][:300]!r}")

    if args.out:
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        torch.save({
            "game": args.game,
            "seed": args.seed,
            "trace": trace,
            "cumulative_reward": cumulative_reward,
        }, args.out)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
