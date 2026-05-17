"""Oracle (O) baseline: T with the slow-model wall-clock budget removed.

Same fast model + T-style text-suffix injection, but the slow model gets up to
`--slow-max-tokens=512` (vs 64 for the T baseline) and is allowed to think as long
as needed per emission. Other text-bridge mechanics are unchanged.

Bounds the maximum any text-channel fast/slow coupling can buy. If L exceeds O,
the latent advantage is providing something text genuinely cannot capture.

Usage:
    HF_HUB_OFFLINE=1 python scripts/run_oracle_baseline.py \
        --game MsPacman --episodes 3 --max-ticks 500 \
        --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
        --out results/eval_oracle_mspacman.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.atari_wrapper import AtariEnv
from src.models.fast_model import FastModel, FastModelConfig
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.imitation_data import legal_action_mask, global_to_local_action
from src.training.prompts import build_slow_model_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-ticks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slow-max-tokens", type=int, default=512,
                    help="Oracle slow-token budget. Default 512 vs T's 64.")
    ap.add_argument("--fast-ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("Loading FastModel...")
    fast = FastModel(FastModelConfig()).load_pretrained()
    if args.fast_ckpt and os.path.exists(args.fast_ckpt):
        ck = torch.load(args.fast_ckpt, map_location="cuda", weights_only=False)
        if "action_head_state" in ck:
            fast.action_head.load_state_dict(ck["action_head_state"])
            print(f"  loaded action_head from {args.fast_ckpt}")
    print("Loading SlowModel...")
    slow = SlowModel(SlowModelConfig()).load_pretrained()

    legal = legal_action_mask(args.game)

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        env = AtariEnv(game_name=args.game, seed=seed)
        obs, _ = env.reset()

        cumulative_reward = 0.0
        n_ticks = 0
        n_slow_emissions = 0
        latest_slow_text = None
        slow_latencies_s = []
        t_ep_start = time.time()

        print(f"\n[ep {ep} seed {seed}] starting...", flush=True)
        for t in range(args.max_ticks):
            with torch.no_grad():
                logits = fast.predict_action(
                    obs, thought_tokens=None,
                    legal_action_mask=legal,
                    slow_text_suffix=latest_slow_text,
                )
            global_action = int(logits.argmax(dim=-1).item())
            local_action = global_to_local_action(args.game, global_action)
            obs, reward, term, trunc, text_state = env.step(local_action)
            cumulative_reward += float(reward)
            n_ticks += 1

            if text_state is not None:
                msgs = build_slow_model_messages(args.game, text_state,
                                                 prior_thought=latest_slow_text)
                t_slow = time.time()
                emit_text, _emit_bridge = slow.emit(
                    msgs, frame=obs, max_new_tokens=args.slow_max_tokens,
                    return_raw_residuals=False,
                )
                slow_latencies_s.append(time.time() - t_slow)
                latest_slow_text = emit_text
                n_slow_emissions += 1
                if n_slow_emissions <= 2 or n_slow_emissions % 10 == 0:
                    print(f"  tick {t:3d}  slow={slow_latencies_s[-1]:.1f}s "
                          f"text_len={len(emit_text)}", flush=True)

            if term or trunc:
                break

        env.close()
        ep_time = time.time() - t_ep_start
        ep_result = {
            "ep": ep, "seed": seed, "game": args.game,
            "score": cumulative_reward, "ticks": n_ticks,
            "n_slow_emissions": n_slow_emissions,
            "mean_slow_latency_s": float(np.mean(slow_latencies_s))
                                  if slow_latencies_s else 0.0,
            "wall_clock_sec": ep_time,
        }
        results.append(ep_result)
        print(f"  [ep {ep} done] score={cumulative_reward:.0f} ticks={n_ticks} "
              f"slow={n_slow_emissions} wall={ep_time:.0f}s", flush=True)

    scores = [r["score"] for r in results]
    summary = {
        "game": args.game, "n_episodes": len(results),
        "slow_max_tokens": args.slow_max_tokens,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "mean_slow_latency_s": float(np.mean([r["mean_slow_latency_s"] for r in results])),
        "episodes": results,
    }
    print(f"\nSummary: O/{args.game}: mean={summary['mean_score']:.1f}"
          f" ± {summary['std_score']:.1f}  median={summary['median_score']:.0f}"
          f"  slow_latency={summary['mean_slow_latency_s']:.1f}s")

    if args.out:
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
