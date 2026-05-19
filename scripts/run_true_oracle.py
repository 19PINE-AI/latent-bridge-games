"""True Oracle baseline: pre-compute slow analyses offline (no time budget),
then deploy F + injected texts. Bounds the maximum any text-channel coupling
can buy.

Procedure:
  Phase 1 (offline, no real-time constraint): run F to collect a trajectory
    of (frame, text_state) tuples, then for each text_state where the slow
    would emit at deployment, query the slow with --slow-max-tokens=1024.
    Save (tick → text) map.
  Phase 2 (deployment): replay the SAME F episode but inject the pre-computed
    slow text at each emission tick. Compare to F (no text) and T (in-loop
    short slow).

Usage:
    HF_HUB_OFFLINE=1 python scripts/run_true_oracle.py \
        --game MsPacman --seed 0 --max-ticks 500 \
        --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
        --out results/eval_true_oracle_mspacman.json
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
    ap.add_argument("--oracle-tokens", type=int, default=1024)
    ap.add_argument("--fast-ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # PHASE 1: offline trajectory + slow-text precomputation
    print("Loading FastModel for trajectory recording...")
    fast = FastModel(FastModelConfig()).load_pretrained()
    if os.path.exists(args.fast_ckpt):
        ck = torch.load(args.fast_ckpt, map_location="cuda", weights_only=False)
        if "action_head_state" in ck:
            fast.action_head.load_state_dict(ck["action_head_state"])

    print("Loading SlowModel for precomputation...")
    slow = SlowModel(SlowModelConfig()).load_pretrained()

    legal = legal_action_mask(args.game)
    results = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        env = AtariEnv(game_name=args.game, seed=seed)
        obs, _ = env.reset()

        # Record F-mode trajectory and pre-compute slow text for each emission
        trajectory_actions = []   # list of (global, local) per tick
        slow_texts_by_tick = {}   # tick_idx -> precomputed text
        latest_slow_text = None
        prior_thought = None

        print(f"\n[ep {ep}] Phase 1: F-mode trajectory + slow precomputation...", flush=True)
        for t in range(args.max_ticks):
            with torch.no_grad():
                logits = fast.predict_action(obs, thought_tokens=None,
                                             legal_action_mask=legal)
            global_action = int(logits.argmax(dim=-1).item())
            local_action = global_to_local_action(args.game, global_action)
            trajectory_actions.append((global_action, local_action))
            obs, reward, term, trunc, text_state = env.step(local_action)

            if text_state is not None:
                msgs = build_slow_model_messages(args.game, text_state,
                                                 prior_thought=prior_thought)
                t_slow = time.time()
                emit_text, _ = slow.emit(
                    msgs, frame=obs, max_new_tokens=args.oracle_tokens,
                    return_raw_residuals=False,
                )
                lat = time.time() - t_slow
                slow_texts_by_tick[t] = emit_text
                prior_thought = emit_text
                if len(slow_texts_by_tick) <= 2 or len(slow_texts_by_tick) % 10 == 0:
                    print(f"  tick {t:3d}  oracle emission ({lat:.1f}s, {len(emit_text)} chars)",
                          flush=True)
            if term or trunc:
                break

        env.close()
        print(f"  ep {ep}: F trajectory of {len(trajectory_actions)} ticks, "
              f"{len(slow_texts_by_tick)} oracle emissions")

        # PHASE 2: deploy F with pre-computed slow text injected
        print(f"[ep {ep}] Phase 2: F with injected oracle text...", flush=True)
        env = AtariEnv(game_name=args.game, seed=seed)
        obs, _ = env.reset()
        cumulative_reward = 0.0
        latest_oracle_text = None
        for t in range(args.max_ticks):
            # Inject precomputed text up to this tick
            if t in slow_texts_by_tick:
                latest_oracle_text = slow_texts_by_tick[t]
            with torch.no_grad():
                logits = fast.predict_action(
                    obs, thought_tokens=None,
                    legal_action_mask=legal,
                    slow_text_suffix=latest_oracle_text,
                )
            global_action = int(logits.argmax(dim=-1).item())
            local_action = global_to_local_action(args.game, global_action)
            obs, reward, term, trunc, _ = env.step(local_action)
            cumulative_reward += float(reward)
            if term or trunc:
                break
        env.close()
        print(f"  ep {ep} ORACLE score: {cumulative_reward:.0f}", flush=True)
        results.append({"ep": ep, "seed": seed, "score": cumulative_reward,
                        "n_oracle_emissions": len(slow_texts_by_tick)})

    scores = [r["score"] for r in results]
    summary = {
        "game": args.game, "n_episodes": len(results),
        "oracle_tokens": args.oracle_tokens,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "episodes": results,
    }
    print(f"\nSummary: O-true/{args.game}: mean={summary['mean_score']:.1f} "
          f"± {summary['std_score']:.1f}  median={summary['median_score']:.0f}")
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
