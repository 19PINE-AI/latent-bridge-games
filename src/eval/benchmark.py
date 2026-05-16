"""Four-strategy benchmark harness.

Runs F (fast-only), T (text-bridge), L (latent-bridge), and optionally S (slow-only)
and O (oracle) on a configured game set with multiple seeds and episodes per cell.
Records per-tick latency, on-clock fraction, and score. Outputs per-cell results to
results/ as JSON for downstream statistical analysis.

Usage:
    HF_HUB_OFFLINE=1 python -m src.eval.benchmark \\
        --config configs/eval.yaml --strategies F T \\
        --games MsPacman --seeds 0 1 2 --episodes 5 \\
        --out results/eval_smoke.json

The harness is designed to amortize model loading across all (game × seed × episode)
cells for a given strategy: load models once, then iterate. Only the env is reset
per episode. Different strategies need different model combinations (F: fast only;
T/L: fast + slow), so strategies are batched at the outer loop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.env.atari_wrapper import AtariEnv
from src.bridge.ring_buffer import ThoughtBuffer
from src.models.fast_model import FastModel, FastModelConfig
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.imitation_data import legal_action_mask, global_to_local_action
from src.training.prompts import build_slow_model_messages


TICK_BUDGET_MS = 67.0  # 15Hz target


def _percentile(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


def _run_episode(strategy: str,
                 game: str,
                 seed: int,
                 max_ticks: int,
                 fast: FastModel,
                 slow: Optional[SlowModel],
                 fast_checkpoint: Optional[str],
                 stop_action_policy: str = "argmax",
                 max_slow_tokens: int = 64,
                 ) -> dict:
    """Run one episode under one strategy on one (game, seed) cell. Returns metrics."""
    legal = legal_action_mask(game)
    legal_indices = legal.nonzero(as_tuple=True)[0].tolist()
    bridge_dim = fast.cfg.bridge_dim
    rng = np.random.default_rng(seed)

    thought_buffer = ThoughtBuffer(
        capacity=16, dim=bridge_dim, device="cuda", dtype=torch.bfloat16,
    )
    env = AtariEnv(game_name=game, seed=seed)
    obs, _ = env.reset()
    latest_slow_text: Optional[str] = None
    tick_latencies_ms = []
    cumulative_reward = 0.0
    n_slow_emissions = 0
    n_ticks = 0
    t_episode_start = time.time()

    for t in range(max_ticks):
        # --- Fast action prediction (timed) ---
        t_tick = time.perf_counter()
        if strategy == "F":
            with torch.no_grad():
                logits = fast.predict_action(
                    obs, thought_buffer=None,
                    legal_action_mask=legal,
                )
        elif strategy in ("T", "L"):
            with torch.no_grad():
                logits = fast.predict_action(
                    obs,
                    thought_buffer=(thought_buffer.read(current_time=(t+1)/15.0)[0].unsqueeze(0)
                                    if strategy == "L" else None),
                    legal_action_mask=legal,
                    slow_text_suffix=(latest_slow_text if strategy == "T" else None),
                )
        else:
            raise ValueError(f"unsupported strategy {strategy!r}")

        if stop_action_policy == "argmax":
            global_action = int(logits.argmax(dim=-1).item())
        elif stop_action_policy == "sample":
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            global_action = int(torch.multinomial(probs, 1).item())
        else:
            global_action = int(rng.choice(legal_indices))
        torch.cuda.synchronize()
        tick_latencies_ms.append((time.perf_counter() - t_tick) * 1000.0)

        local_action = global_to_local_action(game, global_action)
        obs, reward, terminated, truncated, text_state = env.step(local_action)
        cumulative_reward += float(reward)
        n_ticks += 1

        # --- Slow emission (only T/L) ---
        if text_state is not None and slow is not None and strategy in ("T", "L"):
            messages = build_slow_model_messages(game, text_state,
                                                 prior_thought=latest_slow_text)
            emit_text, emit_vecs = slow.emit(messages, frame=obs,
                                             max_new_tokens=max_slow_tokens)
            latest_slow_text = emit_text
            if emit_vecs.numel() > 0:
                last_vec = emit_vecs[-1].to(thought_buffer.device, dtype=thought_buffer.dtype)
                thought_buffer.append(last_vec, timestamp=(t+1)/15.0)
            n_slow_emissions += 1

        if terminated or truncated:
            break

    env.close()
    on_clock = sum(1 for x in tick_latencies_ms if x <= TICK_BUDGET_MS) / max(1, len(tick_latencies_ms))
    return {
        "game": game,
        "strategy": strategy,
        "seed": seed,
        "score": cumulative_reward,
        "ticks": n_ticks,
        "wall_clock_sec": time.time() - t_episode_start,
        "slow_emissions": n_slow_emissions,
        "mean_action_latency_ms": float(np.mean(tick_latencies_ms)) if tick_latencies_ms else 0.0,
        "median_action_latency_ms": _percentile(tick_latencies_ms, 50),
        "p95_action_latency_ms": _percentile(tick_latencies_ms, 95),
        "frac_on_clock": on_clock,
    }


def _aggregate(cells: list[dict]) -> dict:
    """Group by (strategy, game), compute mean ± std + medians."""
    by = {}
    for c in cells:
        k = (c["strategy"], c["game"])
        by.setdefault(k, []).append(c)
    summary = {}
    for (strat, game), runs in by.items():
        scores = [r["score"] for r in runs]
        ticks = [r["ticks"] for r in runs]
        lats = [r["mean_action_latency_ms"] for r in runs]
        on_clock = [r["frac_on_clock"] for r in runs]
        summary[f"{strat}/{game}"] = {
            "n_episodes": len(runs),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "median_score": float(np.median(scores)),
            "mean_ticks": float(np.mean(ticks)),
            "mean_action_latency_ms": float(np.mean(lats)),
            "frac_on_clock": float(np.mean(on_clock)),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--strategies", nargs="+", choices=("F", "T", "L"), default=None,
                    help="override config")
    ap.add_argument("--games", nargs="+", default=None, help="override config")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override config")
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes per (strategy, game, seed) cell")
    ap.add_argument("--max-ticks", type=int, default=750,
                    help="max ticks per episode (default 750 = 50s game time)")
    ap.add_argument("--fast-ckpt", default=None,
                    help="Stage A action_head checkpoint (.pt) to load before eval")
    ap.add_argument("--bridge-ckpt", default=None,
                    help="Stage C bridge checkpoint (.pt) to load before L eval")
    ap.add_argument("--max-slow-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/eval_main.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strategies = args.strategies or cfg["strategies"]
    games = args.games or [g for tier in cfg["games"].values() for g in tier]
    seeds = args.seeds or cfg["seeds"]
    episodes = args.episodes or cfg["episodes_per_cell"]

    print(f"Eval plan:")
    print(f"  strategies: {strategies}")
    print(f"  games:      {games}")
    print(f"  seeds:      {seeds}")
    print(f"  episodes:   {episodes} per (strategy, game, seed)")
    print(f"  total cells: {len(strategies)} × {len(games)} × {len(seeds)} = "
          f"{len(strategies) * len(games) * len(seeds)}")
    print(f"  total episodes: {len(strategies) * len(games) * len(seeds) * episodes}")

    # ---- Load models once (per outer loop) ----
    print("\nLoading FastModel...")
    fast = FastModel(FastModelConfig()).load_pretrained()
    if args.fast_ckpt and os.path.exists(args.fast_ckpt):
        ckpt = torch.load(args.fast_ckpt, map_location="cuda", weights_only=False)
        if "action_head_state" in ckpt:
            fast.action_head.load_state_dict(ckpt["action_head_state"])
            print(f"  loaded action_head from {args.fast_ckpt}")
        if "xattn_state" in ckpt:
            for k, st in ckpt["xattn_state"].items():
                fast.xattn_layers[k].load_state_dict(st)
            print(f"  loaded bridge xattn from {args.fast_ckpt}")
    if args.bridge_ckpt and os.path.exists(args.bridge_ckpt):
        ckpt = torch.load(args.bridge_ckpt, map_location="cuda", weights_only=False)
        for k, st in ckpt["xattn_state"].items():
            fast.xattn_layers[k].load_state_dict(st)
        print(f"  loaded bridge xattn from {args.bridge_ckpt}")

    needs_slow = any(s in strategies for s in ("T", "L"))
    slow = None
    if needs_slow:
        print("Loading SlowModel...")
        slow = SlowModel(SlowModelConfig()).load_pretrained()

    # ---- Run all cells ----
    cells = []
    t_start = time.time()
    for strategy in strategies:
        for game in games:
            for seed in seeds:
                for ep in range(episodes):
                    cell_seed = seed * 1000 + ep  # distinct seed per episode
                    print(f"  [{strategy}/{game}/seed{cell_seed}] running...", flush=True)
                    result = _run_episode(
                        strategy, game, cell_seed,
                        max_ticks=args.max_ticks,
                        fast=fast, slow=slow,
                        fast_checkpoint=args.fast_ckpt,
                        stop_action_policy="argmax",
                        max_slow_tokens=args.max_slow_tokens,
                    )
                    cells.append(result)
                    print(f"    score={result['score']:.0f} ticks={result['ticks']} "
                          f"latency={result['mean_action_latency_ms']:.0f}ms "
                          f"on_clock={result['frac_on_clock']:.0%}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal eval time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    summary = _aggregate(cells)
    output = {
        "cells": cells,
        "summary": summary,
        "config": {"strategies": strategies, "games": games, "seeds": seeds,
                   "episodes": episodes, "max_ticks": args.max_ticks,
                   "fast_ckpt": args.fast_ckpt, "bridge_ckpt": args.bridge_ckpt},
        "elapsed_sec": elapsed,
    }
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.out}")

    print("\nSummary:")
    for k, v in sorted(summary.items()):
        print(f"  {k}: score={v['mean_score']:.1f}±{v['std_score']:.1f} "
              f"(median {v['median_score']:.1f}, n={v['n_episodes']})  "
              f"latency={v['mean_action_latency_ms']:.0f}ms "
              f"on_clock={v['frac_on_clock']:.0%}")


if __name__ == "__main__":
    main()
