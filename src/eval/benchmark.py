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
from collections import deque
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
                 action_temperature: float = 1.0,
                 max_slow_tokens: int = 64,
                 vision_refresh_every: int = 1,
                 trace_dir: Optional[str] = None,
                 text_suffix_window: int = 1,
                 bridge_replace: str = "none",
                 ) -> dict:
    """Run one episode under one strategy on one (game, seed) cell. Returns metrics."""
    legal = legal_action_mask(game)
    legal_indices = legal.nonzero(as_tuple=True)[0].tolist()
    rng = np.random.default_rng(seed)

    if game.startswith("MiniGrid"):
        from src.env.minigrid_wrapper import MiniGridEnv
        env_id = game if game != "MiniGrid" else "MiniGrid-FourRooms-v0"
        env = MiniGridEnv(game_name=env_id, seed=seed)
    elif game == "Highway":
        from src.env.highway_wrapper import HighwayEnv
        env = HighwayEnv(seed=seed)
    elif game == "MetaDrive":
        from src.env.metadrive_wrapper import MetaDriveWrapper
        env = MetaDriveWrapper(seed=seed)
    else:
        env = AtariEnv(game_name=game, seed=seed)
    # AtariEnv.reset() takes no seed (seeded via constructor); the MetaDrive/Highway/
    # MiniGrid wrappers accept reset(seed=...). Dispatch by signature so both work.
    import inspect
    if "seed" in inspect.signature(env.reset).parameters:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    if vision_refresh_every > 1:
        fast.reset_vision_cache()

    # Trace logging — for demo MP4 / live playback
    trace_path: Optional[Path] = None
    trace_events: list[dict] = []
    if trace_dir:
        trace_path = Path(trace_dir) / f"{strategy}_{game}_seed{seed}"
        trace_path.mkdir(parents=True, exist_ok=True)

    latest_slow_text: Optional[str] = None
    # v2: bridge tokens from the most-recent slow emission (no long buffer)
    latest_bridge_tokens: Optional[torch.Tensor] = None
    # Rolling window of the last N slow emissions (for --text-suffix-window > 1)
    recent_slow_texts: deque[str] = deque(maxlen=max(1, text_suffix_window))
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
                    obs, thought_tokens=None,
                    legal_action_mask=legal,
                    vision_refresh_every=vision_refresh_every,
                )
        elif strategy == "T":
            # If text_suffix_window > 1, concatenate the last N emissions
            # (oldest first → newest last) so the head sees a longer T budget.
            if text_suffix_window > 1 and len(recent_slow_texts) > 1:
                suffix = " ".join(recent_slow_texts)
            else:
                suffix = latest_slow_text
            with torch.no_grad():
                logits = fast.predict_action(
                    obs, thought_tokens=None,
                    legal_action_mask=legal,
                    slow_text_suffix=suffix,
                    vision_refresh_every=vision_refresh_every,
                )
        elif strategy == "L":
            with torch.no_grad():
                logits = fast.predict_action(
                    obs, thought_tokens=latest_bridge_tokens,
                    legal_action_mask=legal,
                    slow_text_suffix=None,  # v2 L mode uses latents only
                    vision_refresh_every=vision_refresh_every,
                )
        elif strategy == "B":
            # Both channels at once: text suffix in the prompt AND latent tokens prepended.
            if text_suffix_window > 1 and len(recent_slow_texts) > 1:
                suffix = " ".join(recent_slow_texts)
            else:
                suffix = latest_slow_text
            with torch.no_grad():
                logits = fast.predict_action(
                    obs, thought_tokens=latest_bridge_tokens,
                    legal_action_mask=legal,
                    slow_text_suffix=suffix,
                    vision_refresh_every=vision_refresh_every,
                )
        else:
            raise ValueError(f"unsupported strategy {strategy!r}")

        if stop_action_policy == "argmax":
            global_action = int(logits.argmax(dim=-1).item())
        elif stop_action_policy == "sample":
            # Temperature-scaled sampling: lower T = sharper, higher T = more diffuse
            scaled = logits / max(action_temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1).squeeze(0)
            global_action = int(torch.multinomial(probs, 1).item())
        else:
            global_action = int(rng.choice(legal_indices))
        torch.cuda.synchronize()
        tick_latencies_ms.append((time.perf_counter() - t_tick) * 1000.0)

        local_action = global_to_local_action(game, global_action)
        # Capture pre-step frame for the trace so the (frame, action) pair is aligned
        prev_obs = obs
        obs, reward, terminated, truncated, text_state = env.step(local_action)
        cumulative_reward += float(reward)
        n_ticks += 1

        # --- Slow emission (only T/L) ---
        emit_text_this_tick = None
        if text_state is not None and slow is not None and strategy in ("T", "L", "B"):
            messages = build_slow_model_messages(game, text_state,
                                                 prior_thought=latest_slow_text)
            # v2: emit returns N bridge tokens (in fast LLM embedding space)
            emit_text, emit_bridge = slow.emit(
                messages, frame=obs, max_new_tokens=max_slow_tokens,
                return_raw_residuals=False,
            )
            latest_slow_text = emit_text
            recent_slow_texts.append(emit_text)
            emit_text_this_tick = emit_text
            if emit_bridge.numel() > 0:
                bridge = emit_bridge.unsqueeze(0)  # [1, N, D]
                if bridge_replace == "random":
                    # Replace with random tokens at matched per-position L2 norm.
                    norms = bridge.norm(dim=-1, keepdim=True)  # [1, N, 1]
                    rand = torch.randn_like(bridge)
                    rand = rand * (norms / (rand.norm(dim=-1, keepdim=True) + 1e-9))
                    bridge = rand
                elif bridge_replace == "zero":
                    bridge = torch.zeros_like(bridge)
                # "none" → use trained bridge as is
                latest_bridge_tokens = bridge
            n_slow_emissions += 1

        # --- Trace logging ---
        if trace_path is not None:
            np.savez_compressed(trace_path / f"frame_{t:05d}.npz", frame=prev_obs)
            trace_events.append({
                "tick": t,
                "action": global_action,
                "local_action": local_action,
                "reward": float(reward),
                "cumulative_reward": cumulative_reward,
                "latest_slow_text": latest_slow_text,
                "new_emission": emit_text_this_tick,
            })

        if terminated or truncated:
            break

    env.close()
    if trace_path is not None:
        (trace_path / "events.json").write_text(json.dumps({
            "strategy": strategy, "game": game, "seed": seed,
            "final_score": cumulative_reward,
            "n_ticks": n_ticks,
            "tick_hz": 15,
            "events": trace_events,
        }))
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
    ap.add_argument("--strategies", nargs="+", choices=("F", "T", "L", "B"), default=None,
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
    ap.add_argument("--vision-refresh-every", type=int, default=1,
                    help="Refresh the vision tower (SigLIP + resampler) every N "
                         "fast ticks; 1 = every tick (default). Higher = faster but "
                         "the model sees stale visual context.")
    ap.add_argument("--text-suffix-window", type=int, default=1,
                    help="For strategy T: how many of the most-recent slow "
                         "emissions to concatenate as the text suffix. "
                         "Default 1 (paper baseline). Higher gives T more text "
                         "budget for the bandwidth-thesis falsifier.")
    ap.add_argument("--action-policy", choices=("argmax", "sample"),
                    default="argmax",
                    help="Action selection: 'argmax' (paper default; "
                         "deterministic given env seed) or 'sample' (multinomial "
                         "from softmax(logits/temperature)).")
    ap.add_argument("--action-temperature", type=float, default=1.0,
                    help="Temperature for sampling (only used when "
                         "--action-policy=sample). Default 1.0.")
    ap.add_argument("--bridge-replace", choices=("none", "random", "zero"),
                    default="none",
                    help="L strategy control: 'none' (paper default), "
                         "'random' replaces trained bridge tokens with random "
                         "embeddings at the same per-position L2 norm, 'zero' "
                         "replaces with zeros. Tests whether trained bridge "
                         "tokens carry information.")
    ap.add_argument("--save-trace-dir", default=None,
                    help="If set, per-episode subdirectories are created under this "
                         "path containing per-tick frame_XXXXX.npz files and an "
                         "events.json with actions, rewards, and slow-text emissions. "
                         "Foundation for the demo MP4 renderer and live playback.")
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

    # ---- Determine FastModel config (gated or not) from the bridge ckpt ----
    fast_cfg = FastModelConfig()
    bridge_ckpt_is_v2 = False
    if args.bridge_ckpt and os.path.exists(args.bridge_ckpt):
        ck = torch.load(args.bridge_ckpt, map_location="cpu", weights_only=False)
        bridge_ckpt_is_v2 = bool(ck.get("v2", False))
        if bridge_ckpt_is_v2:
            print(f"[config] bridge_ckpt is v2 (LLaVA-style latent-as-token).")
        else:
            cfg_saved = ck.get("config", {})
            if cfg_saved.get("xattn_gated"):
                fast_cfg.xattn_gated = True
                print(f"[config] bridge_ckpt was trained with xattn_gated=True (v1); "
                      f"instantiating FastModel accordingly.")

    print("\nLoading FastModel...")
    fast = FastModel(fast_cfg).load_pretrained()
    if args.fast_ckpt and os.path.exists(args.fast_ckpt):
        ckpt = torch.load(args.fast_ckpt, map_location="cuda", weights_only=False)
        if "action_head_state" in ckpt:
            fast.action_head.load_state_dict(ckpt["action_head_state"])
            print(f"  loaded action_head from {args.fast_ckpt}")
        if "xattn_state" in ckpt:
            for k, st in ckpt["xattn_state"].items():
                fast.xattn_layers[k].load_state_dict(st)
            print(f"  loaded v1 bridge xattn from {args.fast_ckpt}")
    if args.bridge_ckpt and os.path.exists(args.bridge_ckpt):
        ckpt = torch.load(args.bridge_ckpt, map_location="cuda", weights_only=False)
        if not bridge_ckpt_is_v2:
            # v1 path
            for k, st in ckpt["xattn_state"].items():
                fast.xattn_layers[k].load_state_dict(st)
            print(f"  loaded v1 bridge xattn from {args.bridge_ckpt}")
            if "action_head_state" in ckpt:
                fast.action_head.load_state_dict(ckpt["action_head_state"])
                print(f"  loaded action_head from {args.bridge_ckpt} (overrides fast-ckpt)")

    needs_slow = any(s in strategies for s in ("T", "L", "B"))
    slow = None
    if needs_slow:
        print("Loading SlowModel...")
        slow = SlowModel(SlowModelConfig()).load_pretrained()
        # v2: load slow's trained ThoughtProjection from the bridge checkpoint
        if bridge_ckpt_is_v2 and args.bridge_ckpt:
            ck = torch.load(args.bridge_ckpt, map_location="cuda", weights_only=False)
            slow.projection.load_state_dict(ck["slow_projection_state"])
            print(f"  loaded v2 slow.projection from {args.bridge_ckpt}")
            # Stage D PPO checkpoints ALSO include action_head_state because
            # PPO trains it under the deployment distribution. If present in
            # the bridge ckpt, override the fast-ckpt action_head (more recent).
            if "action_head_state" in ck:
                fast.action_head.load_state_dict(ck["action_head_state"])
                print(f"  loaded action_head from {args.bridge_ckpt} (Stage D PPO; overrides fast-ckpt)")

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
                        stop_action_policy=args.action_policy,
                        action_temperature=args.action_temperature,
                        max_slow_tokens=args.max_slow_tokens,
                        vision_refresh_every=args.vision_refresh_every,
                        trace_dir=args.save_trace_dir,
                        text_suffix_window=args.text_suffix_window,
                        bridge_replace=args.bridge_replace,
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
