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
    ap.add_argument("--action-policy", choices=("zero-head", "random-legal", "sb3-expert"),
                    default="random-legal",
                    help="At init the fast action_head outputs zeros; 'random-legal' "
                         "samples from legal actions for varied demo data. "
                         "'sb3-expert' uses --expert-policy with --epsilon noise (for "
                         "reward-asymmetric games like SpaceInvaders where random "
                         "sampling under-represents reward-bearing actions).")
    ap.add_argument("--expert-policy", default=None,
                    help="Path to SB3 DQN/PPO .zip; required when action-policy=sb3-expert")
    ap.add_argument("--epsilon", type=float, default=0.1,
                    help="exploration rate around the SB3 expert (action-policy=sb3-expert)")
    ap.add_argument("--load-slow", action="store_true", default=True,
                    help="Load Qwen3-VL-8B-Thinking and emit real strategic guidance.")
    ap.add_argument("--no-slow", dest="load_slow", action="store_false",
                    help="Skip slow model (fast-only baseline; F condition).")
    ap.add_argument("--episodes", type=int, default=1,
                    help="Number of episodes to run (each with seed+i). Models loaded once.")
    ap.add_argument("--out-dir", default=None,
                    help="When --episodes > 1, write to <out-dir>/<game>_seed{i}.pt instead of --out.")
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

    # ---- Optional SB3 expert ----
    sb3_policy = None
    if args.action_policy == "sb3-expert":
        if not args.expert_policy:
            raise ValueError("--expert-policy required when action-policy=sb3-expert")
        from scripts.collect_trajectories import _load_sb3_policy, _build_sb3_view_env
        print(f"Loading SB3 expert from {args.expert_policy}...", flush=True)
        sb3_policy = _load_sb3_policy(args.expert_policy)
        print(f"  done.", flush=True)

    # ---- Bridge state (recreated per episode) ----
    bridge_dim = 256

    episode_summaries = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        rng = np.random.default_rng(ep_seed)
        thought_buffer = ThoughtBuffer(
            capacity=16, dim=bridge_dim, device="cuda", dtype=torch.bfloat16,
        )

        env = AtariEnv(game_name=args.game, seed=ep_seed)
        obs, _ = env.reset()

        sb3_view_env = None
        sb3_obs = None
        if sb3_policy is not None:
            sb3_view_env = _build_sb3_view_env(args.game, ep_seed)
            sb3_obs = sb3_view_env.reset()

        trace = []
        latest_slow_text: str | None = None
        cumulative_reward = 0.0
        t_loop_start = time.time()

        for t in range(args.ticks):
            # 1) Predict fast-model action logits (frame-conditioned)
            with torch.no_grad():
                logits = fast.predict_action(
                    obs,
                    thought_buffer=None,  # T-mode: bridge present-but-silent
                    legal_action_mask=legal,
                    slow_text_suffix=latest_slow_text,
                )
            if args.action_policy == "zero-head":
                global_action = int(logits.argmax(dim=-1).item())
            elif args.action_policy == "sb3-expert":
                # SB3 expert predicts in local action space; map back to global.
                if rng.random() < args.epsilon:
                    local_action = int(rng.integers(0, env.action_space_size))
                else:
                    pred, _ = sb3_policy.predict(sb3_obs, deterministic=True)
                    local_action = int(pred[0])
                # We track global actions in the trace for consistency with the
                # global action head. Reverse-map local→global via legal_indices.
                from src.training.imitation_data import GAME_ACTION_TO_GLOBAL
                global_action = GAME_ACTION_TO_GLOBAL[args.game][local_action]
            else:
                global_action = int(rng.choice(legal_indices))

            local_action = global_to_local_action(args.game, global_action)
            obs, reward, terminated, truncated, text_state = env.step(local_action)
            if sb3_view_env is not None:
                sb3_obs, _, sb3_done, _ = sb3_view_env.step(np.array([local_action]))
                if bool(sb3_done[0]):
                    sb3_obs = sb3_view_env.reset()
            action = global_action
            cumulative_reward += float(reward)
            slow_emit_text = None
            slow_emit_vecs = None

            if text_state is not None and slow is not None:
                messages = build_slow_model_messages(args.game, text_state,
                                                     prior_thought=latest_slow_text)
                t_slow = time.time()
                # v2: get RAW slow residuals (un-projected, slow_hidden_dim=4096).
                # Stage C will re-project these through a trainable ThoughtProjection.
                slow_emit_text, slow_emit_residuals = slow.emit(
                    messages, frame=obs, max_new_tokens=args.slow_max_tokens,
                    return_raw_residuals=True,
                )
                slow_elapsed = time.time() - t_slow
                latest_slow_text = slow_emit_text
                print(f"  [ep {ep} tick {t+1:3d}] slow emission ({slow_elapsed:.1f}s, "
                      f"residuals shape={tuple(slow_emit_residuals.shape)})", flush=True)
                slow_emit_vecs = slow_emit_residuals  # alias for backward-compat key
            else:
                slow_emit_vecs = None

            trace.append({
                "tick": t,
                "action": action,
                "reward": float(reward),
                "obs": obs.copy(),
                "text_state": vars(text_state) if text_state is not None else None,
                "slow_text": slow_emit_text,
                # v2: contains raw slow residuals at layer 24, last N positions, 4096-d.
                # Stage C will re-project these via a trainable projection.
                "slow_vecs": (slow_emit_vecs.detach().to("cpu", dtype=torch.float32)
                              if slow_emit_vecs is not None else None),
            })

            if terminated or truncated:
                break

        env.close()
        if sb3_view_env is not None:
            sb3_view_env.close()
        elapsed = time.time() - t_loop_start
        n_ticks = len(trace)
        n_slow = sum(1 for x in trace if x["slow_text"])

        # Per-episode summary
        print(f"  [ep {ep} done] ticks={n_ticks} score={cumulative_reward:.0f} "
              f"slow_emissions={n_slow} wall={elapsed:.1f}s", flush=True)
        episode_summaries.append({
            "ep": ep, "seed": ep_seed, "ticks": n_ticks,
            "score": cumulative_reward, "slow_emissions": n_slow,
            "wall_clock": elapsed,
        })

        # Write trajectory file
        if args.out_dir is not None:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"{args.game}_seed{ep_seed}.pt")
        elif args.out:
            base, ext = os.path.splitext(args.out)
            out_path = args.out if args.episodes == 1 else f"{base}_ep{ep}{ext}"
        else:
            out_path = None

        if out_path:
            Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
            torch.save({
                "game": args.game,
                "seed": ep_seed,
                "trace": trace,
                "cumulative_reward": cumulative_reward,
            }, out_path)
            print(f"  wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB)", flush=True)

    # ---- Aggregate summary ----
    print("\n" + "=" * 60)
    print(f"T-condition multi-episode summary ({args.game}, {len(episode_summaries)} episodes):")
    scores = [s["score"] for s in episode_summaries]
    ticks = [s["ticks"] for s in episode_summaries]
    n_emits = [s["slow_emissions"] for s in episode_summaries]
    total_wall = sum(s["wall_clock"] for s in episode_summaries)
    print(f"  mean score: {np.mean(scores):.1f} ± {np.std(scores):.1f}  "
          f"(range [{min(scores):.0f}, {max(scores):.0f}])")
    print(f"  mean ticks: {np.mean(ticks):.0f}")
    print(f"  total slow emissions: {sum(n_emits)}  (mean {np.mean(n_emits):.1f}/ep)")
    print(f"  total wall-clock: {total_wall:.0f}s")


if __name__ == "__main__":
    main()
