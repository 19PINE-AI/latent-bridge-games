"""Random-policy floor benchmark on ALE games.

Runs N episodes per game with a uniform-random action policy and records mean/std
score plus episode length. Establishes the absolute floor that any trained model
must beat; useful for sanity-checking the eval harness independent of any ML
machinery.

Usage:
    python scripts/random_baseline.py                       # default: 30 episodes, all eval games
    python scripts/random_baseline.py --episodes 5          # quick run
    python scripts/random_baseline.py --games MsPacman      # one game

Output: results/random_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_GAMES = [
    "Pong", "Breakout", "BeamRider",
    "MsPacman", "SpaceInvaders", "Qbert",
    "Frostbite", "Pitfall", "PrivateEye",
]


def run_episode(env, max_steps: int, rng: np.random.Generator) -> tuple[float, int]:
    env.reset()
    total_reward = 0.0
    steps = 0
    n_actions = int(env.action_space.n)
    while steps < max_steps:
        action = int(rng.integers(0, n_actions))
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return total_reward, steps


def benchmark_game(game: str, episodes: int, max_steps: int, base_seed: int = 0) -> dict:
    import gymnasium as gym
    import ale_py  # noqa: F401

    env = gym.make(f"ALE/{game}-v5", frameskip=4)  # ALE-standard 4-frame skip
    scores, lengths = [], []
    t0 = time.time()
    for ep in range(episodes):
        env.reset(seed=base_seed + ep)
        rng = np.random.default_rng(base_seed + ep)
        score, length = run_episode(env, max_steps, rng)
        scores.append(score)
        lengths.append(length)
    env.close()
    elapsed = time.time() - t0
    return {
        "game": game,
        "episodes": episodes,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "mean_length": float(np.mean(lengths)),
        "elapsed_sec": elapsed,
        "scores": scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=DEFAULT_GAMES)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=27000 // 4)  # 27k 15Hz ticks ÷ 4-frame skip
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/random_baseline.json")
    args = ap.parse_args()

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    results = {}
    for game in args.games:
        print(f"[random-baseline] running {game}: {args.episodes} episodes...", flush=True)
        try:
            results[game] = benchmark_game(game, args.episodes, args.max_steps, args.seed)
            r = results[game]
            print(f"  -> mean={r['mean_score']:.1f} ± {r['std_score']:.1f}  "
                  f"len={r['mean_length']:.0f}  elapsed={r['elapsed_sec']:.1f}s", flush=True)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            results[game] = {"error": str(exc)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[random-baseline] wrote {args.out}")


if __name__ == "__main__":
    main()
