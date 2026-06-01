"""Scripted-expert trajectory collection for highway-env Stage A BC.

The expert (rule-based IDM-style controller in highway_wrapper.scripted_expert_action)
earns dense reward 24.8 vs 8.1 random with 1/8 vs 7/8 crash rate, so it is a strong
teacher. Output schema matches the Atari collector (episodes -> per-tick obs/action/
reward) so the existing Stage A loader works unchanged. Actions stored as LOCAL
indices (0..8); the dataset maps local->global via GAME_ACTION_TO_GLOBAL["Highway"].

Usage:
  python scripts/collect_highway_expert.py --episodes 40 --max-ticks 200 \
      --out results/trajectories_Highway.pt --validate
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.highway_wrapper import HighwayEnv, scripted_expert_action


def collect(episodes, seed0, max_ticks, validate):
    eps = []
    rewards = []; lengths = []; crashes = 0
    for ep in range(episodes):
        seed = seed0 + ep
        env = HighwayEnv(seed=seed)
        frame, _ = env.reset(seed=seed)
        obs_l, act_l, rew_l, done_l = [], [], [], []
        tot = 0.0; crashed = False
        for t in range(max_ticks):
            a = scripted_expert_action(env.unwrapped)
            obs_l.append(frame.astype(np.uint8)); act_l.append(int(a))
            frame, r, term, trunc, _ = env.step(a)
            rew_l.append(float(r)); done_l.append(bool(term or trunc)); tot += r
            if term or trunc:
                crashed = term; break
        env.close()
        rewards.append(tot); lengths.append(len(act_l)); crashes += int(crashed)
        if not validate and obs_l:
            eps.append({"obs": np.stack(obs_l).astype(np.uint8),
                        "action": np.array(act_l, dtype=np.int8),
                        "reward": np.array(rew_l, dtype=np.float32),
                        "done": np.array(done_l, dtype=bool)})
        if (ep + 1) % 10 == 0:
            print(f"  {ep+1}/{episodes}: reward {np.mean(rewards):.1f} len {np.mean(lengths):.0f} crashes {crashes}")
    print(f"\nExpert: reward {np.mean(rewards):.2f}±{np.std(rewards):.2f}  "
          f"mean_len {np.mean(lengths):.0f}  crashes {crashes}/{episodes}")
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=200)
    ap.add_argument("--out", default="results/trajectories_Highway.pt")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    eps = collect(args.episodes, args.seed, args.max_ticks, args.validate)
    if args.validate:
        print("validate-only: no file written"); return
    blob = {"game": "Highway", "episodes": eps}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(blob, args.out)
    n = sum(len(e["action"]) for e in eps)
    print(f"Wrote {args.out}  ({len(eps)} episodes, {n} transitions)")


if __name__ == "__main__":
    main()
