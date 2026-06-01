"""Pretrained-PPO-expert trajectory collection for MetaDrive Stage A BC.

Uses metadrive.examples.expert (a strong pretrained PPO policy: gate showed
reward 215 vs 6.4 random, route-completion 0.79). Continuous expert actions are
mapped to the nearest discrete action (metadrive_wrapper.expert_action_idx).

Output schema matches the Atari collector (episodes -> per-tick obs/action/reward),
actions stored as LOCAL indices 0..8. Run under xvfb (DISPLAY set) for GL.

Usage:
  xvfb-run -a python scripts/collect_metadrive_expert.py --episodes 50 \
      --max-ticks 500 --out results/trajectories_MetaDrive.pt
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.metadrive_wrapper import MetaDriveWrapper, expert_action_idx


def collect(episodes, seed0, max_ticks, validate):
    eps = []; rewards = []; lengths = []; arrived = 0
    for ep in range(episodes):
        seed = seed0 + ep
        env = MetaDriveWrapper(seed=seed)
        frame, _ = env.reset(seed=seed)
        obs_l, act_l, rew_l, done_l = [], [], [], []
        tot = 0.0; info = {}
        for t in range(max_ticks):
            a = expert_action_idx(env)
            obs_l.append(frame.astype(np.uint8)); act_l.append(int(a))
            frame, r, term, trunc, _ = env.step(a)
            rew_l.append(float(r)); done_l.append(bool(term or trunc)); tot += r
            if term or trunc:
                break
        # arrive flag from the underlying env info isn't returned by wrapper.step;
        # approximate "arrived" by positive terminal reward tail
        env.close()
        rewards.append(tot); lengths.append(len(act_l))
        if not validate and obs_l:
            eps.append({"obs": np.stack(obs_l).astype(np.uint8),
                        "action": np.array(act_l, dtype=np.int8),
                        "reward": np.array(rew_l, dtype=np.float32),
                        "done": np.array(done_l, dtype=bool)})
        if (ep + 1) % 10 == 0:
            print(f"  {ep+1}/{episodes}: reward {np.mean(rewards):.1f} len {np.mean(lengths):.0f}", flush=True)
    print(f"\nExpert: reward {np.mean(rewards):.2f}+-{np.std(rewards):.2f}  mean_len {np.mean(lengths):.0f}", flush=True)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=500)
    ap.add_argument("--out", default="results/trajectories_MetaDrive.pt")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    eps = collect(args.episodes, args.seed, args.max_ticks, args.validate)
    if args.validate:
        print("validate-only: no file written"); return
    blob = {"game": "MetaDrive", "episodes": eps}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(blob, args.out)
    n = sum(len(e["action"]) for e in eps)
    print(f"Wrote {args.out}  ({len(eps)} episodes, {n} transitions)")


if __name__ == "__main__":
    main()
