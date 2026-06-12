"""Naive straight-driving baseline on the MetaDrive planning map.

Backs the paper claim that the route expert beats a straight-driving policy on
the planning-heavy map (LB_MD_MAP=SXSXSX, out_of_route_done=True). The policy
is constant (steer=0, throttle=accelerate) -- DISCRETE_ACTIONS index 5 -- i.e.
it never takes route decisions. Same episode/seed/tick protocol as the expert
collection run (scripts/run_metadrive_planning.sh: 40 episodes, seeds 0..39,
max 600 ticks).

Usage:
  LB_MD_MAP=SXSXSX SDL_VIDEODRIVER=dummy python3 scripts/eval_metadrive_straight.py \
      --episodes 40 --seed 0 --max-ticks 600 --out results/eval_md_plan_straight.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.metadrive_wrapper import MetaDriveWrapper, DISCRETE_ACTIONS

STRAIGHT_ACTION = DISCRETE_ACTIONS.index((0.0, 0.5))  # steer straight, accelerate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=600)
    ap.add_argument("--out", default="results/eval_md_plan_straight.json")
    args = ap.parse_args()

    rewards, lengths = [], []
    for ep in range(args.episodes):
        seed = args.seed + ep
        env = MetaDriveWrapper(seed=seed)
        env.reset(seed=seed)
        tot = 0.0
        t = 0
        for t in range(args.max_ticks):
            _, r, term, trunc, _ = env.step(STRAIGHT_ACTION)
            tot += r
            if term or trunc:
                break
        env.close()
        rewards.append(tot)
        lengths.append(t + 1)
        print(f"  ep {ep:02d} seed {seed}: reward {tot:.1f} len {t + 1}", flush=True)

    out = {
        "policy": "naive_straight (steer=0, throttle=0.5, constant)",
        "map": os.environ.get("LB_MD_MAP", "3"),
        "episodes": args.episodes,
        "seed0": args.seed,
        "max_ticks": args.max_ticks,
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nStraight baseline: {out['mean_reward']:.2f}+-{out['std_reward']:.2f}"
          f"  mean_len {np.mean(lengths):.0f}\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
