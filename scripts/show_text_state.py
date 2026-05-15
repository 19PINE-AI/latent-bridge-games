"""Print the decoded TextState at each 1Hz emission for a short gameplay session.

Used to visually verify the RAM decoders produce sensible game-state. Run:
    python scripts/show_text_state.py --game MsPacman --ticks 150
    python scripts/show_text_state.py --game Frostbite --ticks 150 --action 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pprint import pformat

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.atari_wrapper import AtariEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--ticks", type=int, default=150,  # 10 seconds at 15Hz
                    help="number of 15Hz fast ticks to run")
    ap.add_argument("--action", type=int, default=-1,
                    help="fixed action (-1 = random)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = AtariEnv(game_name=args.game, seed=args.seed)
    _, t0 = env.reset()

    print(f"=== {args.game} initial text-state ===")
    print(pformat(t0.__dict__, width=100, sort_dicts=False))
    print()

    n_actions = env.action_space_size
    for t in range(args.ticks):
        action = args.action if args.action >= 0 else int(rng.integers(0, n_actions))
        _, reward, terminated, truncated, text = env.step(action)
        if text is not None:
            print(f"--- tick {t+1}  (≈{(t+1)/15:.1f}s)  reward_since_last_emission=? ---")
            print(pformat(text.__dict__, width=100, sort_dicts=False))
            print()
        if terminated or truncated:
            print(f"[episode ended at tick {t+1}]")
            break
    env.close()


if __name__ == "__main__":
    main()
