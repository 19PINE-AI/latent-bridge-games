"""A* expert + trajectory collection for MiniGrid (Stage A behavioral cloning).

The expert is a provably-optimal BFS planner over (position, direction) states:
actions are {turn_left, turn_right, forward}; forward only succeeds into a
non-wall, non-lava cell (doors are passable when open). Because MiniGrid exposes
the symbolic grid, the planner needs no pixel parsing and is guaranteed optimal,
which removes the weak-Stage-A failure mode we hit on several Atari games.

Output schema matches the Atari collector so the existing Stage A / Stage C
loaders work unchanged:
    {"game": "MiniGrid", "trace": [{"obs": HWC uint8, "action": global_idx,
                                    "reward": float}, ...]}

Usage:
    python scripts/collect_minigrid_expert.py --env MiniGrid-FourRooms-v0 \\
        --episodes 30 --out results/trajectories_MiniGrid.pt --validate
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.minigrid_wrapper import MiniGridEnv, _DIR_VEC
from src.training.imitation_data import GAME_ACTION_TO_GLOBAL

# MiniGrid native nav actions
TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2
MINIGRID_GLOBAL = GAME_ACTION_TO_GLOBAL["MiniGrid"]  # [g_left, g_right, g_forward]


def _passable(grid, x, y, w, h):
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    c = grid.get(x, y)
    if c is None:
        return True
    if c.type in ("wall", "lava"):
        return False
    if c.type == "door":
        return getattr(c, "is_open", True)
    return True  # goal, floor, key, etc. are steppable


def astar_plan(unwrapped):
    """BFS over (x,y,dir) from agent to goal. Returns the optimal native action
    list, or None if unreachable."""
    w, h = unwrapped.width, unwrapped.height
    grid = unwrapped.grid
    start = (int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1]), int(unwrapped.agent_dir))
    # goal cell
    goal = None
    for i in range(w):
        for j in range(h):
            c = grid.get(i, j)
            if c is not None and c.type == "goal":
                goal = (i, j)
    if goal is None:
        return None

    q = deque([start])
    came = {start: None}  # state -> (prev_state, action)
    while q:
        x, y, d = q.popleft()
        if (x, y) == goal:
            # reconstruct
            actions = []
            cur = (x, y, d)
            while came[cur] is not None:
                prev, act = came[cur]
                actions.append(act)
                cur = prev
            return actions[::-1]
        # turn_left, turn_right, forward
        nbrs = [
            ((x, y, (d - 1) % 4), TURN_LEFT),
            ((x, y, (d + 1) % 4), TURN_RIGHT),
        ]
        dx, dy = _DIR_VEC[d]
        fx, fy = x + dx, y + dy
        if _passable(grid, fx, fy, w, h):
            nbrs.append(((fx, fy, d), FORWARD))
        for ns, act in nbrs:
            if ns not in came:
                came[ns] = ((x, y, d), act)
                q.append(ns)
    return None


def collect(env_id: str, episodes: int, seed0: int, validate_only: bool):
    # ImitationDataset (Stage A) expects blob["episodes"], each a dict of per-tick
    # arrays, with "action" stored as the game's *local* (native) index — the
    # dataset maps local->global via GAME_ACTION_TO_GLOBAL itself.
    eps_out = []
    n_solved = 0
    steps_to_solve = []
    for ep in range(episodes):
        seed = seed0 + ep
        env = MiniGridEnv(env_id, seed=seed)
        frame, _ = env.reset(seed=seed)
        plan = astar_plan(env.unwrapped)
        if plan is None:
            print(f"  ep{ep} seed{seed}: NO PLAN (unreachable) — skipping")
            env.close()
            continue
        obs_l, act_l, rew_l, done_l = [], [], [], []
        solved = False
        for native_act in plan:  # native_act in {0,1,2}
            obs_l.append(frame.astype(np.uint8))
            act_l.append(int(native_act))  # store LOCAL index
            frame, reward, term, trunc, _ = env.step(native_act)
            rew_l.append(float(reward))
            done_l.append(bool(term or trunc))
            if term or trunc:
                solved = bool(term and reward > 0)
                break
        if not validate_only and obs_l:
            eps_out.append({
                "obs": np.stack(obs_l, axis=0).astype(np.uint8),
                "action": np.array(act_l, dtype=np.int8),
                "reward": np.array(rew_l, dtype=np.float32),
                "done": np.array(done_l, dtype=bool),
            })
        if solved:
            n_solved += 1
            steps_to_solve.append(len(plan))
        env.close()
        if (ep + 1) % 10 == 0:
            print(f"  {ep+1}/{episodes} episodes; solved so far {n_solved}")

    sr = n_solved / max(episodes, 1)
    mean_steps = float(np.mean(steps_to_solve)) if steps_to_solve else 0.0
    print(f"\nA* expert solve rate: {n_solved}/{episodes} = {sr:.1%}  "
          f"(mean optimal steps {mean_steps:.1f})")
    return eps_out, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-FourRooms-v0")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/trajectories_MiniGrid.pt")
    ap.add_argument("--validate", action="store_true",
                    help="Only measure solve rate; don't write trajectories.")
    args = ap.parse_args()

    eps_out, sr = collect(args.env, args.episodes, args.seed, args.validate)

    if args.validate:
        print("validate-only: no file written")
        if sr < 0.95:
            print(f"WARNING: solve rate {sr:.1%} < 95% — expert/parser may be buggy")
        return

    blob = {"game": "MiniGrid", "episodes": eps_out, "env_id": args.env}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(blob, args.out)
    n_trans = sum(len(e["action"]) for e in eps_out)
    print(f"Wrote {args.out}  ({len(eps_out)} episodes, {n_trans} transitions)")


if __name__ == "__main__":
    main()
