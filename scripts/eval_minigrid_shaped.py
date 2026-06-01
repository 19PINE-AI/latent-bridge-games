"""Shaped-metric MiniGrid eval: F/T/L under a dense distance-to-goal metric.

MiniGrid's native reward is sparse (only fires on reaching the goal), so within a
bounded tick budget F/T/L all score 0 (verified: 0/12 nonzero each). We use a
partial-credit metric from the symbolic grid:

  frac_closed = (d_start - d_min) / d_start

d_start = BFS shortest-path distance (cells, respecting walls) start->goal;
d_min   = min such distance over cells the agent visited. 1.0 iff reached goal,
0.5 if it closed half the gap. Also report success_rate (frac_closed==1).

Mirrors src/eval/benchmark.py F/T/L action logic exactly; only scoring differs.
Models loaded from local folders via LB_FAST_MODEL_PATH / LB_SLOW_MODEL_PATH.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import deque, defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.minigrid_wrapper import MiniGridEnv
from src.models.fast_model import FastModel, FastModelConfig
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.imitation_data import legal_action_mask, global_to_local_action
from src.training.prompts import build_slow_model_messages


def cell_bfs_dist(u, goal):
    w, h, grid = u.width, u.height, u.grid
    def passable(x, y):
        if x < 0 or y < 0 or x >= w or y >= h: return False
        c = grid.get(x, y)
        if c is None: return True
        if c.type in ("wall", "lava"): return False
        if c.type == "door": return getattr(c, "is_open", True)
        return True
    dist = {goal: 0}; q = deque([goal])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if (nx,ny) not in dist and passable(nx,ny):
                dist[(nx,ny)] = dist[(x,y)] + 1; q.append((nx,ny))
    return dist


def goal_cell(u):
    for i in range(u.width):
        for j in range(u.height):
            c = u.grid.get(i, j)
            if c is not None and c.type == "goal":
                return (i, j)
    return None


def run_episode(strategy, seed, fast, slow, legal, max_ticks, policy, temp):
    env = MiniGridEnv("MiniGrid-FourRooms-v0", seed=seed)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    goal = goal_cell(u)
    dist = cell_bfs_dist(u, goal)
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]))
    d_start = dist.get(start)
    if not d_start:
        env.close(); return None
    d_min = d_start; latest_text = None; latest_bridge = None; reached = False
    for t in range(max_ticks):
        with torch.no_grad():
            if strategy == "F":
                logits = fast.predict_action(obs, thought_tokens=None, legal_action_mask=legal)
            elif strategy == "T":
                logits = fast.predict_action(obs, thought_tokens=None, legal_action_mask=legal,
                                             slow_text_suffix=latest_text)
            else:
                logits = fast.predict_action(obs, thought_tokens=latest_bridge,
                                             legal_action_mask=legal, slow_text_suffix=None)
        if policy == "argmax":
            ga = int(logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(logits / max(temp, 1e-6), dim=-1).squeeze(0)
            ga = int(torch.multinomial(probs, 1).item())
        la = global_to_local_action("MiniGrid", ga)
        obs, reward, term, trunc, text_state = env.step(la)
        cell = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        if cell in dist:
            d_min = min(d_min, dist[cell])
        if dist.get(cell, 99) == 0 or reward > 0:
            reached = True
        if text_state is not None and slow is not None and strategy in ("T", "L"):
            msgs = build_slow_model_messages("MiniGrid", text_state, prior_thought=latest_text)
            etext, ebridge = slow.emit(msgs, frame=obs, max_new_tokens=64, return_raw_residuals=False)
            latest_text = etext
            if ebridge.numel() > 0:
                latest_bridge = ebridge.unsqueeze(0)
        if term or trunc or reached:
            break
    env.close()
    return {"seed": seed, "d_start": d_start, "d_min": d_min,
            "frac_closed": float((d_start - d_min) / d_start), "reached": bool(reached)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["F", "T", "L"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-ticks", type=int, default=100)
    ap.add_argument("--action-policy", choices=("argmax", "sample"), default="argmax")
    ap.add_argument("--action-temperature", type=float, default=1.0)
    ap.add_argument("--fast-ckpt", required=True)
    ap.add_argument("--bridge-ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    legal = legal_action_mask("MiniGrid")
    print("Loading FastModel...", flush=True)
    fast = FastModel(FastModelConfig()).load_pretrained()
    ck = torch.load(args.fast_ckpt, map_location="cuda", weights_only=False)
    fast.action_head.load_state_dict(ck["action_head_state"])
    slow = None
    if any(s in args.strategies for s in ("T", "L")):
        print("Loading SlowModel...", flush=True)
        slow = SlowModel(SlowModelConfig()).load_pretrained()
        bk = torch.load(args.bridge_ckpt, map_location="cuda", weights_only=False)
        slow.projection.load_state_dict(bk["slow_projection_state"])

    cells = []
    for strat in args.strategies:
        for seed in args.seeds:
            for ep in range(args.episodes):
                cs = seed * 1000 + ep
                r = run_episode(strat, cs, fast, slow, legal, args.max_ticks,
                                args.action_policy, args.action_temperature)
                if r is None: continue
                r["strategy"] = strat; cells.append(r)
                print(f"  [{strat}/seed{cs}] frac_closed={r['frac_closed']:.2f} "
                      f"reached={r['reached']} d {r['d_start']}->{r['d_min']}", flush=True)

    by = defaultdict(list)
    for c in cells: by[c["strategy"]].append(c)
    summary = {}
    for s, rs in by.items():
        fr = np.array([r["frac_closed"] for r in rs])
        sr = np.array([r["reached"] for r in rs], dtype=float)
        summary[s] = {"n": len(rs), "frac_closed_mean": float(fr.mean()),
                      "frac_closed_std": float(fr.std(ddof=1) if len(fr) > 1 else 0),
                      "success_rate": float(sr.mean())}
    out = {"cells": cells, "summary": summary, "config": vars(args)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== SUMMARY ===")
    for s in ["F", "T", "L"]:
        if s in summary:
            v = summary[s]
            print(f"  {s}: frac_closed={v['frac_closed_mean']:.3f}±{v['frac_closed_std']:.3f} "
                  f"success={v['success_rate']:.2f} (n={v['n']})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
