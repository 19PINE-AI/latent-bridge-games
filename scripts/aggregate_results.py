"""Aggregate all eval_*.json files in results/ into a single comparison table.

Produces:
  - Console table (markdown-formatted)
  - results/eval_aggregate.json with the full data

Useful when generating paper figures or comparing across many variants.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --pattern 'results/eval_v2*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/eval_*.json")
    ap.add_argument("--out", default="results/eval_aggregate.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print(f"No files matching {args.pattern!r}")
        return

    rows = []
    by_file = {}
    for path in paths:
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"[skip {path}: {e}]")
            continue
        cells = data.get("cells", [])
        if not cells:
            continue
        by_file[path] = []
        # Group by (strategy, game)
        groups = {}
        for c in cells:
            k = (c.get("strategy"), c.get("game"))
            groups.setdefault(k, []).append(c)
        for (strat, game), runs in groups.items():
            scores = np.array([r.get("score", 0) for r in runs], dtype=float)
            ticks = np.array([r.get("ticks", 0) for r in runs], dtype=float)
            lats = np.array([r.get("mean_action_latency_ms", 0) for r in runs], dtype=float)
            on_clock = np.array([r.get("frac_on_clock", 0) for r in runs], dtype=float)
            row = {
                "file": os.path.basename(path),
                "strategy": strat,
                "game": game,
                "n_episodes": len(runs),
                "mean_score": float(scores.mean()),
                "std_score": float(scores.std()),
                "median_score": float(np.median(scores)),
                "min_score": float(scores.min()) if len(scores) else 0,
                "max_score": float(scores.max()) if len(scores) else 0,
                "mean_ticks": float(ticks.mean()),
                "mean_latency_ms": float(lats.mean()),
                "frac_on_clock": float(on_clock.mean()),
            }
            rows.append(row)
            by_file[path].append(row)

    if not rows:
        print("No rows extracted.")
        return

    # Sort: by game, then strategy F/T/L
    strat_order = {"F": 0, "T": 1, "L": 2, "S": 3, "O": 4}
    rows.sort(key=lambda r: (r["game"], strat_order.get(r["strategy"], 99), r["file"]))

    # Console output
    print(f"\nAggregated {len(rows)} (strategy, game) cells from {len(by_file)} eval file(s)")
    print()
    headers = ["file", "game", "strat", "n", "mean ± std", "median", "[min, max]",
               "lat ms", "on-clock"]
    widths = [38, 15, 5, 3, 16, 7, 14, 7, 8]
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        cells = [
            r["file"][-38:],
            r["game"],
            r["strategy"],
            str(r["n_episodes"]),
            f"{r['mean_score']:.0f} ± {r['std_score']:.0f}",
            f"{r['median_score']:.0f}",
            f"[{r['min_score']:.0f}, {r['max_score']:.0f}]",
            f"{r['mean_latency_ms']:.0f}",
            f"{r['frac_on_clock']:.0%}",
        ]
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |")

    # Write JSON
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "by_file": by_file}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
