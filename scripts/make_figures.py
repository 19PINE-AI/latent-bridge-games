"""Produce paper-quality matplotlib figures from eval JSON files.

Outputs (PNG, 300 DPI):
  - figs/main_bar.png         — F vs T vs L mean scores (per game, all eval files)
  - figs/main_box.png         — F/T/L score distribution boxplots
  - figs/latency.png          — per-strategy mean latency (ms) with 67ms target line
  - figs/v1_vs_v2.png         — Bridge architecture comparison (v1 cross-attn vs v2 LLaVA)

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --eval-files results/eval_v2_*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib required: pip install matplotlib")


STRAT_ORDER = ["F", "T", "L"]
STRAT_COLORS = {"F": "#7f8fa6", "T": "#3498db", "L": "#e74c3c"}
STRAT_LABELS = {"F": "F (fast-only)", "T": "T (text bridge)", "L": "L (latent bridge)"}


def _load_cells(paths: list[str]) -> dict:
    """Return {(file_basename, strategy, game): [scores...], ...}."""
    out = {}
    for p in paths:
        try:
            data = json.load(open(p))
        except Exception:
            continue
        for c in data.get("cells", []):
            k = (os.path.basename(p), c.get("strategy"), c.get("game"))
            out.setdefault(k, []).append(c.get("score", 0))
    return out


def fig_main_bar(cells: dict, out: str):
    """Per-game bar chart, F vs T vs L, using the v2 eval file for each game."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Prefer the v2 file per game
    v2_keys = [k for k in cells if "eval_v2_" in k[0]]
    games = sorted({k[2] for k in v2_keys})
    if not games:
        print("no v2 eval files found; using all")
        v2_keys = list(cells.keys())
        games = sorted({k[2] for k in v2_keys})

    x = np.arange(len(games))
    width = 0.25
    for i, strat in enumerate(STRAT_ORDER):
        means = []
        errs = []
        for game in games:
            scores = []
            for k, v in cells.items():
                if k[1] == strat and k[2] == game and "eval_v2_" in k[0]:
                    scores.extend(v)
            if not scores:
                means.append(0)
                errs.append(0)
            else:
                means.append(np.mean(scores))
                errs.append(np.std(scores))
        ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=4,
               color=STRAT_COLORS[strat], label=STRAT_LABELS[strat])

    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.set_ylabel("Mean score (12 episodes per cell)")
    ax.set_title("Latent vs Text bridge — F / T / L head-to-head (v2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def fig_main_box(cells: dict, out: str):
    """Score distribution box plot, v2 only, per game."""
    fig, ax = plt.subplots(figsize=(9, 5))
    v2_keys = [k for k in cells if "eval_v2_" in k[0]]
    games = sorted({k[2] for k in v2_keys})

    positions = []
    box_data = []
    box_colors = []
    tick_labels = []
    pos = 1
    for game in games:
        for strat in STRAT_ORDER:
            scores = []
            for k, v in cells.items():
                if k[1] == strat and k[2] == game and "eval_v2_" in k[0]:
                    scores.extend(v)
            if scores:
                box_data.append(scores)
                box_colors.append(STRAT_COLORS[strat])
                tick_labels.append(f"{game}\n{strat}")
                positions.append(pos)
                pos += 1
        pos += 1  # gap between games

    bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                    showmeans=True, meanline=True, widths=0.7)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_ylabel("Episode score")
    ax.set_title("Score distribution — v2 eval (12 episodes per cell)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def fig_v1_vs_v2(cells: dict, out: str):
    """Compare L scores across v1 variants and v2 (MsPacman only)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    variants = [
        ("eval_full.json",        "L (v1 cross-attn)", "#888"),
        ("eval_gated_only.json",  "L (v1 gated only)", "#aaa"),
        ("eval_v2_mspacman.json", "L (v2 LLaVA-style)", "#e74c3c"),
    ]
    means, stds, labels, colors = [], [], [], []
    for fname, label, color in variants:
        scores = []
        for k, v in cells.items():
            if k[0] == fname and k[1] == "L" and k[2] == "MsPacman":
                scores.extend(v)
        if scores:
            means.append(np.mean(scores))
            stds.append(np.std(scores))
            labels.append(label)
            colors.append(color)
    # Reference lines for F and T (v2 eval)
    f_scores = [v for k, v in cells.items()
                if k[0] == "eval_v2_mspacman.json" and k[1] == "F"]
    t_scores = [v for k, v in cells.items()
                if k[0] == "eval_v2_mspacman.json" and k[1] == "T"]
    f_mean = np.mean([s for sl in f_scores for s in sl]) if f_scores else None
    t_mean = np.mean([s for sl in t_scores for s in sl]) if t_scores else None

    x = np.arange(len(means))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean MsPacman score (12 episodes)")
    ax.set_title("Latent-bridge variants — v1 (cross-attn) → v2 (LLaVA-style)")
    if f_mean is not None:
        ax.axhline(f_mean, color="#7f8fa6", linestyle="--", label=f"F baseline ({f_mean:.0f})")
    if t_mean is not None:
        ax.axhline(t_mean, color="#3498db", linestyle="--", label=f"T baseline ({t_mean:.0f})")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def fig_latency(cells_all: dict, eval_paths: list[str], out: str):
    """Per-strategy mean latency, from v2 eval files."""
    fig, ax = plt.subplots(figsize=(7, 4))
    v2_paths = [p for p in eval_paths if "eval_v2_" in os.path.basename(p)]
    latencies = {s: [] for s in STRAT_ORDER}
    for p in v2_paths:
        try:
            data = json.load(open(p))
        except Exception:
            continue
        for c in data.get("cells", []):
            s = c.get("strategy")
            if s in latencies:
                latencies[s].append(c.get("mean_action_latency_ms", 0))

    means = [np.mean(latencies[s]) if latencies[s] else 0 for s in STRAT_ORDER]
    colors = [STRAT_COLORS[s] for s in STRAT_ORDER]
    ax.bar(STRAT_ORDER, means, color=colors)
    for i, m in enumerate(means):
        ax.text(i, m + 2, f"{m:.0f}ms", ha="center", fontsize=9)
    ax.axhline(67, color="r", linestyle="--", label="15 Hz target (67 ms)")
    ax.set_ylabel("Mean per-tick latency (ms)")
    ax.set_title("Per-tick latency by strategy (v2 eval)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-files", nargs="+", default=None,
                    help="Glob patterns or paths. Default: results/eval_*.json")
    ap.add_argument("--out-dir", default="figs")
    args = ap.parse_args()

    if args.eval_files:
        paths = []
        for pat in args.eval_files:
            paths.extend(sorted(glob.glob(pat)) or [pat])
    else:
        paths = sorted(glob.glob("results/eval_*.json"))

    if not paths:
        print("No eval JSONs found")
        return

    print(f"Loading {len(paths)} eval files...")
    cells = _load_cells(paths)
    print(f"  groups: {len(cells)}")

    Path(args.out_dir).mkdir(exist_ok=True)
    fig_main_bar(cells, os.path.join(args.out_dir, "main_bar.png"))
    fig_main_box(cells, os.path.join(args.out_dir, "main_box.png"))
    fig_v1_vs_v2(cells, os.path.join(args.out_dir, "v1_vs_v2.png"))
    fig_latency(cells, paths, os.path.join(args.out_dir, "latency.png"))


if __name__ == "__main__":
    main()
