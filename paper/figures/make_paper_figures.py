"""Generate paper-quality figures for the Latent Bridge arXiv submission.

Produces (in paper/figures/):
  0. fig_system.pdf            — whole-system runtime loop + F/T/L + training pipeline
  1. fig_architecture.pdf      — v1 vs v2 bridge schematic
  2. fig_headline.pdf          — cross-game F/T/L bar chart
  3. fig_roadrunner.pdf        — RoadRunner F=0 vs L=608 close-up
  4. fig_stage_a_ood.pdf       — action-head OOD-brittleness diagnosis chart
  5. fig_bandwidth.pdf         — N=4/8/16 Goldilocks ablation
  6. fig_continuous_vs_categorical.pdf  — refined claim scatter
  7. fig_latency.pdf           — vrf trade-off

Run from project root:
    python3 paper/figures/make_paper_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Load post-hoc statistics computed by analysis.py
STATS_PATH = Path(__file__).parent / "stats.json"
STATS = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}


def _cell(game: str, variant: str):
    """Return summary dict (mean, std, ci_lo, ci_hi, ...) per strategy, or {}."""
    return STATS.get("cells", {}).get(f"{game}/{variant}", {}).get("summary", {})


def _ci_yerr(d: dict, strat: str):
    """Convert mean + ci into asymmetric yerr for errorbar plots."""
    if strat not in d:
        return 0.0, 0.0, 0.0
    s = d[strat]
    m = s["mean"]
    return m, max(m - s["ci_lo"], 0.0), max(s["ci_hi"] - m, 0.0)

# NeurIPS-friendly styling: serif fonts, restrained palette, clean grid.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed real fonts (not Type 3)
    "ps.fonttype": 42,
})

# Colorblind-safe palette
C_F = "#7f8c8d"   # neutral grey — fast
C_T = "#3498db"   # blue — text bridge
C_L = "#e74c3c"   # red — latent bridge
C_S = "#95a5a6"   # light grey — slow only
C_O = "#9b59b6"   # purple — oracle
C_GOOD = "#27ae60"
C_BAD  = "#c0392b"
C_ACC  = "#1f6feb"

OUT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Figure 0 — System architecture: async fast/slow runtime loop, F/T/L channels,
#            and the training pipeline.
# ---------------------------------------------------------------------------

def fig_system():
    # Single-panel runtime architecture, sized close to \textwidth so on-page
    # downscaling is mild and the labels stay legible. The F/T/L strategies and
    # the training pipeline are described in the body text, not here.
    fig = plt.figure(figsize=(8.6, 5.1))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.4)

    def box(a, x, y, w, h, fc, ec, text, fs=8.5, tc="black", lw=1.0,
            weight="normal", style="round,pad=0.06", va="center"):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                   facecolor=fc, edgecolor=ec, linewidth=lw))
        ty = y + h / 2 if va == "center" else y + h - 0.18
        a.text(x + w / 2, ty, text, ha="center", va=va,
               fontsize=fs, color=tc, fontweight=weight)

    def arrow(a, x0, y0, x1, y1, color="grey", lw=1.4, ls="-", style="-|>",
              rad=0.0):
        cs = f"arc3,rad={rad}" if rad else "arc3,rad=0"
        a.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                   linestyle=ls, connectionstyle=cs))

    # ============== FAST LOOP (top) ==============
    # Fast-loop container, with a clear header band so text never overlaps content.
    ax.add_patch(FancyBboxPatch((3.0, 4.5), 12.8, 5.25, boxstyle="round,pad=0.04",
                                facecolor="#eef6ff", edgecolor=C_ACC, linewidth=1.3))
    ax.text(9.4, 9.42,
            "FAST reactive loop  —  MiniCPM-o 4.5 (9 B, frozen)  —  $\\sim$15 Hz",
            fontsize=9.5, color=C_ACC, ha="center", va="center", fontweight="bold")

    # Environment (far left, spans both halves)
    box(ax, 0.15, 5.35, 2.45, 2.1, "#f5f5f5", "0.4",
        "Environment\n\nAtari @ 15 Hz\nMetaDrive @ 10 Hz", fs=8.5)

    # vision tower
    box(ax, 3.35, 6.95, 2.2, 1.1, "white", C_ACC, "vision\ntower\n(frozen)", fs=8.2)
    arrow(ax, 2.6, 6.9, 3.3, 7.35, color="0.45", lw=1.2)      # env frame -> vision tower
    ax.text(2.98, 7.5, "frame", fontsize=7.8, color="0.35", ha="center", va="bottom")

    # ---- input token strip: [L prefix][vision][state prompt][T suffix] ----
    sy, sh = 5.45, 1.0
    lx = 3.35
    ax.text(8.0, sy + sh + 0.30, "fast-model input sequence", fontsize=8.2,
            color="0.4", ha="center", va="bottom", style="italic")
    # L prefix: 8 thin red bars
    for i in range(8):
        ax.add_patch(FancyBboxPatch((lx + i * 0.18, sy), 0.16, sh,
                                    boxstyle="round,pad=0", facecolor=C_L,
                                    edgecolor="white", linewidth=0.4, alpha=0.92))
    ax.text(lx + 0.72, sy - 0.20, "8 latent\ntokens", fontsize=7.8,
            color=C_L, ha="center", va="top", fontweight="bold")
    box(ax, 5.05, sy, 1.95, sh, "#e4efe4", "0.5", "vision\ntokens", fs=8.0,
        style="round,pad=0.02")
    box(ax, 7.1, sy, 2.0, sh, "#f4f4f4", "0.5", "game-state\nprompt", fs=8.0,
        style="round,pad=0.02")
    box(ax, 9.2, sy, 2.15, sh, "#dbeafe", C_T, "text\nsuffix", fs=8.0,
        tc="#1d5e8a", style="round,pad=0.02")
    arrow(ax, 6.05, 6.9, 6.02, sy + sh + 0.02, color="0.5", lw=1.1)  # vision tower -> strip

    # LLM + action head
    box(ax, 11.95, 6.85, 1.7, 1.2, "white", C_ACC, "36-layer\nLLM\n(frozen)", fs=8.0)
    box(ax, 13.9, 6.85, 1.75, 1.2, "white", C_ACC, "action head\n(trained)",
        fs=8.0)
    arrow(ax, 11.4, sy + sh / 2, 12.0, 7.0, color="0.3", lw=1.7)   # strip -> LLM
    arrow(ax, 13.6, 7.45, 13.85, 7.45, color="0.3", lw=1.7)        # LLM -> head
    # action back to env: right-angle route along the top band of the fast box
    ax.plot([14.78, 14.78], [8.05, 8.62], color="0.3", lw=1.4)
    ax.plot([14.78, 1.38], [8.62, 8.62], color="0.3", lw=1.4)
    arrow(ax, 1.38, 8.62, 1.38, 7.45, color="0.3", lw=1.4)
    ax.text(8.0, 8.74, "action (greedy argmax), every tick",
            fontsize=8.0, color="0.3", ha="center", va="bottom")

    # ============== async divider ==============
    ax.plot([0.15, 15.8], [4.2, 4.2], ls=(0, (4, 3)), color="0.6", lw=1.0)
    ax.text(15.78, 4.34, "synchronous  $\\sim$15 Hz", fontsize=7.8, color="0.4",
            ha="right", va="bottom", style="italic")
    ax.text(15.78, 4.06, "asynchronous  $\\sim$1 Hz", fontsize=7.8, color="0.4",
            ha="right", va="top", style="italic")

    # ============== SLOW MODEL (bottom) ==============
    box(ax, 0.15, 2.0, 2.45, 1.55, "#f5f5f5", "0.4",
        "structured state\n(RAM objects /\ndriving state)", fs=7.9)
    arrow(ax, 1.38, 5.35, 1.38, 3.6, color="0.5", lw=1.2)   # env -> structured state

    box(ax, 3.15, 1.9, 3.15, 1.75, "#fff0e0", C_ACC,
        "SLOW\nQwen3-VL-8B-Thinking\n(frozen, $\\sim$1.5 s/emission)", fs=8.0)
    arrow(ax, 2.6, 2.75, 3.1, 2.78, color="0.5", lw=1.2)

    # text emission -> T channel
    box(ax, 7.15, 2.75, 2.4, 0.95, "#dbeafe", C_T,
        "text emission\n($\\sim$300 chars)", fs=7.9, tc="#1d5e8a")
    arrow(ax, 6.3, 3.0, 7.1, 3.2, color=C_T, lw=1.5)
    arrow(ax, 10.2, 3.22, 10.2, sy - 0.06, color=C_T, lw=1.9)   # up to T box in strip

    # residuals -> MLP -> L channel
    box(ax, 7.15, 1.55, 2.4, 0.9, "#f4f4f4", "0.5",
        "layer-24 residuals\n(last 8 positions)", fs=7.9)
    arrow(ax, 6.3, 2.45, 7.1, 2.0, color="0.5", lw=1.2)
    box(ax, 10.3, 1.55, 3.05, 0.9, "#fff8c0", C_ACC,
        "bridge MLP\n(33 M params,\nonly trained part)", fs=7.9)
    arrow(ax, 9.6, 2.0, 10.25, 2.0, color="0.5", lw=1.2)
    # MLP up to L prefix in the strip
    arrow(ax, 11.3, 2.45, lx + 0.72, sy - 0.06, color=C_L, lw=2.0, rad=0.16)

    fig.savefig(OUT / "fig_system.pdf")
    fig.savefig(OUT / "fig_system.png", dpi=200)
    plt.close(fig)
    print("wrote fig_system.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 1 — Architecture (v1 cross-attn vs v2 LLaVA-style)
# ---------------------------------------------------------------------------

def fig_architecture():
    # Two stacked panels. A dedicated diagram band (y 1.7-4.4) sits above a clear
    # caption band (y~0.7) so the italic captions never underlap the boxes.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.5))

    def slow_box(ax):
        ax.add_patch(FancyBboxPatch((0.2, 2.25), 2.0, 1.4, boxstyle="round,pad=0.05",
                                    facecolor="#fff0e0", edgecolor=C_ACC, linewidth=0.8))
        ax.text(1.2, 2.95, "Slow\n(Qwen3-VL-8B)", ha="center", va="center", fontsize=8)

    def fast_box(ax):
        ax.add_patch(FancyBboxPatch((7.0, 1.7), 4.6, 2.7, boxstyle="round,pad=0.05",
                                    facecolor="#e0f0ff", edgecolor=C_ACC, linewidth=0.8))
        ax.text(9.3, 4.08, "Fast (MiniCPM-o)  —  36 LLM layers",
                ha="center", va="center", fontsize=8.5)

    def draw_v1(ax):
        ax.set_xlim(0, 12); ax.set_ylim(0.45, 4.7)
        ax.axis("off")
        ax.set_title("v1 — cross-attention into a 256-d ring buffer (2 of 36 layers)  —  "
                     "failed at deployment", fontsize=9.5, color=C_BAD, pad=6, loc="left")
        slow_box(ax)
        # Ring buffer
        ax.add_patch(FancyBboxPatch((3.3, 2.25), 2.5, 1.4, boxstyle="round,pad=0.05",
                                    facecolor="#f0e0d0", edgecolor="grey", linewidth=0.8))
        ax.text(4.55, 2.95, "256-d\nring buffer", ha="center", va="center", fontsize=8)
        fast_box(ax)
        # Cross-attn at 12 & 24 inside the fast box
        for yb, lab in [(3.0, "cross-attn @ layer 24"), (2.0, "cross-attn @ layer 12")]:
            ax.add_patch(FancyBboxPatch((7.2, yb), 4.2, 0.5, boxstyle="round,pad=0.02",
                                        facecolor=C_BAD, edgecolor="none", alpha=0.7))
            ax.text(9.3, yb + 0.25, lab, ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
        # Arrows
        ax.annotate("", xy=(3.3, 2.95), xytext=(2.2, 2.95),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.annotate("", xy=(7.2, 3.25), xytext=(5.85, 2.95),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.annotate("", xy=(7.2, 2.25), xytext=(5.85, 2.85),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.text(0.2, 0.7, "Only 2 of 36 layers see the bridge; no LLM inductive bias for 256-d vectors.",
                fontsize=8.5, style="italic", color=C_BAD)

    def draw_v2(ax):
        ax.set_xlim(0, 12); ax.set_ylim(0.45, 4.7)
        ax.axis("off")
        ax.set_title("v2 — LLaVA-style prepend into the 4096-d input space (all 36 layers)  —  "
                     "works", fontsize=9.5, color=C_GOOD, pad=6, loc="left")
        slow_box(ax)
        # Projection
        ax.add_patch(FancyBboxPatch((3.0, 2.25), 1.8, 1.4, boxstyle="round,pad=0.05",
                                    facecolor="#fff8c0", edgecolor=C_ACC, linewidth=0.8))
        ax.text(3.9, 2.95, "MLP\n4096→4096\n33M params", ha="center", va="center", fontsize=7.5)
        # 8 bridge tokens
        for i in range(8):
            ax.add_patch(FancyBboxPatch((5.2 + i*0.16, 2.45), 0.14, 1.0, boxstyle="round,pad=0",
                                        facecolor=C_L, edgecolor="white", linewidth=0.4, alpha=0.85))
        ax.text(5.85, 3.72, "8 tokens · 4096-d", ha="center", fontsize=7.8, color=C_ACC)
        fast_box(ax)
        # Highlight that ALL layers see bridge
        for y in np.linspace(1.95, 3.7, 12):
            ax.add_patch(FancyBboxPatch((7.2, y), 4.2, 0.13, boxstyle="round,pad=0",
                                        facecolor=C_GOOD, edgecolor="none", alpha=0.20))
        ax.text(9.3, 2.85, "all 36 layers attend\nvia standard causal attention",
                ha="center", va="center", fontsize=8, color=C_GOOD, fontweight="bold")
        # Arrows
        ax.annotate("", xy=(3.0, 2.95), xytext=(2.2, 2.95),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.annotate("", xy=(5.15, 2.95), xytext=(4.8, 2.95),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.annotate("", xy=(7.0, 2.95), xytext=(6.55, 2.95),
                    arrowprops=dict(arrowstyle="-|>", color=C_GOOD, lw=2))
        ax.text(0.2, 0.7, "Bridge tokens live in the LLM's input embedding space; full causal attention from all layers.",
                fontsize=8.5, style="italic", color=C_GOOD)

    draw_v1(ax1)
    draw_v2(ax2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, hspace=0.42)
    fig.savefig(OUT / "fig_architecture.pdf")
    fig.savefig(OUT / "fig_architecture.png", dpi=200)
    plt.close(fig)
    print("wrote fig_architecture.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 2 — Cross-game headline bar chart
# ---------------------------------------------------------------------------

# Headline: one bar per game, using each game's "reported" variant.
# Reported variant policy: bare SA where it gives evaluable T and L (i.e. neither
# collapses to zero); otherwise robust SA. This makes the headline comparison the
# fairest single configuration per game.
HEADLINE_ORDER = [
    # (label, game, variant)
    ("MsPacman",           "MsPacman",      "bare"),
    ("Seaquest",           "Seaquest",      "bare"),
    ("RoadRunner",         "RoadRunner",    "bare"),
    ("River\nRaid*",       "RiverRaid",     "robust"),
    ("Enduro*",            "Enduro",        "robust"),
    ("Q*bert*",            "Qbert",         "robust"),
    ("Space\nInvaders*",   "SpaceInvaders", "robust"),
    ("Pong",               "Pong",          "bare"),
]


# Best-achievable (held-out, leave-one-seed-out) per-channel-decoder cells, reproduced
# from scripts/decoder_select_cv.py (mean, std; n=12). This is the canonical headline:
# each channel at its own best decoder, the fair comparison. Greedy is the cautionary
# contrast that lives only in the decoder-sensitivity section.
BEST_ACHIEVABLE = [
    # label, robust?, F(mean,std), T(mean,std), L(mean,std), L-vs-T sig, delta% (if L-win)
    ("MsPacman",        False, (273, 89),  (401, 115), (628, 341), "**",  +57),
    ("RoadRunner",      False, (0,   0),   (475, 153), (608, 28),  "***", +28),
    ("Seaquest",        False, (57,  33),  (143, 35),  (125, 18),  "ns",  None),
    ("River\nRaid",     True,  (994, 146), (639, 238), (566, 223), "ns",  None),
    ("Q*bert",          True,  (65,  61),  (185, 167), (146, 95),  "ns",  None),
    ("Space\nInvaders", True,  (135, 62),  (163, 111), (142, 83),  "ns",  None),
    ("Enduro",          True,  (4,   3),   (3,   5),   (2,   1),   "ns",  None),
]


def fig_headline():
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    labels = [r[0] + ("*" if r[1] else "") for r in BEST_ACHIEVABLE]
    means = {"F": [r[2][0] for r in BEST_ACHIEVABLE],
             "T": [r[3][0] for r in BEST_ACHIEVABLE],
             "L": [r[4][0] for r in BEST_ACHIEVABLE]}
    stds  = {"F": [r[2][1] for r in BEST_ACHIEVABLE],
             "T": [r[3][1] for r in BEST_ACHIEVABLE],
             "L": [r[4][1] for r in BEST_ACHIEVABLE]}

    x = np.arange(len(BEST_ACHIEVABLE))
    w = 0.26
    names = {"F": "F (Fast-Only)", "T": "T (Text Bridge)", "L": "L (Latent Bridge)"}
    for off, s, c in [(-w, "F", C_F), (0, "T", C_T), (w, "L", C_L)]:
        ax.bar(x + off, means[s], w, yerr=stds[s], color=c, label=names[s],
               capsize=2, error_kw=dict(elinewidth=0.8))

    # Annotate the two significant Latent wins; mark the rest as ties.
    for i, r in enumerate(BEST_ACHIEVABLE):
        sig, delta = r[5], r[6]
        t, l = means["T"][i], means["L"][i]
        top = max(l + stds["L"][i], t + stds["T"][i])
        if sig != "ns" and delta is not None:
            ax.text(i + w/2, top + 30, f"$L$ +{delta}%\n{sig}", ha="center", va="bottom",
                    fontsize=7.5, color=C_GOOD, fontweight="bold")
        else:
            ax.text(i, top + 30, "tie", ha="center", va="bottom",
                    fontsize=7, color="0.5", style="italic")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Best-achievable score  (mean $\\pm$ std, $n=12$)")
    ax.set_ylim(top=1320)
    ax.set_title("Cross-game scores with each channel at its own best decoder (held-out)\n"
                 "The Latent Bridge significantly beats the Text Bridge on 2 of 7 games; "
                 "ties elsewhere   ($*$ = robust action head)",
                 fontsize=9.4)
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=7.8,
              handletextpad=0.5, borderaxespad=0.4)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_headline.pdf")
    fig.savefig(OUT / "fig_headline.png", dpi=200)
    plt.close(fig)
    print("wrote fig_headline.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 3 — RoadRunner: F=0 → L=608 (reproducible value)
# ---------------------------------------------------------------------------

def fig_roadrunner():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    conditions = ["S\n(slow only)", "F\n(fast only)", "T\n(text bridge)", "L\n(latent bridge)"]
    scores = [None, 0, 475, 608]  # no S baseline for RoadRunner (reproducible re-run)
    errs   = [None, 0, 160, 29]
    colors = [C_S, C_F, C_T, C_L]
    x = np.arange(len(conditions))
    valid = [s is not None for s in scores]

    ax.bar([xi for xi, v in zip(x, valid) if v],
           [s for s in scores if s is not None],
           yerr=[e for s, e in zip(scores, errs) if s is not None],
           color=[c for c, v in zip(colors, valid) if v],
           capsize=3, error_kw=dict(elinewidth=1))
    for xi, s in zip(x, scores):
        if s is None:
            ax.text(xi, 50, "n/a", ha="center", color="grey", fontsize=8)
        else:
            ax.text(xi, s + (errs[x.tolist().index(xi)] if s > 0 else 0) + 30,
                    str(s), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("RoadRunner score (12 episodes)")
    ax.set_title("RoadRunner: F cannot score → L unlocks scoring policy (+28% L over T)",
                 fontsize=10)
    ax.set_ylim(-50, 1200)
    # callout
    ax.annotate("F has the reflex machinery\nbut not the directional bias.\nThe slow's compressed\nstrategic context unlocks it.",
                xy=(3, 608), xytext=(0.5, 800),
                fontsize=7.5, style="italic", color=C_ACC,
                arrowprops=dict(arrowstyle="->", color=C_ACC, lw=0.8))
    fig.tight_layout()
    fig.savefig(OUT / "fig_roadrunner.pdf")
    fig.savefig(OUT / "fig_roadrunner.png", dpi=200)
    plt.close(fig)
    print("wrote fig_roadrunner.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 4 — Action-head OOD-brittleness diagnosis (SI + RR before/after)
# ---------------------------------------------------------------------------

def fig_stage_a_ood():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    for ax, game, bare, robust, title in [
        (axes[0], "SpaceInvaders",
         {"F": (105, 0), "T": (0, 0), "L": (0, 0)},
         {"F": (107, 60), "T": (18, 18), "L": (15, 0)},
         "SpaceInvaders"),
        (axes[1], "RiverRaid",
         {"F": (1067, 84), "T": (383, 57), "L": (360, 0)},
         {"F": (1033, 19), "T": (337, 77), "L": (612, 297)},
         "River Raid"),
    ]:
        x = np.arange(3)
        w = 0.4
        bare_vals = [bare["F"][0], bare["T"][0], bare["L"][0]]
        bare_errs = [bare["F"][1], bare["T"][1], bare["L"][1]]
        robust_vals = [robust["F"][0], robust["T"][0], robust["L"][0]]
        robust_errs = [robust["F"][1], robust["T"][1], robust["L"][1]]
        b1 = ax.bar(x - w/2, bare_vals, w, yerr=bare_errs, capsize=2,
                    color=[C_F, C_T, C_L], alpha=0.45, edgecolor="black",
                    linewidth=0.5, label="bare action head", hatch="//")
        b2 = ax.bar(x + w/2, robust_vals, w, yerr=robust_errs, capsize=2,
                    color=[C_F, C_T, C_L], edgecolor="black", linewidth=0.5,
                    label="robust action head (suffix-prob=0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(["F", "T", "L"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Score")
        if game == "SpaceInvaders":
            ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("Action-head OOD-brittleness: one fix recovers two distinct games",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage_a_ood.pdf")
    fig.savefig(OUT / "fig_stage_a_ood.png", dpi=200)
    plt.close(fig)
    print("wrote fig_stage_a_ood.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 5 — Bandwidth Goldilocks (N=4/8/16)
# ---------------------------------------------------------------------------

def fig_bandwidth():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    Ns = [4, 8, 16]
    L_means = [296, 628, 259]
    L_errs  = [63, 341, 71]
    ax.errorbar(Ns, L_means, yerr=L_errs, fmt="o-", color=C_L, capsize=4,
                lw=1.5, markersize=8, label="L score (MsPacman, n=12)")
    ax.axhline(256, color=C_F, ls="--", lw=1, label="F baseline (256)")
    ax.axhline(408, color=C_T, ls="--", lw=1, label="T baseline (408)")
    ax.set_xscale("log", base=2)
    # Deploy-only series: train at N=8, deploy at N
    L_deploy_means = [232, 628, 720]
    L_deploy_errs  = [52,  341, 123]
    ax.errorbar(Ns, L_deploy_means, yerr=L_deploy_errs, fmt="s--", color=C_ACC,
                capsize=4, lw=1.2, markersize=7, alpha=0.85,
                label="L score (deploy-only: train=8, deploy=N)")
    ax.set_xticks(Ns)
    ax.set_xticklabels(Ns)
    ax.set_xlabel("Latent token count N (per emission)")
    ax.set_ylabel("L score (mean ± std)")
    ax.set_title("Matched-N suggests Goldilocks at N=8;\n"
                 "deploy-only series inverts it (best at N=16). 3 points fit either story.",
                 fontsize=9.5)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bandwidth.pdf")
    fig.savefig(OUT / "fig_bandwidth.png", dpi=200)
    plt.close(fig)
    print("wrote fig_bandwidth.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 6 — Refined claim (continuous-content vs categorical-content)
# ---------------------------------------------------------------------------

def fig_continuous_vs_categorical():
    """Quantitative x-axis: lexical diversity (unique whitespace-split tokens
    per emission) of slow-model text on each game. Higher = the slow tends to
    say new things each emission (continuous-rich state to describe).
    Lower = the slow re-uses the same words (categorical, repetitive).
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    # reported_variant — the row used in headline figure
    REPORTED = {
        "MsPacman": "bare",
        "Seaquest": "bare",
        "RoadRunner": "bare",
        "Riverraid": "robust",
        "Enduro": "robust",
        "Qbert": "robust",
        "SpaceInvaders": "robust",
    }
    LABEL = {
        "MsPacman": "MsPacman", "Seaquest": "Seaquest", "RoadRunner": "RoadRunner",
        "Riverraid": "River Raid*", "Enduro": "Enduro*", "Qbert": "Q*bert*",
        "SpaceInvaders": "SpaceInvaders*",
    }
    rows = []
    for game, variant in REPORTED.items():
        cells_key = f"{game.replace('Riverraid','RiverRaid')}/{variant}"
        # raw cells use canonical RiverRaid; emission key uses 'Riverraid'
        cell_key_alt = cells_key
        cells = STATS.get("cells", {}).get(cell_key_alt, {}).get("summary", {})
        emm = STATS.get("emissions", {}).get(game, {})
        if not cells or not emm:
            continue
        T = cells.get("T", {}).get("mean", float("nan"))
        L = cells.get("L", {}).get("mean", float("nan"))
        if T <= 0 and L <= 0:
            continue
        ratio = (L - T) / T if T > 0 else (1.0 if L > 0 else 0.0)
        rows.append((LABEL[game], emm["unique_per_emission"], ratio,
                     emm["gzip_ratio"]))

    if not rows:
        plt.close(fig); print("fig_continuous_vs_categorical: no data"); return

    names = [r[0] for r in rows]
    xs    = [r[1] for r in rows]
    ys    = [r[2] for r in rows]

    # Per-game label offsets (points) to avoid collisions in the crowded
    # mid-diversity band (Seaquest / RoadRunner / SpaceInvaders).
    OFFS = {
        "MsPacman":       (9, 5),
        "Seaquest":       (-10, 12),
        "RoadRunner":     (9, 5),
        "River Raid*":    (10, -13),
        "Enduro*":        (9, 5),
        "Q*bert*":        (9, -14),
        "SpaceInvaders*": (-12, -16),
    }
    for name, x, y in zip(names, xs, ys):
        color = C_L if y > 0 else C_BAD
        ax.scatter([x], [y], s=110, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        dx, dy = OFFS.get(name, (8, 4 if y > 0 else -10))
        ha = "right" if dx < 0 else "left"
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha=ha, zorder=4)

    # Trend hint
    if len(xs) >= 3:
        slope, intercept = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs)-0.5, max(xs)+0.5, 50)
        ax.plot(xline, slope * xline + intercept, "--", color="grey",
                lw=0.8, alpha=0.7,
                label=f"linear fit: slope={slope:+.2f}/diversity-unit")

    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlim(min(xs) - 1.4, max(xs) + 1.8)
    ax.set_ylim(min(ys) - 0.18, max(ys) + 0.20)
    ax.set_xlabel("Lexical diversity of slow emissions  "
                  "(unique whitespace tokens / emission)")
    ax.set_ylabel("$(L - T)\\,/\\,T$")
    # Compute Pearson r for honest annotation
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 2 else float("nan")
    ax.set_title(f"Emission lexical diversity does not predict sign($L-T$): "
                 f"Pearson r = {r:+.2f} (n={len(xs)}).\n"
                 "Q*bert inverts, but Enduro and River Raid (lower diversity) do not.",
                 fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_continuous_vs_categorical.pdf")
    fig.savefig(OUT / "fig_continuous_vs_categorical.png", dpi=200)
    plt.close(fig)
    print("wrote fig_continuous_vs_categorical.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 7 — Latency × vrf trade-off
# ---------------------------------------------------------------------------

def fig_latency():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    # n=12 values (the more rigorous sweep)
    vrf = [1, 4, 15]
    lat = [157, 101, 77]
    score = [183, 110, 140]
    color_s = [C_L if s == max(score) else (C_T if s == min(score) else C_F) for s in score]
    ax2 = ax.twinx()
    l1, = ax.plot(vrf, lat, "o-", color=C_ACC, lw=1.5, markersize=8, label="latency (ms)")
    l2, = ax2.plot(vrf, score, "s--", color=C_L, lw=1.5, markersize=8, label="score (MsPacman F)")
    ax.set_xlabel("Vision refresh every N ticks")
    ax.set_ylabel("Latency (ms)", color=C_ACC)
    ax2.set_ylabel("Score", color=C_L)
    ax.tick_params(axis="y", labelcolor=C_ACC)
    ax2.tick_params(axis="y", labelcolor=C_L)
    ax.set_xticks(vrf)
    ax.set_title("Vision-token cache: −51% latency at vrf=15;\nscore non-monotonic (perception staleness × policy interaction)",
                 fontsize=9.5)
    ax.legend(handles=[l1, l2], loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.pdf")
    fig.savefig(OUT / "fig_latency.png", dpi=200)
    plt.close(fig)
    print("wrote fig_latency.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 8 — The L>T predictor: latent helps iff slow guidance helps (T>F),
#            with MetaDrive (driving) as the controlled negative.
# ---------------------------------------------------------------------------
def fig_predictor():
    data_path = OUT / "predictor_data.json"
    if not data_path.exists():
        print("fig_predictor: no predictor_data.json, skipping"); return
    d = json.loads(data_path.read_text())
    rows = d["canonical"]["rows"]; r = d["canonical"]["pearson"]; n = d["canonical"]["n"]
    all_rows = d["all_cells"]["rows"]; r_all = d["all_cells"]["pearson"]; n_all = d["all_cells"]["n"]

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [2.05, 1.0]})

    # ---- left: scatter L-F vs T-F ----
    xs = np.array([row["TmF"] for row in rows], float)
    ys = np.array([row["LmF"] for row in rows], float)
    ax_all = np.array([row["TmF"] for row in all_rows], float)
    ay_all = np.array([row["LmF"] for row in all_rows], float)
    # symmetric log-ish scaling: use signed-sqrt so the huge RoadRunner point
    # doesn't crush the cluster near zero, while keeping sign + ordering honest.
    def sst(v):
        return np.sign(v) * np.sqrt(np.abs(v))

    lim = max(np.abs(sst(ax_all)).max(), np.abs(sst(ay_all)).max()) * 1.18
    # quadrant shading: upper-right = bridge helps (T>F and L>F)
    ax.axhspan(0, lim, xmin=0.5, xmax=1.0, color=C_GOOD, alpha=0.06, zorder=0)
    ax.axhline(0, color="0.5", lw=0.8, zorder=1)
    ax.axvline(0, color="0.5", lw=0.8, zorder=1)
    # y=x reference (L tracks T)
    ax.plot([-lim, lim], [-lim, lim], ls=":", color="0.55", lw=1.0,
            zorder=1, label="$L\\!-\\!F = T\\!-\\!F$")
    # faint: ALL 16 (game,variant) cells — shows the correlation is not an artifact
    # of variant selection.
    ax.scatter(sst(ax_all), sst(ay_all), s=22, color="0.6", alpha=0.45,
               edgecolor="none", zorder=2,
               label=f"all {n_all} cells ($r={r_all:.2f}$)")

    for row in rows:
        x, y = sst(row["TmF"]), sst(row["LmF"])
        is_md = row["game"] == "MetaDrive"
        col = C_L if (row["LmF"] > 0 and row["TmF"] > 0) else (
              C_BAD if row["LmF"] < -1 else C_F)
        if is_md:
            ax.scatter([x], [y], s=120, marker="D", facecolor="white",
                       edgecolor=C_ACC, linewidth=1.8, zorder=5)
        else:
            ax.scatter([x], [y], s=70, color=col, edgecolor="white",
                       linewidth=0.8, zorder=4)
        # label placement — explicit per-game nudges to declutter the near-origin
        # cluster (Seaquest / Q*bert / Enduro / MetaDrive).
        u = lim
        NUDGE = {
            "RoadRunner":    (-0.06 * u,  0.05 * u, "right"),
            "MsPacman":      (-0.05 * u, -0.10 * u, "right"),
            "Seaquest":      (-0.05 * u,  0.10 * u, "right"),
            "Qbert":         ( 0.06 * u, -0.04 * u, "left"),
            "Enduro":        ( 0.06 * u,  0.05 * u, "left"),
            "SpaceInvaders": ( 0.06 * u, -0.04 * u, "left"),
            "Riverraid":     ( 0.06 * u, -0.10 * u, "left"),
            "MetaDrive":     ( 0.07 * u, -0.02 * u, "left"),
        }
        dx, dy, ha = NUDGE.get(row["game"], (0.06 * u, 0.06 * u, "left"))
        display = {"Riverraid": "River Raid", "Qbert": "Q*bert"}
        lab = display.get(row["game"], row["game"]) + ("  (driving)" if is_md else "")
        ax.annotate(lab, (x, y), (x + dx, y + dy), fontsize=7.0,
                    ha=ha, color=(C_ACC if is_md else "0.2"),
                    fontweight=("bold" if is_md else "normal"), zorder=6)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    # honest tick labels: ticks live in signed-sqrt space, but show TRUE reward deltas
    true_ticks = [-700, -100, -10, 0, 10, 100, 700]
    tk = [sst(v) for v in true_ticks]
    lab = [("0" if v == 0 else f"{v:+d}") for v in true_ticks]
    ax.set_xticks(tk); ax.set_xticklabels(lab, fontsize=7)
    ax.set_yticks(tk); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("Text bridge benefit  $T-F$   (signed-$\\sqrt{\\cdot}$ axis)")
    ax.set_ylabel("Latent bridge benefit  $L-F$")
    ax.set_title(f"Latent helps iff slow reasoning helps  (reported-variant $r={r:.2f}$, $n={n}$)")
    # annotate quadrant
    ax.text(0.97 * lim, 0.10 * lim, "bridge\nhelps", fontsize=7.5,
            color=C_GOOD, ha="right", va="bottom", style="italic")
    ax.text(-0.97 * lim, -0.10 * lim, "bridge\nhurts", fontsize=7.5,
            color=C_BAD, ha="left", va="top", style="italic")
    ax.legend(loc="upper left", frameon=False, fontsize=7.0)

    # ---- right: MetaDrive reactive vs planning, F/T/L ----
    # both MetaDrive regimes: slow never beats F regardless of planning demand.
    groups = ["Reactive\nlane-keep", "Planning\nintersections"]
    F_vals = [71.2, 87.8]; T_vals = [69.5, 85.1]; L_vals = [69.5, 85.1]
    x = np.arange(len(groups)); w = 0.26
    axb.bar(x - w, F_vals, w, color=C_F, label="F (fast)")
    axb.bar(x,     T_vals, w, color=C_T, label="T (text)")
    axb.bar(x + w, L_vals, w, color=C_L, label="L (latent)")
    axb.set_xticks(x); axb.set_xticklabels(groups, fontsize=7.2)
    axb.set_ylabel("Driving reward")
    axb.set_title("MetaDrive: slow adds\nnothing, even with planning", fontsize=8.6)
    axb.legend(loc="upper left", frameon=False, fontsize=6.8, ncol=1)
    axb.set_ylim(0, 118)

    fig.tight_layout()
    fig.savefig(OUT / "fig_predictor.pdf")
    fig.savefig(OUT / "fig_predictor.png", dpi=200)
    plt.close(fig)
    print("wrote fig_predictor.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 9 — Bridge-replacement decomposition (MsPacman): how much of L's lift
#            over F is architectural "slots" vs learned bridge content.
# ---------------------------------------------------------------------------
def fig_bridge_decomp():
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    labels = ["$F$\nno tokens", "$L_{\\rm zero}$\n8 zero", "$L_{\\rm random}$\n8 random",
              "$L_{\\rm trained}$\nbridge"]
    vals = [256, 379, 387, 628]
    errs = [25, 95, 161, 356]
    cols = [C_F, "#f3b8b0", "#ef9a90", C_L]
    x = np.arange(4)
    ax.bar(x, vals, 0.62, yerr=errs, color=cols, edgecolor="black", linewidth=0.5,
           capsize=3, error_kw=dict(elinewidth=0.8))
    for xi, v in zip(x, vals):
        ax.text(xi, v + (errs[xi] if xi != 0 else 0) + 18, str(v), ha="center",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MsPacman score ($n=12$)")
    ax.set_ylim(0, 1080)
    # decomposition note in the clear upper-left area (avoids the tall error bars)
    ax.text(0.03, 0.97,
            "$L$'s lift over $F$ splits into:\n"
            "  $\\approx$40% empty slots (zero $\\approx$ random)\n"
            "  $\\approx$60% learned bridge content",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fbeeec", ec=C_L, lw=0.9))
    ax.set_title("What the Latent Bridge carries: empty slots vs. learned content\n"
                 "(MsPacman: 8 empty/random tokens already lift $F$; trained content lifts it further)",
                 fontsize=9.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bridge_decomp.pdf")
    fig.savefig(OUT / "fig_bridge_decomp.png", dpi=200)
    plt.close(fig)
    print("wrote fig_bridge_decomp.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 10 — Longer text suffix hurts: T decays as older emissions are
#             concatenated, while L (latest emission only) is fixed.
# ---------------------------------------------------------------------------
def fig_longer_t():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    W = [1, 2, 3]
    panels = [
        (axes[0], "MsPacman", [408, 378, 352], 628, "gentle decline"),
        (axes[1], "RoadRunner", [608, 8, 0], 608, "collapse to 0"),
    ]
    for ax, game, Tvals, Lval, note in panels:
        ax.plot(W, Tvals, "o-", color=C_T, lw=1.8, markersize=8,
                label="Text Bridge ($T$)")
        ax.axhline(Lval, ls="--", color=C_L, lw=1.6,
                   label="Latent Bridge ($L$), fixed")
        for w, v in zip(W, Tvals):
            ax.annotate(str(v), (w, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.6, color=C_T, fontweight="bold")
        ax.set_title(f"{game}", fontsize=9.5)
        ax.set_xlabel("emissions concatenated into suffix ($w$)")
        ax.set_xticks(W)
        ax.set_ylim(-40, 760)
        ax.margins(x=0.18)
    axes[0].set_ylabel("score ($n=12$)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=7.5)
    fig.suptitle("Longer text suffix, worse policy: the Text Bridge decays as older "
                 "emissions pile up; the Latent Bridge is fixed", fontsize=9.4, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_longer_t.pdf")
    fig.savefig(OUT / "fig_longer_t.png", dpi=200)
    plt.close(fig)
    print("wrote fig_longer_t.{pdf,png}")


if __name__ == "__main__":
    fig_system()
    fig_architecture()
    fig_headline()
    fig_roadrunner()
    fig_stage_a_ood()
    fig_bandwidth()
    fig_continuous_vs_categorical()
    fig_latency()
    fig_predictor()
    fig_bridge_decomp()
    fig_longer_t()
    print("\nall figures written to:", OUT)
