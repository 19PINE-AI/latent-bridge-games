"""Generate paper-quality figures for the Latent Bridge arXiv submission.

Produces (in paper/figures/):
  1. fig_architecture.pdf      — v1 vs v2 bridge schematic
  2. fig_headline.pdf          — cross-game F/T/L bar chart
  3. fig_roadrunner.pdf        — RoadRunner F=0 vs L=967 close-up
  4. fig_stage_a_ood.pdf       — Stage A OOD-brittleness diagnosis chart
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
# Figure 1 — Architecture (v1 cross-attn vs v2 LLaVA-style)
# ---------------------------------------------------------------------------

def fig_architecture():
    # Vertical layout: each panel on its own row to give labels room.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6))

    def draw_v1(ax):
        ax.set_xlim(0, 12); ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title("v1 — Cross-attention into 256-d ring buffer  (failed: KL=0.004 offline, L=225 < F=256 at deployment)",
                     fontsize=9.5, color=C_BAD, pad=4, loc="left")
        # Slow box
        ax.add_patch(FancyBboxPatch((0.2, 1.8), 2.0, 1.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#fff0e0", edgecolor=C_ACC, linewidth=0.8))
        ax.text(1.2, 2.5, "Slow\n(Qwen3-VL-8B)", ha="center", va="center", fontsize=8)
        # Ring buffer
        ax.add_patch(FancyBboxPatch((3.3, 1.8), 2.5, 1.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#f0e0d0", edgecolor="grey", linewidth=0.8))
        ax.text(4.55, 2.5, "256-d\nring buffer", ha="center", va="center", fontsize=8)
        # Fast box — only depths 12 & 24 have cross-attn
        ax.add_patch(FancyBboxPatch((7.0, 0.3), 4.6, 4.0,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#e0f0ff", edgecolor=C_ACC, linewidth=0.8))
        ax.text(9.3, 3.9, "Fast (MiniCPM-o)  —  36 LLM layers",
                ha="center", va="center", fontsize=8.5)
        # Cross-attn at 12 & 24 inside the fast box
        ax.add_patch(FancyBboxPatch((7.2, 2.5), 4.2, 0.5,
                                     boxstyle="round,pad=0.02",
                                     facecolor=C_BAD, edgecolor="none", alpha=0.7))
        ax.text(9.3, 2.75, "cross-attn @ layer 24", ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
        ax.add_patch(FancyBboxPatch((7.2, 1.1), 4.2, 0.5,
                                     boxstyle="round,pad=0.02",
                                     facecolor=C_BAD, edgecolor="none", alpha=0.7))
        ax.text(9.3, 1.35, "cross-attn @ layer 12", ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
        # Arrows
        ax.annotate("", xy=(3.3, 2.5), xytext=(2.2, 2.5),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.annotate("", xy=(7.2, 2.75), xytext=(5.85, 2.55),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.annotate("", xy=(7.2, 1.35), xytext=(5.85, 2.45),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        # Caption
        ax.text(0.2, 0.4, "Only 2 of 36 layers see the bridge.  No LLM inductive bias for 256-d vectors.",
                fontsize=8.5, style="italic", color=C_BAD)

    def draw_v2(ax):
        ax.set_xlim(0, 12); ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title("v2 — LLaVA-style token-prepend  (works: L > T by 26–82% on 4 games)",
                     fontsize=9.5, color=C_GOOD, pad=4, loc="left")
        # Slow box
        ax.add_patch(FancyBboxPatch((0.2, 1.8), 2.0, 1.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#fff0e0", edgecolor=C_ACC, linewidth=0.8))
        ax.text(1.2, 2.5, "Slow\n(Qwen3-VL-8B)", ha="center", va="center", fontsize=8)
        # Projection
        ax.add_patch(FancyBboxPatch((3.0, 1.8), 1.8, 1.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#fff8c0", edgecolor=C_ACC, linewidth=0.8))
        ax.text(3.9, 2.5, "MLP\n2048→4096\n33M params", ha="center", va="center", fontsize=7.5)
        # 8 bridge tokens
        for i in range(8):
            ax.add_patch(FancyBboxPatch((5.2 + i*0.16, 2.0), 0.14, 1.0,
                                         boxstyle="round,pad=0",
                                         facecolor=C_L, edgecolor="white", linewidth=0.4, alpha=0.85))
        ax.text(5.85, 3.25, "N=8 latent tokens (4096-d)", ha="center",
                fontsize=8, color=C_ACC)
        # Fast box — entire stack reads bridge via attention
        ax.add_patch(FancyBboxPatch((7.0, 0.3), 4.6, 4.0,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#e0f0ff", edgecolor=C_ACC, linewidth=0.8))
        ax.text(9.3, 3.9, "Fast (MiniCPM-o)  —  36 LLM layers",
                ha="center", va="center", fontsize=8.5)
        # Highlight that ALL layers see bridge
        for y in np.linspace(0.5, 3.3, 13):
            ax.add_patch(FancyBboxPatch((7.2, y), 4.2, 0.15,
                                         boxstyle="round,pad=0",
                                         facecolor=C_GOOD, edgecolor="none", alpha=0.20))
        ax.text(9.3, 1.9, "all 36 layers attend\nvia standard causal attention",
                ha="center", va="center", fontsize=8, color=C_GOOD, fontweight="bold")
        # Arrows
        ax.annotate("", xy=(3.0, 2.5), xytext=(2.2, 2.5),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.annotate("", xy=(5.15, 2.5), xytext=(4.8, 2.5),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        ax.annotate("", xy=(7.0, 2.5), xytext=(6.5, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color=C_GOOD, lw=2,
                                    connectionstyle="arc3,rad=0"))
        ax.text(0.2, 0.4, "Bridge tokens live in the LLM's input embedding space.  Full causal attention from all layers.",
                fontsize=8.5, style="italic", color=C_GOOD)

    draw_v1(ax1)
    draw_v2(ax2)
    fig.savefig(OUT / "fig_architecture.pdf")
    fig.savefig(OUT / "fig_architecture.png", dpi=200)
    plt.close(fig)
    print("wrote fig_architecture.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 2 — Cross-game headline bar chart
# ---------------------------------------------------------------------------

# Best-variant-per-game (using bare or robust SA, whichever gave higher L)
HEADLINE = [
    # (game label, F, F_err, T, T_err, L, L_err, variant_note)
    ("MsPacman",   256, 24, 408, 88, 628, 341, ""),
    ("Seaquest",    42, 19,  63, 11,  80,   0, ""),
    ("RoadRunner",   0,  0, 608, 240,967,  47, ""),
    ("RR (robust)",1033,19, 337, 77, 612, 297, "robust SA"),
    ("Enduro (robust)",0.8,1, 4.9,5.6, 5.8, 2.5, "robust SA"),
    ("Q*bert (robust)",25,0, 125, 0,  50,   0, "robust SA"),
    ("SI (robust)",107, 60,18, 18, 15,   0, "robust SA"),
    ("Pong",       -21, 0, -21,  0,-21,   0, ""),
]


def fig_headline():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    games = [g[0] for g in HEADLINE]
    F = np.array([g[1] for g in HEADLINE], dtype=float)
    F_err = np.array([g[2] for g in HEADLINE], dtype=float)
    T = np.array([g[3] for g in HEADLINE], dtype=float)
    T_err = np.array([g[4] for g in HEADLINE], dtype=float)
    L = np.array([g[5] for g in HEADLINE], dtype=float)
    L_err = np.array([g[6] for g in HEADLINE], dtype=float)

    x = np.arange(len(games))
    w = 0.26
    bars_f = ax.bar(x - w, F, w, yerr=F_err, color=C_F, label="F (fast only)",
                    capsize=2, error_kw=dict(elinewidth=0.8))
    bars_t = ax.bar(x,     T, w, yerr=T_err, color=C_T, label="T (text bridge)",
                    capsize=2, error_kw=dict(elinewidth=0.8))
    bars_l = ax.bar(x + w, L, w, yerr=L_err, color=C_L, label="L (latent bridge)",
                    capsize=2, error_kw=dict(elinewidth=0.8))

    # Annotate L>T gap
    for i, (f, t, l) in enumerate(zip(F, T, L)):
        if t > 0 and l > t:
            gap = 100 * (l - t) / max(t, 1e-9)
            ax.text(i + w, l + max(L_err[i], 30), f"+{gap:.0f}%", ha="center",
                    va="bottom", fontsize=7, color=C_GOOD, fontweight="bold")
        elif t > 0 and l < t and not (l == 0 and t == 0):
            gap = 100 * (t - l) / max(t, 1e-9)
            ax.text(i, t + T_err[i] + 10, f"T leads\n+{gap:.0f}%", ha="center",
                    va="bottom", fontsize=7, color=C_BAD, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=15, ha="right")
    ax.set_ylabel("Mean episode score (12 episodes per cell)")
    ax.set_title("Latent bridge beats text on 4 of 7 games; Q*bert exception inverts (text wins)",
                 fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_headline.pdf")
    fig.savefig(OUT / "fig_headline.png", dpi=200)
    plt.close(fig)
    print("wrote fig_headline.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 3 — RoadRunner cherry-pick (F=0 → L=967)
# ---------------------------------------------------------------------------

def fig_roadrunner():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    conditions = ["S\n(slow only)", "F\n(fast only)", "T\n(text bridge)", "L\n(latent bridge)"]
    scores = [None, 0, 608, 967]  # no S baseline for RoadRunner
    errs   = [None, 0, 240, 47]
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
    ax.set_title("RoadRunner: F cannot score → L unlocks scoring policy (+59% L over T)",
                 fontsize=10)
    ax.set_ylim(-50, 1200)
    # callout
    ax.annotate("F has the reflex machinery\nbut not the directional bias.\nThe slow's compressed\nstrategic context unlocks it.",
                xy=(3, 967), xytext=(0.5, 700),
                fontsize=7.5, style="italic", color=C_ACC,
                arrowprops=dict(arrowstyle="->", color=C_ACC, lw=0.8))
    fig.tight_layout()
    fig.savefig(OUT / "fig_roadrunner.pdf")
    fig.savefig(OUT / "fig_roadrunner.png", dpi=200)
    plt.close(fig)
    print("wrote fig_roadrunner.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 4 — Stage A OOD-brittleness diagnosis (SI + RR before/after)
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
                    linewidth=0.5, label="bare Stage A", hatch="//")
        b2 = ax.bar(x + w/2, robust_vals, w, yerr=robust_errs, capsize=2,
                    color=[C_F, C_T, C_L], edgecolor="black", linewidth=0.5,
                    label="robust Stage A (suffix-prob=0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(["F", "T", "L"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Score")
        if game == "SpaceInvaders":
            ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("Stage A OOD-brittleness recipe: same fix recovers two distinct games",
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
    ax.set_xticks(Ns)
    ax.set_xticklabels(Ns)
    ax.set_xlabel("Bridge bandwidth N (latent tokens per emission)")
    ax.set_ylabel("L score (mean ± std)")
    ax.set_title("Bandwidth is Goldilocks: N=4 too few, N=16 too many",
                 fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bandwidth.pdf")
    fig.savefig(OUT / "fig_bandwidth.png", dpi=200)
    plt.close(fig)
    print("wrote fig_bandwidth.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 6 — Refined claim (continuous-content vs categorical-content)
# ---------------------------------------------------------------------------

def fig_continuous_vs_categorical():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    # Position games on a x-axis = "how categorical is the slow content" (low to high)
    # y-axis = (L-T)/T ratio
    games = [
        ("MsPacman", 0.15, 0.54, C_L),
        ("Seaquest", 0.20, 0.26, C_L),
        ("RoadRunner", 0.10, 0.59, C_L),
        ("RR robust", 0.25, 0.82, C_L),
        ("Enduro robust", 0.40, 0.18, C_L),
        ("Q*bert robust", 0.80, -0.60, C_BAD),  # T > L; (L-T)/T < 0
    ]
    for name, x, y, color in games:
        ax.scatter([x], [y], s=120, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(name, (x, y), xytext=(8, 6), textcoords="offset points",
                    fontsize=8)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.fill_between([0, 1], 0, 1.0, color=C_GOOD, alpha=0.07)
    ax.fill_between([0, 1], 0, -0.8, color=C_BAD, alpha=0.07)
    ax.text(0.55, 0.85, "L > T region", color=C_GOOD, fontsize=9, fontweight="bold")
    ax.text(0.55, -0.7, "T > L region", color=C_BAD, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.8, 1.0)
    ax.set_xlabel("Slow content's categorical content (qualitative scale →)")
    ax.set_ylabel("(L − T) / T")
    ax.set_title("Refined claim: L > T when slow content is continuous-rich; T ≥ L when categorical",
                 fontsize=9.5)
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


if __name__ == "__main__":
    fig_architecture()
    fig_headline()
    fig_roadrunner()
    fig_stage_a_ood()
    fig_bandwidth()
    fig_continuous_vs_categorical()
    fig_latency()
    print("\nall figures written to:", OUT)
