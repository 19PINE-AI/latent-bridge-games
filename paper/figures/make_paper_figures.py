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


def fig_headline():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    games = [g[0] for g in HEADLINE_ORDER]
    means = {"F": [], "T": [], "L": []}
    errlo = {"F": [], "T": [], "L": []}
    errhi = {"F": [], "T": [], "L": []}
    pvals = []
    for _, game, variant in HEADLINE_ORDER:
        d = _cell(game, variant)
        for s in ("F", "T", "L"):
            m, lo, hi = _ci_yerr(d, s)
            means[s].append(m); errlo[s].append(lo); errhi[s].append(hi)
        cmp = STATS.get("comparisons", {}).get(f"{game}/{variant}/L_vs_T", {})
        pvals.append(cmp.get("welch_t", [None, None])[1])

    x = np.arange(len(games))
    w = 0.26
    for off, s, c in [(-w, "F", C_F), (0, "T", C_T), (w, "L", C_L)]:
        yerr = np.array([errlo[s], errhi[s]])
        ax.bar(x + off, means[s], w, yerr=yerr, color=c,
               label=f"{s} ({ {'F':'fast only','T':'text bridge','L':'latent bridge'}[s] })",
               capsize=2, error_kw=dict(elinewidth=0.8))

    # Annotate effect & sig
    for i in range(len(games)):
        t, l = means["T"][i], means["L"][i]
        if t <= 0 and l <= 0:
            ax.text(i, 5, "n/a", ha="center", va="bottom", fontsize=7, color="grey")
            continue
        if t > 0:
            delta = 100 * (l - t) / max(t, 1e-9)
        else:
            delta = float("inf") if l > 0 else 0.0
        p = pvals[i]
        if p is None or (isinstance(p, float) and np.isnan(p)):
            sig = ""
        elif p < 0.001: sig = "***"
        elif p < 0.01:  sig = "**"
        elif p < 0.05:  sig = "*"
        else:           sig = "n.s."
        color = C_GOOD if l > t else (C_BAD if t > l else "grey")
        if l > t:
            tag = f"+{delta:.0f}%\n{sig}"
            yloc = max(l + errhi["L"][i], t + errhi["T"][i]) + 25
            ax.text(i + w, yloc, tag, ha="center", va="bottom",
                    fontsize=7, color=color, fontweight="bold")
        elif t > l:
            tag = f"T leads\n+{(100*(t-l)/max(l,1e-9)):.0f}%\n{sig}"
            yloc = max(l + errhi["L"][i], t + errhi["T"][i]) + 5
            ax.text(i, yloc, tag, ha="center", va="bottom",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(games, fontsize=8)
    ax.set_ylabel("Mean episode score  (95% bootstrap CI; n=12)")
    ax.set_title("Cross-game F/T/L. L > T on 4 games (***/**); Q*bert inverts (T > L, ***).  "
                 "Asterisks (*) tag robust-SA variants.",
                 fontsize=9.5)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=7.5)
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

    for name, x, y in zip(names, xs, ys):
        color = C_L if y > 0 else C_BAD
        ax.scatter([x], [y], s=110, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        # nudge labels off the point
        ax.annotate(name, (x, y), xytext=(8, 4 if y > 0 else -10),
                    textcoords="offset points", fontsize=8)

    # Trend hint
    if len(xs) >= 3:
        slope, intercept = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs)-0.5, max(xs)+0.5, 50)
        ax.plot(xline, slope * xline + intercept, "--", color="grey",
                lw=0.8, alpha=0.7,
                label=f"linear fit: slope={slope:+.2f}/diversity-unit")

    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("Lexical diversity of slow emissions  "
                  "(unique whitespace tokens / emission)")
    ax.set_ylabel("(L − T) / T")
    ax.set_title("Refined bandwidth claim, quantitative axis:\n"
                 "L > T when slow's content is lexically diverse (continuous-rich).",
                 fontsize=9.5)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
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
