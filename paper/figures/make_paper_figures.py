"""Generate paper-quality figures for the Latent Bridge arXiv submission.

Produces (in paper/figures/):
  0. fig_system.pdf            — whole-system runtime loop + F/T/L + training pipeline
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
# Figure 0 — System architecture: async fast/slow runtime loop, F/T/L channels,
#            and the Stage A/B/C training pipeline.
# ---------------------------------------------------------------------------

def fig_system():
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.10)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])
    for a in (ax, axt):
        a.axis("off")
        a.set_xlim(0, 14)
    ax.set_ylim(0, 9.9)
    axt.set_ylim(0, 2.4)

    def box(a, x, y, w, h, fc, ec, text, fs=7.5, tc="black", lw=0.9,
            weight="normal", style="round,pad=0.06"):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                   facecolor=fc, edgecolor=ec, linewidth=lw))
        a.text(x + w / 2, y + h / 2, text, ha="center", va="center",
               fontsize=fs, color=tc, fontweight=weight)

    def arrow(a, x0, y0, x1, y1, color="grey", lw=1.2, ls="-", style="->"):
        a.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                   linestyle=ls))

    # ---------------- fast half (top) ----------------
    # Environment
    box(ax, 0.2, 6.0, 2.1, 1.9, "#f5f5f5", "0.4",
        "Environment\nAtari @ 15 Hz\nMetaDrive @ 10 Hz", fs=7.5)

    # Fast-loop container
    ax.add_patch(FancyBboxPatch((3.1, 4.6), 10.7, 4.3, boxstyle="round,pad=0.06",
                                facecolor="#eef6ff", edgecolor=C_ACC, linewidth=1.0))
    ax.text(3.4, 8.55, "Fast reactive loop  —  MiniCPM-o 4.5 (9 B, frozen)  —  "
                       "one action per ~67 ms tick",
            fontsize=8, color=C_ACC, ha="left", va="center", fontweight="bold")

    # vision tower
    box(ax, 5.0, 7.1, 1.9, 1.0, "white", C_ACC, "vision tower\n(frozen)", fs=7)
    arrow(ax, 2.35, 7.3, 4.95, 7.6)                     # env frame -> vision tower
    ax.text(3.3, 7.62, "frame", fontsize=6.5, color="0.35", ha="center",
            va="bottom")

    # input token strip: [L prefix][vision][state prompt][T suffix]
    sy, sh = 5.6, 0.85
    # L prefix: 8 thin red bars (same motif as fig_architecture)
    for i in range(8):
        ax.add_patch(FancyBboxPatch((3.55 + i * 0.155, sy), 0.135, sh,
                                    boxstyle="round,pad=0", facecolor=C_L,
                                    edgecolor="white", linewidth=0.4, alpha=0.9))
    ax.text(4.17, sy + sh + 0.12, "L: 8 latent tokens\n(prepended)", fontsize=6.6,
            color=C_L, ha="center", va="bottom", fontweight="bold")
    box(ax, 4.95, sy, 1.75, sh, "#e4efe4", "0.55", "vision\ntokens", fs=6.8,
        style="round,pad=0.02")
    box(ax, 6.8, sy, 1.75, sh, "#f4f4f4", "0.55", "game-state\nprompt", fs=6.8,
        style="round,pad=0.02")
    box(ax, 8.65, sy, 2.0, sh, "#dbeafe", C_T, "T: slow text suffix\n(appended)",
        fs=6.6, tc="#1d5e8a", style="round,pad=0.02")
    arrow(ax, 5.9, 7.1, 5.85, sy + sh + 0.05, color="0.55", lw=1.0)  # vision -> strip

    # LLM + action head
    box(ax, 11.0, 5.6, 1.25, 1.5, "white", C_ACC, "36-layer\nLLM\n(frozen)", fs=6.8)
    box(ax, 12.45, 5.6, 1.2, 1.5, "white", C_ACC, "action\nhead\n(Stage A)", fs=6.8)
    arrow(ax, 10.7, sy + sh / 2, 10.95, 6.35, color="0.3", lw=1.4)
    arrow(ax, 12.3, 6.35, 12.42, 6.35, color="0.3", lw=1.4)
    # action back to env: right-angle route around the top of the fast box
    ax.plot([13.05, 13.05], [7.15, 9.35], color="0.3", lw=1.2)
    ax.plot([13.05, 1.25], [9.35, 9.35], color="0.3", lw=1.2)
    arrow(ax, 1.25, 9.35, 1.25, 8.0, color="0.3", lw=1.2)
    ax.text(7.2, 9.46, "action (greedy argmax over game actions), every tick",
            fontsize=6.8, color="0.3", ha="center", va="bottom")

    # F/T/L definition note inside fast box (bottom right, clear of the red arc)
    ax.text(13.55, 5.22,
            "F = no colored segment\nT = + blue suffix   ·   L = + red prefix",
            fontsize=6.6, color="0.25", style="italic", ha="right", va="center")

    # ---------------- async divider ----------------
    ax.plot([0.2, 13.8], [4.0, 4.0], ls=(0, (4, 3)), color="0.6", lw=0.9)
    ax.text(0.25, 4.12, "synchronous, ~15 Hz", fontsize=6.8, color="0.4",
            ha="left", va="bottom", style="italic")
    ax.text(0.25, 3.86, "asynchronous, ~1 Hz", fontsize=6.8, color="0.4",
            ha="left", va="top", style="italic")

    # ---------------- slow half (bottom) ----------------
    box(ax, 0.2, 1.6, 2.1, 1.5, "#f5f5f5", "0.4",
        "structured state\n(RAM objects /\ndriving state)", fs=7)
    arrow(ax, 1.25, 5.95, 1.25, 3.2, color="0.55", lw=1.0)  # env -> structured state

    box(ax, 3.1, 1.35, 3.0, 2.0, "#fff0e0", C_ACC,
        "Slow — Qwen3-VL-8B-\nThinking (8 B, frozen)\n~1.5 s per emission",
        fs=7.5)
    arrow(ax, 2.35, 2.35, 3.05, 2.35, color="0.55", lw=1.0)

    # text emission -> T channel
    box(ax, 7.0, 2.5, 2.3, 1.0, "#dbeafe", C_T,
        "text emission\n(~300 chars)", fs=7, tc="#1d5e8a")
    arrow(ax, 6.15, 2.95, 6.95, 3.0, color=C_T, lw=1.3)
    arrow(ax, 9.65, 3.5, 9.65, sy - 0.08, color=C_T, lw=1.6)

    # residuals -> MLP -> L channel
    box(ax, 7.0, 0.5, 2.3, 1.0, "#f4f4f4", "0.5",
        "layer-24 residuals\n(last 8 positions)", fs=7)
    arrow(ax, 6.15, 1.75, 6.95, 1.1, color="0.55", lw=1.0)
    box(ax, 10.0, 0.35, 2.6, 1.3, "#fff8c0", C_ACC,
        "bridge MLP 4096→4096\n33 M params — the only\ntrained component", fs=7)
    arrow(ax, 9.35, 1.0, 9.95, 1.0, color="0.55", lw=1.0)
    # MLP up to L prefix in the strip
    ax.annotate("", xy=(4.17, sy - 0.08), xytext=(10.6, 1.72),
                arrowprops=dict(arrowstyle="-|>", color=C_L, lw=1.8,
                                connectionstyle="arc3,rad=0.22"))
    # async behaviour note, bottom-left empty corner
    ax.text(0.2, 0.95, "The fast loop never blocks on the slow model;\n"
                       "the latest emission is reused (~15 ticks)\n"
                       "until the next one replaces it.",
            fontsize=6.6, color="0.4", style="italic", ha="left", va="top")

    # ---------------- training strip (bottom panel) ----------------
    axt.text(0.2, 2.15, "Training pipeline (per game; both base models stay frozen):",
             fontsize=7.8, color="0.25", ha="left", va="center", fontweight="bold")
    box(axt, 0.2, 0.25, 4.2, 1.5, "#f0f4f8", "0.45",
        "Stage A — action head\nbehavioral cloning from SB3 expert\n"
        "(bare, or robust: suffix-prob 0.5)", fs=7)
    box(axt, 4.9, 0.25, 4.2, 1.5, "#f0f4f8", "0.45",
        "Stage B — data\nroll out T; cache (frame, slow text,\nlayer-24 residuals)",
        fs=7)
    box(axt, 9.6, 0.25, 4.2, 1.5, "#f0f4f8", "0.45",
        "Stage C — bridge\ntrain the MLP only:  KL$(\\pi_L\\,\\|\\,\\pi_T)$\n"
        "(~5K samples/game, final KL ≈ 0.005)", fs=7)
    arrow(axt, 4.45, 1.0, 4.85, 1.0, color="0.45", lw=1.2)
    arrow(axt, 9.15, 1.0, 9.55, 1.0, color="0.45", lw=1.2)

    fig.savefig(OUT / "fig_system.pdf")
    fig.savefig(OUT / "fig_system.png", dpi=200)
    plt.close(fig)
    print("wrote fig_system.{pdf,png}")


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
        ax.text(3.9, 2.5, "MLP\n4096→4096\n33M params", ha="center", va="center", fontsize=7.5)
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
        pvals.append((cmp.get("welch_t", [None, None])[1],
                      cmp.get("mann_whitney", [None, None])[1]))

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
        p, p_mwu = pvals[i]
        if p is None or (isinstance(p, float) and np.isnan(p)):
            sig = ""
        elif p < 0.001: sig = "***"
        elif p < 0.01:  sig = "**"
        elif p < 0.05:  sig = "*"
        else:           sig = "n.s."
        # When the two tests disagree (Welch n.s. but MWU significant, e.g.
        # bimodal MsPacman), annotate both rather than letting Welch hide it.
        if sig == "n.s." and p_mwu is not None and p_mwu < 0.05:
            mwu_sig = "***" if p_mwu < 0.001 else "**" if p_mwu < 0.01 else "*"
            sig = f"W n.s.\nMWU {mwu_sig}"
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
    ax.set_title("Cross-game F/T/L (reported variant per game).  "
                 "L beats T on 4 games; Q*bert inverts; 2 games are ties.  "
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
        # label placement
        dx, dy = 0.06 * lim, 0.06 * lim
        ha = "left"
        if row["game"] in ("SpaceInvaders", "Riverraid"):
            dy = -0.12 * lim
        if row["game"] == "RoadRunner":
            dx = -0.06 * lim; ha = "right"
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
    print("\nall figures written to:", OUT)
