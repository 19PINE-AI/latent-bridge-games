"""Post-hoc statistical analysis for the latent-bridge paper.

Reads results/eval_v2_*.json files (raw per-episode scores), computes:
  * mean, std, 95% bootstrap CI per (game, variant, strategy) cell.
  * Welch's t and Mann-Whitney U for each (game, T vs L) comparison.
  * Effect size (Cohen's d) per comparison.
  * Per-game slow-emission lengths (proxy for "categorical-vs-continuous") from
    results/t_trajectories_v2/*_seed0.pt.

Outputs:
  paper/figures/stats.json         -- machine-readable
  paper/figures/stats_table.tex    -- LaTeX results table
  paper/figures/stats_summary.md   -- human-readable summary
"""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures"

EVAL_FILES = {
    # (game, variant): filename
    ("Enduro", "bare"): "eval_v2_enduro.json",
    ("Enduro", "robust"): "eval_v2_enduro_robust.json",
    ("MsPacman", "bare"): "eval_v2_mspacman.json",
    ("MsPacman", "robust"): "eval_v2_mspacman_robust.json",
    ("Pong", "bare"): "eval_v2_pong.json",
    ("Qbert", "bare"): "eval_v2_qbert.json",
    ("Qbert", "robust"): "eval_v2_qbert_robust.json",
    ("RiverRaid", "bare"): "eval_v2_riverraid.json",
    ("RiverRaid", "robust"): "eval_v2_riverraid_robust.json",
    ("RoadRunner", "bare"): "eval_v2_roadrunner.json",
    ("RoadRunner", "robust"): "eval_v2_roadrunner_robust.json",
    ("Seaquest", "bare"): "eval_v2_seaquest.json",
    ("Seaquest", "robust"): "eval_v2_seaquest_robust.json",
    ("SpaceInvaders", "bare"): "eval_v2_spaceinvaders.json",
    ("SpaceInvaders", "robust"): "eval_v2_spaceinvaders_robust.json",
}

BANDWIDTH_FILES = {
    # (n_tokens, kind): filename. "true" varies collection + training + deploy N.
    (4, "deploy_only"): "eval_v2_mspacman_n4.json",
    (16, "deploy_only"): "eval_v2_mspacman_n16.json",
    (4, "true"): "eval_v2_mspacman_true_n4.json",
    (16, "true"): "eval_v2_mspacman_true_n16.json",
}


def load_scores(fname: str) -> Dict[str, List[float]]:
    with (RESULTS / fname).open() as f:
        d = json.load(f)
    out: Dict[str, List[float]] = {}
    for cell in d.get("cells", []):
        out.setdefault(cell["strategy"], []).append(float(cell["score"]))
    return out


def bootstrap_ci(xs: List[float], n_boot: int = 10_000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> Tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    if not xs:
        return (float("nan"), float("nan"))
    arr = np.asarray(xs, dtype=float)
    if arr.std(ddof=1) == 0 or len(arr) == 1:
        return (float(arr.mean()), float(arr.mean()))
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def welch_t(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Welch's t-test statistic and approximate two-sided p (using scipy if available)."""
    from scipy import stats  # type: ignore

    if not xs or not ys:
        return (float("nan"), float("nan"))
    if statistics.pstdev(xs) == 0 and statistics.pstdev(ys) == 0:
        return (float("nan"), 1.0 if statistics.mean(xs) == statistics.mean(ys) else 0.0)
    t, p = stats.ttest_ind(xs, ys, equal_var=False)
    return float(t), float(p)


def mann_whitney(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    from scipy import stats  # type: ignore

    if not xs or not ys:
        return (float("nan"), float("nan"))
    if len(set(xs + ys)) == 1:
        return (float("nan"), 1.0)
    res = stats.mannwhitneyu(xs, ys, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def cohens_d(xs: List[float], ys: List[float]) -> float:
    if not xs or not ys:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = statistics.variance(xs)
    vy = statistics.variance(ys)
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return float("nan") if mx != my else 0.0
    return (mx - my) / pooled


def summarize_cell(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    arr = np.asarray(xs, dtype=float)
    lo, hi = bootstrap_ci(xs)
    return {
        "n": len(xs),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def emission_stats() -> Dict[str, Dict[str, float]]:
    """Per-game slow-emission stats. Quantitative axis for the
    continuous-vs-categorical claim:
      * unique_per_emission: lexical diversity (higher = continuous-rich)
      * gzip_ratio: gzip(joined)/len(joined). Lower = more redundant/categorical.
      * numeric_density: digits per char (higher = more coordinates/budgets).
    """
    import gzip
    import re

    import torch  # type: ignore

    out: Dict[str, Dict[str, float]] = {}
    for game in ["RoadRunner", "Qbert", "MsPacman", "Riverraid", "Seaquest",
                 "SpaceInvaders", "Enduro"]:
        path = RESULTS / "t_trajectories_v2" / f"{game}_seed0.pt"
        if not path.exists():
            continue
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        emissions = [tick.get("slow_text", "") for tick in d.get("trace", [])
                     if isinstance(tick, dict) and tick.get("slow_text")]
        if not emissions:
            continue
        lens = [len(e) for e in emissions]
        all_toks: List[str] = []
        for e in emissions:
            all_toks.extend(e.split())
        joined = "\n".join(emissions).encode("utf-8")
        gz = gzip.compress(joined, compresslevel=9)
        digit_count = sum(1 for ch in joined.decode("utf-8") if ch.isdigit())
        # number of distinct integers (proxy for coordinate-budget content)
        numbers = re.findall(r"-?\d+", joined.decode("utf-8"))
        out[game] = {
            "n_emissions": len(emissions),
            "char_mean": float(statistics.mean(lens)),
            "char_median": float(statistics.median(lens)),
            "tok_total": len(all_toks),
            "tok_unique": len(set(all_toks)),
            "unique_per_emission": len(set(all_toks)) / len(emissions),
            "gzip_ratio": len(gz) / max(len(joined), 1),
            "numeric_density": digit_count / max(len(joined), 1),
            "distinct_numbers": len(set(numbers)),
            "numbers_per_emission": len(numbers) / len(emissions),
        }
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    stats: Dict[str, Dict] = {"cells": {}, "comparisons": {}, "bandwidth": {}, "emissions": {}}

    for (game, variant), fname in EVAL_FILES.items():
        if not (RESULTS / fname).exists():
            continue
        scores = load_scores(fname)
        cell_summary = {strat: summarize_cell(s) for strat, s in scores.items()}
        stats["cells"][f"{game}/{variant}"] = {
            "raw": {strat: s for strat, s in scores.items()},
            "summary": cell_summary,
        }
        if "T" in scores and "L" in scores:
            t_scores, l_scores = scores["T"], scores["L"]
            cmp_key = f"{game}/{variant}/L_vs_T"
            stats["comparisons"][cmp_key] = {
                "T_mean": statistics.mean(t_scores) if t_scores else float("nan"),
                "L_mean": statistics.mean(l_scores) if l_scores else float("nan"),
                "delta_pct": (statistics.mean(l_scores) - statistics.mean(t_scores))
                              / max(statistics.mean(t_scores), 1e-9) * 100
                              if t_scores and statistics.mean(t_scores) != 0 else float("nan"),
                "welch_t": welch_t(l_scores, t_scores),
                "mann_whitney": mann_whitney(l_scores, t_scores),
                "cohens_d_LvT": cohens_d(l_scores, t_scores),
            }

    for (n_tok, kind), fname in BANDWIDTH_FILES.items():
        if not (RESULTS / fname).exists():
            continue
        scores = load_scores(fname)
        l = scores.get("L", [])
        stats["bandwidth"][f"N={n_tok}/{kind}"] = summarize_cell(l)

    stats["emissions"] = emission_stats()

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2, default=lambda o: None)

    # ---- LaTeX results table ----
    lines = []
    lines.append("% Auto-generated by analysis.py — do not edit by hand.")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Game (variant) & F & T & L & $\\Delta_{L-T}$\\,(\\%, $p$) \\\\")
    lines.append("\\midrule")
    order = [
        ("MsPacman", "bare", "MsPacman"),
        ("MsPacman", "robust", "MsPacman (robust SA)"),
        ("Seaquest", "bare", "Seaquest"),
        ("Seaquest", "robust", "Seaquest (robust SA)"),
        ("RoadRunner", "bare", "RoadRunner"),
        ("RoadRunner", "robust", "RoadRunner (robust SA)"),
        ("RiverRaid", "bare", "River Raid"),
        ("RiverRaid", "robust", "River Raid (robust SA)"),
        ("Enduro", "bare", "Enduro"),
        ("Enduro", "robust", "Enduro (robust SA)"),
        ("Qbert", "bare", "Q*bert"),
        ("Qbert", "robust", "Q*bert (robust SA)"),
        ("SpaceInvaders", "bare", "SpaceInvaders"),
        ("SpaceInvaders", "robust", "SpaceInvaders (robust SA)"),
        ("Pong", "bare", "Pong"),
    ]
    for game, variant, label in order:
        key = f"{game}/{variant}"
        if key not in stats["cells"]:
            continue
        sm = stats["cells"][key]["summary"]
        def fmt(strat: str) -> str:
            if strat not in sm:
                return "--"
            d = sm[strat]
            mean = d["mean"]
            std = d["std"]
            return f"{mean:.0f}\\,$\\pm$\\,{std:.0f}"
        cmp_key = f"{game}/{variant}/L_vs_T"
        cmp = stats["comparisons"].get(cmp_key)
        if cmp and not math.isnan(cmp.get("delta_pct", float("nan"))):
            dp = cmp["delta_pct"]
            welch_p = cmp["welch_t"][1]
            mwu_p = cmp["mann_whitney"][1]
            # Prefer Welch when defined and finite; fall back to MWU when Welch
            # is degenerate (one or both cells have zero variance).
            sm = stats["cells"][key]["summary"]
            zero_var = (sm.get("T", {}).get("std", 0) == 0
                        or sm.get("L", {}).get("std", 0) == 0)
            if not zero_var and welch_p is not None and not math.isnan(welch_p):
                p, tag = welch_p, ""
            elif mwu_p is not None and not math.isnan(mwu_p):
                p, tag = mwu_p, "$^{\\dagger}$"  # marked as MWU
            else:
                p, tag = float("nan"), ""
            if math.isnan(p):
                psymbol = "n/a"
            elif p < 0.001:
                psymbol = "$p$<.001"
            elif p < 0.01:
                psymbol = f"$p$={p:.3f}"
            else:
                psymbol = f"$p$={p:.2f}"
            delta_str = f"{dp:+.0f}\\,\\%\\,({psymbol}{tag})"
        else:
            delta_str = "--"
        lines.append(f"{label} & {fmt('F')} & {fmt('T')} & {fmt('L')} & {delta_str} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "stats_table.tex").write_text("\n".join(lines))

    # ---- human summary ----
    md = ["# Stats summary\n"]
    md.append("## Cell summaries (mean ± std, 95% bootstrap CI)\n")
    for (game, variant), fname in EVAL_FILES.items():
        key = f"{game}/{variant}"
        if key not in stats["cells"]:
            continue
        md.append(f"### {game} / {variant}\n")
        for strat, d in stats["cells"][key]["summary"].items():
            md.append(
                f"- {strat}: n={d['n']}, mean={d['mean']:.1f}, std={d['std']:.1f}, "
                f"95% CI=[{d['ci_lo']:.1f}, {d['ci_hi']:.1f}], min={d['min']:.0f}, max={d['max']:.0f}"
            )
        md.append("")

    md.append("## L vs T comparisons\n")
    for k, v in stats["comparisons"].items():
        if math.isnan(v["welch_t"][1]):
            line = (f"- {k}: T={v['T_mean']:.1f}, L={v['L_mean']:.1f}, "
                    f"Welch t={v['welch_t'][0]:.2f} p=n/a, MWU={v['mann_whitney']}, "
                    f"Cohen's d={v['cohens_d_LvT']:.2f}")
        else:
            line = (f"- {k}: T={v['T_mean']:.1f}, L={v['L_mean']:.1f}, "
                    f"Δ={v['delta_pct']:+.1f}%, Welch t={v['welch_t'][0]:.2f} p={v['welch_t'][1]:.4f}, "
                    f"MWU p={v['mann_whitney'][1]:.4f}, Cohen's d={v['cohens_d_LvT']:.2f}")
        md.append(line)

    md.append("\n## Bandwidth sweep (MsPacman)\n")
    for k, v in stats["bandwidth"].items():
        md.append(f"- {k}: n={v['n']}, mean={v['mean']:.1f}, std={v['std']:.1f}, "
                  f"95% CI=[{v['ci_lo']:.1f}, {v['ci_hi']:.1f}]")

    md.append("\n## Slow-emission stats (per game, seed 0 trajectory)\n")
    for game, d in stats["emissions"].items():
        md.append(f"- {game}: n_emissions={d['n_emissions']}, char_mean={d['char_mean']:.0f}, "
                  f"unique_per_emission={d['unique_per_emission']:.2f}, "
                  f"gzip_ratio={d['gzip_ratio']:.3f}, "
                  f"numbers_per_emission={d['numbers_per_emission']:.2f}, "
                  f"distinct_numbers={d['distinct_numbers']}")

    (OUT / "stats_summary.md").write_text("\n".join(md))
    print(f"wrote stats.json, stats_table.tex, stats_summary.md to {OUT}")


if __name__ == "__main__":
    main()
