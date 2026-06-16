#!/usr/bin/env python3
"""Held-out (leave-one-seed-out) per-channel decoder selection — review-proof best-achievable.

For each (game, channel), we treat the action decoder (greedy / sampling temperature) as a
deployment hyperparameter and select it the honest way: tune on held-in seeds, report on the
held-out seed. With 3 ALE seeds (each = 4 episodes) we do leave-one-seed-out:

  for held-out seed s in {0,1,2}:
      pick the decoder with the best mean over the OTHER two seeds (8 episodes)
      score = that decoder's 4 episodes on seed s
  held-out estimate = the 12 held-out episodes (each seed contributes its 4)

This is an unbiased estimate of "tune the decoder, then deploy", unlike the oracle max over
decoders on the same episodes (which we also print, as the optimistic upper bound).
"""
import json, glob, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

RESDIR = "results/rev_sampling"

# decoder label -> json suffix (a decoder is included for a game only if its file exists)
DECODERS = [
    ("greedy", "greedy_reconfirm"),
    ("t0.3",   "sample_t03"),
    ("t0.5",   "sample_t05"),
    ("t0.7",   "sample_t07"),
    ("t1.0",   "sample_t1"),
    ("t1.5",   "sample_t15"),
]
GAMES = [
    ("MsPacman",   "mspacman_bare"),
    ("RoadRunner", "roadrunner_bare"),
    ("River Raid", "riverraid_robust"),
    ("Seaquest",   "seaquest_bare"),
    ("Q*bert",     "qbert_robust"),
    ("Enduro",     "enduro_robust"),
    ("SpaceInvaders", "spaceinvaders_robust"),
]

def load(tag):
    """game tag -> {decoder_label: {strategy: {base_seed: [4 episode scores]}}}.
    F/T/L come from <tag>_<suffix>.json; the combined channel B (if present) comes from the
    separate <tag>_B_<suffix>.json, merged under the same decoder label."""
    out = {}
    for lab, suf in DECODERS:
        bystrat = defaultdict(lambda: defaultdict(list))
        for path in (f"{RESDIR}/{tag}_{suf}.json", f"{RESDIR}/{tag}_B_{suf}.json"):
            if not os.path.exists(path):
                continue
            for c in json.load(open(path))["cells"]:
                base = c["seed"] // 1000       # 0,1,2,3 -> 0 ; 1000.. -> 1 ; 2000.. -> 2
                bystrat[c["strategy"]][base].append(c["score"])
        if bystrat:
            out[lab] = {s: dict(v) for s, v in bystrat.items()}
    return out

def heldout_mean(decoders, strat):
    """leave-one-seed-out selection for one strategy. returns (heldout_scores, picks)."""
    seeds = sorted({s for lab in decoders for s in decoders[lab].get(strat, {})})
    held = []
    picks = []
    for s in seeds:
        # mean over the other seeds, per decoder
        best_lab, best_m = None, -1e18
        for lab in decoders:
            d = decoders[lab].get(strat, {})
            train = [x for ss, ep in d.items() if ss != s for x in ep]
            if not train:
                continue
            m = np.mean(train)
            if m > best_m:
                best_m, best_lab = m, lab
        if best_lab is None:
            continue
        held += decoders[best_lab][strat].get(s, [])
        picks.append((s, best_lab))
    return np.array(held, dtype=float), picks

def oracle_best(decoders, strat):
    best_lab, best_m, best_vec = None, -1e18, None
    for lab in decoders:
        vec = [x for ep in decoders[lab].get(strat, {}).values() for x in ep]
        if vec and np.mean(vec) > best_m:
            best_m, best_lab, best_vec = np.mean(vec), lab, vec
    return best_lab, best_m, np.array(best_vec, dtype=float)

def main():
    print(f"{'Game':11} {'channel':7} | {'held-out best (picks)':38} | {'oracle best':16}")
    print("-" * 92)
    summary = []
    for name, tag in GAMES:
        decoders = load(tag)
        if not decoders:
            print(f"{name:11} (no data yet)"); continue
        row = {"game": name}
        present = [s for s in ["F", "T", "L", "B"] if any(s in decoders[l] for l in decoders)]
        for strat in present:
            ho, picks = heldout_mean(decoders, strat)
            ol_lab, ol_m, ol_vec = oracle_best(decoders, strat)
            row[strat] = {"ho": ho, "ho_mean": float(np.mean(ho)) if len(ho) else float("nan"),
                          "picks": picks, "oracle_lab": ol_lab, "oracle_mean": ol_m}
            pk = ",".join(f"{s}:{l}" for s, l in picks)
            print(f"{name:11} {strat:7} | {np.mean(ho):6.0f} ({pk:28}) | {ol_m:6.0f} ({ol_lab})")

        def cmp(a, b, label):
            A, B = row.get(a, {}).get("ho"), row.get(b, {}).get("ho")
            if A is None or B is None or not len(A) or not len(B):
                return
            p = stats.ttest_ind(A, B, equal_var=False).pvalue
            u = stats.mannwhitneyu(A, B, alternative="two-sided").pvalue
            dm = 100 * (np.mean(A) - np.mean(B)) / np.mean(B) if np.mean(B) else float("nan")
            rel = ">" if np.mean(A) > np.mean(B) else ("<" if np.mean(A) < np.mean(B) else "=")
            sig = "sig" if min(p, u) < 0.05 else "n.s."
            print(f"{'':11} {label:7} | held-out {a}{rel}{b} {dm:+.0f}%  Welch p={p:.2f}  MWU p={u:.2f}  [{sig}]")
            row[label] = dict(rel=rel, dpct=dm, p=p, u=u, sig=(min(p, u) < 0.05))

        cmp("L", "T", "L vs T")
        if "B" in row:
            cmp("B", "T", "B vs T")
            cmp("B", "L", "B vs L")
            # does "both" beat the BETTER of the two single channels (per-game best)?
            bestsingle = "T" if row["T"]["ho_mean"] >= row["L"]["ho_mean"] else "L"
            cmp("B", bestsingle, "B vs best")
        print()
        summary.append(row)
    return summary

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
