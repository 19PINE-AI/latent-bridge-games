"""Direct test of the Stage A OOD-brittleness diagnosis (paper §5).

Diagnosis (paraphrased): Stage A trains the action head on the bare game-state
prompt. At deployment, T appends a text suffix; this is an OOD input for the
head. On precision-required games (SpaceInvaders, River Raid, Q*bert) the
OOD-induced drift collapses scoring to zero; on reward-symmetric games
(MsPacman, RoadRunner, Seaquest) the drift still scores.

Prediction: KL(action_head(bare) || action_head(T-suffixed)) per emission
should be SYSTEMATICALLY HIGHER on collapsed games than on non-collapsed games.

This script:
  1. Loads the fast model + bare Stage A action head for each game.
  2. Walks cached t_trajectories_v2/<game>_seed0.pt, finds ticks with a
     slow_text emission.
  3. For each such tick, computes action-head logits under bare prompt and
     under T-suffixed prompt, then KL between them over legal actions.
  4. Reports per-game mean ± std + comparison against the collapse outcome.

Needs ~18-20 GB GPU (fast model only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.fast_model import FastModel, FastModelConfig
from src.training.imitation_data import legal_action_mask


# Per-paper convention: bare Stage A for games where bare didn't collapse,
# robust for games where bare collapsed. The diagnosis is specifically about
# what happens at deployment under suffix, so we test the *bare* Stage A on
# every game so the OOD-induced drift is directly measurable.
GAMES_BARE_OK = ["MsPacman", "Seaquest", "RoadRunner"]
GAMES_COLLAPSED = ["SpaceInvaders", "Qbert", "Riverraid"]

STAGE_A_CKPT = {
    "MsPacman":      "checkpoints/stage_a/mspacman_sb3dqn_v2.pt",
    "Seaquest":      "checkpoints/stage_a/seaquest_sb3dqn.pt",
    "RoadRunner":    "checkpoints/stage_a/roadrunner_sb3dqn.pt",
    "SpaceInvaders": "checkpoints/stage_a/spaceinvaders_sb3dqn.pt",
    "Qbert":         "checkpoints/stage_a/qbert_sb3dqn.pt",
    "Riverraid":     "checkpoints/stage_a/riverraid_sb3dqn.pt",
}
ROBUST_A_CKPT = {  # for comparison; robust should have near-zero KL by design
    "MsPacman":      "checkpoints/stage_a/mspacman_robust.pt",
    "Seaquest":      "checkpoints/stage_a/seaquest_robust.pt",
    "RoadRunner":    "checkpoints/stage_a/roadrunner_robust.pt",
    "SpaceInvaders": "checkpoints/stage_a/spaceinvaders_robust.pt",
    "Qbert":         "checkpoints/stage_a/qbert_robust.pt",
    "Riverraid":     "checkpoints/stage_a/riverraid_robust.pt",
}
TRACE_PATH = {g: f"results/t_trajectories_v2/{g}_seed0.pt"
              for g in STAGE_A_CKPT}


def per_emission_metrics(fast: FastModel, game: str, trace_path: str,
                          max_emissions: int) -> dict:
    """For each emission tick, compare action-head behavior under bare vs
    suffixed prompt. Returns KL, argmax-change rate, and bare-argmax
    probability under suffix."""
    blob = torch.load(trace_path, map_location="cpu", weights_only=False)
    trace = blob["trace"]
    legal = legal_action_mask(game)
    legal_idx = legal.nonzero(as_tuple=True)[0].to("cuda")

    kls, argmax_changed, bare_argmax_psuff, bare_top1_pb = [], [], [], []
    n_total = 0
    for item in trace:
        text = item.get("slow_text")
        if text is None:
            continue
        frame_np = np.asarray(item["obs"])
        with torch.no_grad():
            logits_bare = fast.predict_action(
                frame_np, thought_tokens=None,
                legal_action_mask=legal,
                slow_text_suffix=None,
            ).float()
            logits_suff = fast.predict_action(
                frame_np, thought_tokens=None,
                legal_action_mask=legal,
                slow_text_suffix=text,
            ).float()
        # Restrict to legal action indices
        b = logits_bare.index_select(-1, legal_idx)
        s = logits_suff.index_select(-1, legal_idx)
        log_pb = F.log_softmax(b, dim=-1)
        log_ps = F.log_softmax(s, dim=-1)
        pb = log_pb.exp()
        ps = log_ps.exp()
        kl = (pb * (log_pb - log_ps)).sum(dim=-1).mean().item()

        # argmax over legal-indices (within the legal subset)
        am_b = int(b.argmax(dim=-1).item())
        am_s = int(s.argmax(dim=-1).item())
        kls.append(float(kl))
        argmax_changed.append(int(am_b != am_s))
        bare_argmax_psuff.append(float(ps.squeeze(0)[am_b].item()))
        bare_top1_pb.append(float(pb.squeeze(0)[am_b].item()))

        n_total += 1
        if max_emissions and n_total >= max_emissions:
            break

    if not kls:
        return {"n": 0}
    return {
        "n": len(kls),
        "kl_mean": float(np.mean(kls)),
        "kl_std": float(np.std(kls)),
        "kl_median": float(np.median(kls)),
        "argmax_change_rate": float(np.mean(argmax_changed)),
        "bare_top1_p_under_suffix": float(np.mean(bare_argmax_psuff)),
        "bare_top1_p_under_bare":  float(np.mean(bare_top1_pb)),
        "raw_kls": kls,
        "raw_argmax_changed": argmax_changed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-emissions", type=int, default=20)
    ap.add_argument("--out", default="results/ood_kl_probe.json")
    args = ap.parse_args()

    print("Loading FastModel (frozen, bare action head per game)...")
    fast = FastModel(FastModelConfig()).load_pretrained()
    for p in fast.parameters():
        p.requires_grad = False

    out = {"bare": {}, "robust": {}, "config": vars(args)}
    all_games = GAMES_BARE_OK + GAMES_COLLAPSED

    for variant, ckpt_map in [("bare", STAGE_A_CKPT),
                              ("robust", ROBUST_A_CKPT)]:
        print(f"\n##### Stage A variant: {variant} #####")
        for game in all_games:
            ckpt_path = ckpt_map[game]
            trace_path = TRACE_PATH[game]
            if not os.path.exists(ckpt_path):
                print(f"  {game} [{variant}]: MISSING {ckpt_path}; skip")
                continue
            if not os.path.exists(trace_path):
                print(f"  {game} [{variant}]: MISSING {trace_path}; skip")
                continue

            print(f"\n=== {game} / {variant} ({os.path.basename(ckpt_path)}) ===")
            ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
            fast.action_head.load_state_dict(ckpt["action_head_state"])

            stats = per_emission_metrics(fast, game, trace_path,
                                          args.max_emissions)
            out[variant][game] = {
                "ckpt": ckpt_path,
                "trace": trace_path,
                "expected_collapse": game in GAMES_COLLAPSED,
                **stats,
            }
            if stats["n"] > 0:
                print(f"  n={stats['n']}  KL={stats['kl_mean']:.3f}±{stats['kl_std']:.3f}  "
                      f"argmax_change_rate={stats['argmax_change_rate']:.2f}  "
                      f"bare_top1_p: bare→suff = "
                      f"{stats['bare_top1_p_under_bare']:.3f}→"
                      f"{stats['bare_top1_p_under_suffix']:.3f}")

    # Summary comparison
    print("\n=== Group summary (bare Stage A) ===")
    for metric in ["kl_mean", "argmax_change_rate", "bare_top1_p_under_suffix"]:
        nb = [out["bare"][g][metric] for g in GAMES_BARE_OK
              if g in out["bare"] and out["bare"][g]["n"] > 0]
        co = [out["bare"][g][metric] for g in GAMES_COLLAPSED
              if g in out["bare"] and out["bare"][g]["n"] > 0]
        if nb and co:
            print(f"  {metric:35s}  non-collapsed = {np.mean(nb):.4f}   "
                  f"collapsed = {np.mean(co):.4f}")

    print("\n=== Group summary (robust Stage A) ===")
    for metric in ["kl_mean", "argmax_change_rate"]:
        nb = [out["robust"][g][metric] for g in GAMES_BARE_OK
              if g in out["robust"] and out["robust"][g]["n"] > 0]
        co = [out["robust"][g][metric] for g in GAMES_COLLAPSED
              if g in out["robust"] and out["robust"][g]["n"] > 0]
        if nb and co:
            print(f"  {metric:35s}  non-collapsed = {np.mean(nb):.4f}   "
                  f"collapsed = {np.mean(co):.4f}")

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
