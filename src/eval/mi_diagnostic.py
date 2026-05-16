"""Mutual-information diagnostic for the trained bridge.

From the experiment plan:
> Mutual information between thought vectors and (a) optimal action, (b)
> post-30-tick reward, on a held-out trajectory set. Diagnostic — confirms the
> bridge isn't collapsed.

We do a *practical* version with the cached T-trajectory data:
  - X = projected bridge tokens at each emission (re-projected through the
    trained Stage C projection, then mean-pooled over the N=8 token dim).
  - Y_action = the discretized action taken at the next fast-tick (proxy for
    "optimal action", since with random rollout we don't have a Bayes-optimal
    target — but the bridge should still be MI'd to the action because the
    fast model's logits depend on the bridge).
  - Y_reward = sign(future_30_tick_reward).

We compute I(X; Y) via the binning + plug-in entropy estimator (Kraskov is
overkill for a sanity check; with ~5K samples binning is fine).

This runs OFFLINE — no fast/slow forward passes during the diagnostic itself.
Only the slow.projection (loaded from the v2 bridge checkpoint) is applied to
the cached raw residuals.

Usage:
  python -m src.eval.mi_diagnostic \\
    --trace 'results/t_trajectories_v2/MsPacman_seed*.pt' \\
    --bridge-ckpt checkpoints/stage_c/v2_mspacman.pt \\
    --out results/mi_diagnostic_mspacman.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _load_projection_from_ckpt(ckpt_path: str, slow_hidden: int = 4096,
                               bridge_dim: int = 4096) -> nn.Module:
    """Load the trained ThoughtProjection from a v2 Stage C checkpoint."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.models.slow_model import ThoughtProjection

    proj_module = ThoughtProjection(hidden_dim_in=slow_hidden, hidden_dim_out=bridge_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ckpt.get("v2"):
        raise ValueError(f"checkpoint {ckpt_path!r} is not v2 (missing v2=True)")
    proj_module.load_state_dict(ckpt["slow_projection_state"])
    return proj_module


def collect_pairs(trace_paths: list[str], projection: nn.Module,
                  reward_window: int = 30) -> dict:
    """Walk T trajectories, project the cached raw residuals, collect (X, Y) pairs.

    Returns:
      X: [N_emit, bridge_dim] mean-pooled bridge tokens (one per emission)
      y_action: [N_emit] action_global at the tick of the emission
      y_reward: [N_emit] sign of next-`reward_window`-ticks cumulative reward
    """
    projection.eval()
    # The trained ThoughtProjection is a module whose `forward(residual)` returns
    # the projected tokens. Its inner `proj` is the Sequential.
    bridge_dim = projection.proj[3].out_features  # type: ignore[index]

    X_list = []
    y_action_list = []
    y_reward_list = []
    n_emissions = 0
    for p in trace_paths:
        blob = torch.load(p, weights_only=False)
        trace = blob["trace"]
        n_total_rewards = [t["reward"] for t in trace]
        for ti, item in enumerate(trace):
            if item["slow_vecs"] is None or item["slow_text"] is None:
                continue
            n_emissions += 1
            # Project raw residuals via the trained projection
            residuals = item["slow_vecs"].float()  # [N, 4096] CPU float32
            if residuals.numel() == 0:
                continue
            with torch.no_grad():
                tokens = projection(residuals.unsqueeze(0))  # [1, N, bridge_dim]
                pooled = tokens.mean(dim=1).squeeze(0).numpy()  # [bridge_dim]
            X_list.append(pooled)
            y_action_list.append(int(item["action"]))
            future_reward = sum(n_total_rewards[ti:ti + reward_window])
            y_reward_list.append(int(np.sign(future_reward)))

    return {
        "X": np.stack(X_list, axis=0) if X_list else np.zeros((0, bridge_dim)),
        "y_action": np.array(y_action_list, dtype=np.int64),
        "y_reward": np.array(y_reward_list, dtype=np.int64),
        "n_emissions": n_emissions,
    }


def _entropy_from_counts(counts: np.ndarray) -> float:
    p = counts / max(1, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mi_via_binning(X: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
    """Estimate I(X; Y) with axis-wise binning.

    X: [N, D] continuous. y: [N] discrete.
    For each feature dim, discretize via quantile bins; sum the per-feature MI
    (an upper bound on true MI but a sane proxy when features are roughly
    independent — fine for a diagnostic).

    Returns: total per-dim MI summed.
    """
    if X.shape[0] == 0:
        return 0.0
    N, D = X.shape
    H_y = _entropy_from_counts(np.bincount(y - y.min()))
    H_y_given_X_per_dim = []
    for d in range(D):
        col = X[:, d]
        # Discretize using quantile bins
        qs = np.quantile(col, np.linspace(0, 1, n_bins + 1))
        # Ensure strictly increasing bins
        qs = np.unique(qs)
        if len(qs) < 2:
            H_y_given_X_per_dim.append(H_y)  # no info
            continue
        xb = np.digitize(col, qs[1:-1]) if len(qs) > 2 else np.zeros_like(col, dtype=int)
        H_y_cond = 0.0
        for b in np.unique(xb):
            mask = xb == b
            if mask.sum() == 0:
                continue
            p_b = mask.sum() / N
            counts = np.bincount(y[mask] - y.min())
            H_y_cond += p_b * _entropy_from_counts(counts)
        H_y_given_X_per_dim.append(H_y_cond)
    # Per-dim MI = H(Y) - H(Y|X_d); average across dims
    mi_per_dim = [H_y - h for h in H_y_given_X_per_dim]
    return float(np.mean(mi_per_dim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", nargs="+", required=True)
    ap.add_argument("--bridge-ckpt", required=True)
    ap.add_argument("--reward-window", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trace_paths = []
    for p in args.trace:
        m = sorted(glob.glob(p)) or [p]
        trace_paths.extend(m)

    print(f"Loading bridge projection from {args.bridge_ckpt}...")
    proj = _load_projection_from_ckpt(args.bridge_ckpt)

    print(f"Collecting (X, action, reward_sign) pairs from {len(trace_paths)} trace(s)...")
    data = collect_pairs(trace_paths, proj, reward_window=args.reward_window)

    X = data["X"]
    y_action = data["y_action"]
    y_reward = data["y_reward"]
    print(f"  N emissions used: {X.shape[0]}")
    print(f"  X shape: {X.shape}")
    print(f"  action distribution: {dict(zip(*np.unique(y_action, return_counts=True)))}")
    print(f"  reward_sign distribution: {dict(zip(*np.unique(y_reward, return_counts=True)))}")

    print("\nEstimating MI...")
    mi_action = mi_via_binning(X, y_action, n_bins=args.n_bins)
    mi_reward = mi_via_binning(X, y_reward, n_bins=args.n_bins)
    # Baselines: same X, random y
    rng = np.random.default_rng(0)
    y_action_shuf = rng.permutation(y_action)
    y_reward_shuf = rng.permutation(y_reward)
    mi_action_baseline = mi_via_binning(X, y_action_shuf, n_bins=args.n_bins)
    mi_reward_baseline = mi_via_binning(X, y_reward_shuf, n_bins=args.n_bins)

    summary = {
        "n_emissions": int(data["n_emissions"]),
        "n_used": int(X.shape[0]),
        "bridge_dim": int(X.shape[1]) if X.size > 0 else 0,
        "reward_window_ticks": args.reward_window,
        "n_bins": args.n_bins,
        "mi_action_nats": mi_action,
        "mi_action_baseline_shuffled": mi_action_baseline,
        "mi_action_minus_baseline": mi_action - mi_action_baseline,
        "mi_reward_nats": mi_reward,
        "mi_reward_baseline_shuffled": mi_reward_baseline,
        "mi_reward_minus_baseline": mi_reward - mi_reward_baseline,
    }

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    print("\nSummary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("\nInterpretation: positive values of `*_minus_baseline` mean the bridge "
          "carries information about that Y; near-zero means the bridge is collapsed "
          "(no signal beyond chance). Action MI is the primary diagnostic.")


if __name__ == "__main__":
    main()
