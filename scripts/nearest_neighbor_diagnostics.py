"""Richer diagnostics on the trained bridge tokens.

Beyond the top-k nearest-neighbor decode in nearest_neighbor_decode.py:

1. Norm comparison
   - L2 norm of bridge tokens vs L2 norm distribution of vocab embeddings.
   - If bridges are much larger/smaller than vocab tokens, cosine-NN is the
     right metric, not L2.

2. Cosine-to-vocab distribution
   - For each bridge token, compute cosine similarity against every one of the
     151,748 vocab embeddings. Report (max, mean, p99, p999). This shows
     whether the "top-k" tokens we saw above are genuinely close (max≈1) or
     just the maximum of a uniform-low distribution (max≈0.1).

3. Game-conditioning
   - For each pair of (game_a, game_b), compute mean cosine sim between their
     bridge tokens. If bridge encodes game-specific info, within-game sim >
     across-game sim.

4. Across-emission consistency per game
   - Within a game, do bridge tokens vary across emissions (encoding per-tick
     state) or stay constant (degenerate)? Reported as std-of-cosine across
     emission pairs.

Run after nearest_neighbor_decode.py. Independent of GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open


class ThoughtProjection(nn.Module):
    def __init__(self, hidden_dim_in: int, hidden_dim_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim_in),
            nn.Linear(hidden_dim_in, hidden_dim_out),
            nn.GELU(),
            nn.Linear(hidden_dim_out, hidden_dim_out),
            nn.LayerNorm(hidden_dim_out),
        )

    def forward(self, x):
        return self.proj(x)


FAST_MODEL_DIR = "/home/ubuntu/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/snapshots/6e885630cbe907859c441ff915aa789729f3a5c4"
EMBED_TENSOR_NAME = "llm.model.embed_tokens.weight"
EMBED_SAFETENSORS = f"{FAST_MODEL_DIR}/model-00001-of-00004.safetensors"

BRIDGE_VARIANT = {
    "MsPacman":      "v2_mspacman.pt",
    "Seaquest":      "v2_seaquest.pt",
    "RoadRunner":    "v2_roadrunner.pt",
    "Riverraid":     "v2_riverraid_robust.pt",
    "Enduro":        "v2_enduro_robust.pt",
    "Qbert":         "v2_qbert_robust.pt",
    "SpaceInvaders": "v2_spaceinvaders_robust.pt",
}
TRACE_GLOB = {g: f"results/t_trajectories_v2/{g}_seed0.pt"
              for g in BRIDGE_VARIANT}


def load_fast_embed_tokens() -> torch.Tensor:
    print(f"Loading {EMBED_TENSOR_NAME}...")
    with safe_open(EMBED_SAFETENSORS, framework="pt", device="cpu") as f:
        emb = f.get_tensor(EMBED_TENSOR_NAME)
    return emb.float()


def load_projection(path: str) -> ThoughtProjection:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["slow_projection_state"]
    linear1_w = state["proj.1.weight"]
    out_dim, in_dim = linear1_w.shape
    proj = ThoughtProjection(in_dim, out_dim)
    proj.load_state_dict(state)
    proj.eval()
    return proj


def project_emissions(game: str, max_emissions: int) -> torch.Tensor:
    """Returns [E, N, D] bridge tokens for up to max_emissions emissions of `game`."""
    bridge_ckpt = f"checkpoints/stage_c/{BRIDGE_VARIANT[game]}"
    trace_path = TRACE_GLOB[game]
    if not (os.path.exists(bridge_ckpt) and os.path.exists(trace_path)):
        return torch.empty(0)
    proj = load_projection(bridge_ckpt)
    blob = torch.load(trace_path, map_location="cpu", weights_only=False)
    out = []
    for item in blob["trace"]:
        if item.get("slow_text") and item.get("slow_vecs") is not None:
            with torch.no_grad():
                b = proj(item["slow_vecs"].float().unsqueeze(0)).squeeze(0)
            out.append(b)
            if len(out) >= max_emissions:
                break
    if not out:
        return torch.empty(0)
    return torch.stack(out)  # [E, N, D]


def cosine_to_vocab_summary(token_vec: torch.Tensor,
                            embed_norm: torch.Tensor) -> dict:
    """token_vec [D], embed_norm [V, D] L2-normalized. Returns stats."""
    tn = token_vec / (token_vec.norm() + 1e-9)
    sims = embed_norm @ tn  # [V]
    sims_np = sims.numpy()
    return {
        "max":  float(sims_np.max()),
        "mean": float(sims_np.mean()),
        "std":  float(sims_np.std()),
        "p99":  float(np.percentile(sims_np, 99)),
        "p999": float(np.percentile(sims_np, 99.9)),
        "min":  float(sims_np.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["MsPacman", "RoadRunner", "Qbert", "Seaquest",
                             "Riverraid", "SpaceInvaders", "Enduro"])
    ap.add_argument("--max-emissions-per-game", type=int, default=8)
    ap.add_argument("--out", default="results/nn_diagnostics.json")
    args = ap.parse_args()

    embed = load_fast_embed_tokens()              # [V, D] float32
    embed_norm = embed / (embed.norm(dim=-1, keepdim=True) + 1e-9)
    vocab_norms = embed.norm(dim=-1)
    print(f"  vocab_size={embed.shape[0]}, dim={embed.shape[1]}")
    print(f"  vocab norm dist: mean={vocab_norms.mean():.3f} "
          f"std={vocab_norms.std():.3f} "
          f"p1={np.percentile(vocab_norms.numpy(),1):.3f} "
          f"p99={np.percentile(vocab_norms.numpy(),99):.3f}")

    # Project bridge tokens for each game
    game_bridges = {}
    for g in args.games:
        b = project_emissions(g, args.max_emissions_per_game)
        if b.numel() == 0:
            print(f"  {g}: skipped (no data)")
            continue
        game_bridges[g] = b
        print(f"  {g}: {tuple(b.shape)}  "
              f"norm mean={b.flatten(0,1).norm(dim=-1).mean():.3f} "
              f"std={b.flatten(0,1).norm(dim=-1).std():.3f}")

    # --- (1) Norm comparison ---
    norms_out = {
        "vocab": {
            "mean": float(vocab_norms.mean()),
            "std":  float(vocab_norms.std()),
            "p1":   float(np.percentile(vocab_norms.numpy(), 1)),
            "p50":  float(np.percentile(vocab_norms.numpy(), 50)),
            "p99":  float(np.percentile(vocab_norms.numpy(), 99)),
        },
        "bridges": {
            g: {
                "mean": float(b.flatten(0, 1).norm(dim=-1).mean()),
                "std":  float(b.flatten(0, 1).norm(dim=-1).std()),
            }
            for g, b in game_bridges.items()
        },
    }

    # --- (2) Cosine-to-vocab distribution per bridge token ---
    print("\nCosine-to-vocab distribution per bridge position (one emission per game):")
    cos_out = {}
    for g, b in game_bridges.items():
        first_em = b[0]  # [N, D]
        per_pos = []
        for pos in range(first_em.shape[0]):
            stats = cosine_to_vocab_summary(first_em[pos], embed_norm)
            per_pos.append(stats)
        cos_out[g] = per_pos
        avg_max = np.mean([p["max"] for p in per_pos])
        avg_p999 = np.mean([p["p999"] for p in per_pos])
        print(f"  {g}: avg_max={avg_max:.3f}  avg_p99.9={avg_p999:.3f}  "
              f"(if max≈1, latents look like text; max≈p99.9 ≈ uniform low)")

    # --- (3) Within-game vs across-game similarity ---
    print("\nGame-conditioning:")
    games_list = list(game_bridges.keys())
    # For each game pair, compute mean cosine between bridge tokens (averaged
    # over emissions and positions).
    pair_sim = {}
    for ga in games_list:
        ba = game_bridges[ga].flatten(0, 1)  # [E*N, D]
        ba_n = ba / (ba.norm(dim=-1, keepdim=True) + 1e-9)
        for gb in games_list:
            bb = game_bridges[gb].flatten(0, 1)
            bb_n = bb / (bb.norm(dim=-1, keepdim=True) + 1e-9)
            sim_mat = ba_n @ bb_n.T  # [E_a*N, E_b*N]
            if ga == gb:
                # mask diagonal (self-self)
                eye = torch.eye(sim_mat.shape[0], dtype=torch.bool)
                vals = sim_mat[~eye]
            else:
                vals = sim_mat.flatten()
            pair_sim[f"{ga}__vs__{gb}"] = float(vals.mean())
    # Print as matrix
    print("                   " + " ".join(f"{g[:8]:>8s}" for g in games_list))
    for ga in games_list:
        row = [f"{pair_sim[f'{ga}__vs__{gb}']:+.3f}" for gb in games_list]
        print(f"  {ga[:18]:18s} " + " ".join(f"{r:>8s}" for r in row))

    # --- (4) Within-game across-emission consistency ---
    print("\nWithin-game across-emission consistency (cosine between emission means):")
    consist = {}
    for g, b in game_bridges.items():
        emission_means = b.mean(dim=1)  # [E, D]
        emission_means_n = emission_means / (emission_means.norm(dim=-1, keepdim=True) + 1e-9)
        sim = emission_means_n @ emission_means_n.T  # [E, E]
        eye = torch.eye(sim.shape[0], dtype=torch.bool)
        off = sim[~eye]
        consist[g] = {
            "n_emissions": int(b.shape[0]),
            "off_diag_mean": float(off.mean()),
            "off_diag_std":  float(off.std()),
        }
        print(f"  {g}: off_diag_mean={consist[g]['off_diag_mean']:+.3f} "
              f"std={consist[g]['off_diag_std']:.3f}  "
              f"(1.0=identical, lower=more diverse per emission)")

    out = {
        "norms": norms_out,
        "cosine_to_vocab_per_position": cos_out,
        "game_pair_cosine": pair_sim,
        "within_game_consistency": consist,
    }
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
