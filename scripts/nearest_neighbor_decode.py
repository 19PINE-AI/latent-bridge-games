"""Nearest-neighbor decoding of trained latent bridge tokens.

For each (game, emission) sample:
  1. Load cached slow_vecs from the T-trajectory (already raw layer-24 residuals).
  2. Project through the trained Stage C v2 MLP to get 8 latent tokens in
     fast LLM's input embedding space (4096-d each).
  3. Find the top-k nearest vocab token embeddings (cosine similarity) for each
     of the 8 latent positions.

The point: ask "what does this latent look like under the fast LLM's lexicon?"
If the latents resemble action verbs / spatial words / numeric tokens, that's
a mechanistic interpretation of what the channel is carrying.

Runs on CPU only — no need to load fast or slow base models.

Usage:
    python3 scripts/nearest_neighbor_decode.py \\
        --games MsPacman RoadRunner Qbert \\
        --top-k 8 --n-emissions 3 \\
        --out results/nn_decode.json
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
from transformers import AutoTokenizer


# Mirror src/models/slow_model.py ThoughtProjection — must match state_dict layout.
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


# Where the fast LLM's embed_tokens lives (split across 4 safetensors files; we
# need only the first one).
FAST_MODEL_DIR = "/home/ubuntu/.cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/snapshots/6e885630cbe907859c441ff915aa789729f3a5c4"
EMBED_TENSOR_NAME = "llm.model.embed_tokens.weight"
EMBED_SAFETENSORS = f"{FAST_MODEL_DIR}/model-00001-of-00004.safetensors"

BRIDGE_VARIANT = {  # canonical Stage-C variant per game (matches headline figure)
    "MsPacman":      "v2_mspacman.pt",            # bare
    "Seaquest":      "v2_seaquest.pt",            # bare
    "RoadRunner":    "v2_roadrunner.pt",          # bare
    "Riverraid":     "v2_riverraid_robust.pt",    # robust
    "Enduro":        "v2_enduro_robust.pt",       # robust
    "Qbert":         "v2_qbert_robust.pt",        # robust
    "SpaceInvaders": "v2_spaceinvaders_robust.pt", # robust
}

TRACE_GLOB = {  # which trajectory file to read emissions from
    "MsPacman":      "results/t_trajectories_v2/MsPacman_seed0.pt",
    "Seaquest":      "results/t_trajectories_v2/Seaquest_seed0.pt",
    "RoadRunner":    "results/t_trajectories_v2/RoadRunner_seed0.pt",
    "Riverraid":     "results/t_trajectories_v2/Riverraid_seed0.pt",
    "Enduro":        "results/t_trajectories_v2/Enduro_seed0.pt",
    "Qbert":         "results/t_trajectories_v2/Qbert_seed0.pt",
    "SpaceInvaders": "results/t_trajectories_v2/SpaceInvaders_seed0.pt",
}


def load_fast_embed_tokens() -> torch.Tensor:
    """Load only the input-embedding matrix from the fast model's safetensors.
    Returns a [vocab_size, 4096] tensor in float32."""
    print(f"Loading {EMBED_TENSOR_NAME} from safetensors...")
    with safe_open(EMBED_SAFETENSORS, framework="pt", device="cpu") as f:
        if EMBED_TENSOR_NAME not in f.keys():
            keys = list(f.keys())
            raise RuntimeError(f"{EMBED_TENSOR_NAME!r} not in shard. "
                               f"First 5 keys: {keys[:5]}")
        emb = f.get_tensor(EMBED_TENSOR_NAME)
    print(f"  embed_tokens shape={tuple(emb.shape)} dtype={emb.dtype}")
    return emb.float()


def load_projection(bridge_ckpt_path: str) -> ThoughtProjection:
    ck = torch.load(bridge_ckpt_path, map_location="cpu", weights_only=False)
    assert ck.get("v2"), f"{bridge_ckpt_path} is not a v2 checkpoint"
    state = ck["slow_projection_state"]
    # Infer dims from the first Linear's weight
    linear1_w = state["proj.1.weight"]  # shape [out, in]
    out_dim, in_dim = linear1_w.shape
    proj = ThoughtProjection(in_dim, out_dim)
    proj.load_state_dict(state)
    proj.eval()
    return proj


def sample_emissions(trace_path: str, n_emissions: int, seed: int = 0):
    """Return up to n_emissions (slow_text, slow_vecs) pairs spread across the
    trajectory."""
    blob = torch.load(trace_path, map_location="cpu", weights_only=False)
    trace = blob["trace"]
    emissions = []
    for item in trace:
        if item.get("slow_text") and item.get("slow_vecs") is not None:
            emissions.append((item["slow_text"], item["slow_vecs"]))
    if len(emissions) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(emissions) <= n_emissions:
        chosen = emissions
    else:
        idx = sorted(rng.choice(len(emissions), n_emissions, replace=False))
        chosen = [emissions[i] for i in idx]
    return chosen


def top_k_nearest(latent_vec: torch.Tensor,
                  embed_matrix: torch.Tensor,
                  k: int,
                  metric: str = "cosine"):
    """Given one [D] vector and [V, D] vocab embeddings, return top-k indices
    and similarities (highest first)."""
    if metric == "cosine":
        lv = latent_vec / (latent_vec.norm() + 1e-9)
        em = embed_matrix / (embed_matrix.norm(dim=-1, keepdim=True) + 1e-9)
        sims = em @ lv
    elif metric == "dot":
        sims = embed_matrix @ latent_vec
    else:
        raise ValueError(metric)
    top = sims.topk(k=k)
    return top.indices.tolist(), top.values.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+",
                    default=["MsPacman", "RoadRunner", "Qbert"])
    ap.add_argument("--n-emissions", type=int, default=3,
                    help="emissions per game to sample")
    ap.add_argument("--top-k", type=int, default=8,
                    help="top-k vocab tokens per latent position")
    ap.add_argument("--metric", choices=("cosine", "dot"), default="cosine")
    ap.add_argument("--out", default="results/nn_decode.json")
    args = ap.parse_args()

    # ---- Load tokenizer + embed matrix once ----
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(FAST_MODEL_DIR, trust_remote_code=True)
    embed = load_fast_embed_tokens()

    out = {"per_game": {}, "config": vars(args)}

    for game in args.games:
        ckpt_path = f"checkpoints/stage_c/{BRIDGE_VARIANT[game]}"
        trace_path = TRACE_GLOB[game]
        print(f"\n=== {game} ({BRIDGE_VARIANT[game]}) ===")
        if not os.path.exists(ckpt_path):
            print(f"  MISSING {ckpt_path}; skipping")
            continue
        if not os.path.exists(trace_path):
            print(f"  MISSING {trace_path}; skipping")
            continue

        proj = load_projection(ckpt_path)
        emissions = sample_emissions(trace_path, args.n_emissions, seed=0)
        if not emissions:
            print(f"  no emissions in {trace_path}; skipping")
            continue

        per_emission = []
        for ei, (text, vecs) in enumerate(emissions):
            # vecs: [N, 4096] float32. Project to bridge tokens (still 4096-d).
            vecs_in = vecs.float().unsqueeze(0)  # [1, N, D_in]
            with torch.no_grad():
                bridge = proj(vecs_in).squeeze(0)  # [N, D_out]
            N, D = bridge.shape
            print(f"\n  Emission {ei}: bridge shape={tuple(bridge.shape)}")
            print(f"    text: {text[:120]!r}")

            per_position = []
            for pos in range(N):
                latent = bridge[pos]
                idx_list, sim_list = top_k_nearest(latent, embed,
                                                   k=args.top_k,
                                                   metric=args.metric)
                tokens = [tok.decode([i]).replace("\n", "\\n") for i in idx_list]
                per_position.append({
                    "position": pos,
                    "top_tokens": tokens,
                    "top_sims": [round(s, 4) for s in sim_list],
                    "top_token_ids": idx_list,
                    "latent_norm": float(latent.norm().item()),
                })
                print(f"    pos {pos}: " + " ".join(
                    f"{t!r}({s:.2f})" for t, s in
                    zip(tokens[:5], sim_list[:5])
                ))

            per_emission.append({
                "text": text,
                "positions": per_position,
            })

        out["per_game"][game] = {
            "bridge_ckpt": BRIDGE_VARIANT[game],
            "trace": trace_path,
            "n_emissions_sampled": len(per_emission),
            "emissions": per_emission,
        }

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
