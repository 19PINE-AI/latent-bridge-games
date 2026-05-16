"""Stage C v2: LLaVA-style latent-as-token bridge training.

Differences from v1 (`stage_c_bridge.py`):
  - Bridge mechanism: NOT mid-layer cross-attention; bridge tokens are PREPENDED
    to fast model's input embedding sequence (LLaVA pattern).
  - Trainable params: only the slow's ThoughtProjection (~32M MLP). The fast model
    is entirely frozen — no xattn weights to learn, no action_head tuning.
  - Inputs from T trajectories: raw slow residuals at layer 24, last N positions,
    4096-d each. Stage C re-projects them every forward pass via the trainable
    ThoughtProjection.

Training loss: KL(student || teacher) where
  - Teacher: fast.predict_action(frame, slow_text_suffix=<emission>) → logits_text
  - Student: fast.predict_action(frame, thought_tokens=proj(residuals)) → logits_latent

KL is computed over legal-action indices only (avoids 0 × -inf from masked positions).

Usage:
  HF_HUB_OFFLINE=1 python -m src.training.stage_c_v2 \\
    --trace 'results/t_trajectories_v2/MsPacman_seed*.pt' \\
    --stage-a-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \\
    --epochs 1 --lr 5e-5 --grad-accum 4 \\
    --out checkpoints/stage_c/v2_mspacman.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.fast_model import FastModel, FastModelConfig
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.stage_c_data import StageCDataset


def _train_step(fast: FastModel, slow: SlowModel, sample: dict,
                kl_temp: float = 1.0) -> dict:
    """One v2 Stage C training step.

    Teacher: fast(frame, slow_text_suffix=text) — no_grad.
    Student: fast(frame, thought_tokens=slow.project(residuals)) — grad on projection.
    Loss: KL(student || teacher) over legal actions.

    `slow.projection` is the only set of parameters that backprop flows into.
    """
    frame = sample["frame"].numpy()
    legal = sample["legal_action_mask"]
    slow_text = sample["slow_text"]
    raw_residuals = sample["slow_vecs"]  # [N, slow_hidden_dim=4096], float32 on CPU

    if slow_text is None or raw_residuals is None or raw_residuals.numel() == 0:
        return {"skipped": True}

    # Teacher (frozen): text suffix path
    with torch.no_grad():
        teacher_logits = fast.predict_action(
            frame, thought_tokens=None,
            legal_action_mask=legal,
            slow_text_suffix=slow_text,
        ).float()

    # Student (grad on projection): re-project residuals through current projection
    residuals = raw_residuals.unsqueeze(0).to(slow.cfg.device)  # [1, N, 4096]
    bridge_tokens = slow.project_residuals(residuals)  # [1, N, 4096] in fast's input embedding space
    student_logits = fast.predict_action(
        frame, thought_tokens=bridge_tokens,
        legal_action_mask=legal,
        slow_text_suffix=None,  # v2 latent-only at student
    ).float()

    # KL over legal indices to avoid 0 * -inf
    legal_idx = legal.nonzero(as_tuple=True)[0].to(student_logits.device)
    s = student_logits.index_select(-1, legal_idx) / kl_temp
    t = teacher_logits.index_select(-1, legal_idx) / kl_temp
    log_s = F.log_softmax(s, dim=-1)
    log_t = F.log_softmax(t, dim=-1)
    p_t = log_t.exp()
    kl = (p_t * (log_t - log_s)).sum(dim=-1).mean()
    return {"loss": kl, "skipped": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-a-ckpt", default=None,
                    help="Stage A checkpoint to load action_head from")
    ap.add_argument("--out", default="checkpoints/stage_c/v2_bridge.pt")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Expand globs in trace list
    trace_paths = []
    for p in args.trace:
        matches = sorted(glob.glob(p)) or [p]
        trace_paths.extend(matches)

    print(f"v2 Stage C bridge training")
    print(f"Loading {len(trace_paths)} trace file(s)...")
    ds = StageCDataset(trace_paths)
    print(f"  total samples: {len(ds)}")

    if args.max_samples and args.max_samples < len(ds):
        ds = torch.utils.data.Subset(
            ds, np.random.RandomState(args.seed).permutation(len(ds))[:args.max_samples].tolist()
        )
        print(f"  subsampled to: {len(ds)}")

    # ---- Models ----
    print("Loading FastModel (frozen)...")
    fast = FastModel(FastModelConfig()).load_pretrained()
    if args.stage_a_ckpt:
        ckpt = torch.load(args.stage_a_ckpt, map_location="cuda", weights_only=False)
        fast.action_head.load_state_dict(ckpt["action_head_state"])
        print(f"  loaded action_head from {args.stage_a_ckpt}")
    # Freeze EVERYTHING in fast (action_head, base, any v1 xattn weights that exist)
    for p in fast.parameters():
        p.requires_grad = False
    print(f"  fast frozen: {sum(p.numel() for p in fast.parameters()):,} params")

    print("Loading SlowModel (base frozen; ThoughtProjection trainable)...")
    slow = SlowModel(SlowModelConfig()).load_pretrained()
    # slow.model is already frozen in load_pretrained. The projection is trainable
    # by default (it's a new module). Confirm.
    for p in slow.model.parameters():
        p.requires_grad = False
    proj_params = list(slow.projection.parameters())
    for p in proj_params:
        p.requires_grad = True
    print(f"  trainable: {sum(p.numel() for p in proj_params):,} "
          f"(slow.projection only)")

    optimizer = torch.optim.AdamW(proj_params, lr=args.lr, weight_decay=0.0)

    # ---- Training loop ----
    indices = list(range(len(ds)))
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        losses, n_train, n_skip = [], 0, 0
        t_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, idx in enumerate(indices):
            sample = ds[idx]
            out = _train_step(fast, slow, sample, kl_temp=args.kl_temp)
            if out.get("skipped"):
                n_skip += 1
                continue
            loss = out["loss"]
            (loss / args.grad_accum).backward()
            if (n_train + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(proj_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            n_train += 1
            if (step + 1) % 50 == 0:
                print(f"  [ep {epoch} step {step+1}/{len(indices)}] "
                      f"KL={np.mean(losses[-50:]):.4f}  trained={n_train}  "
                      f"skipped={n_skip}  elapsed={time.time()-t_start:.0f}s", flush=True)

        print(f"Epoch {epoch}: mean KL={np.mean(losses) if losses else 0:.4f}  "
              f"trained={n_train}  skipped={n_skip}  "
              f"({time.time()-t_start:.0f}s)")

    # ---- Save ----
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({
        "v2": True,
        "slow_projection_state": slow.projection.state_dict(),
        "slow_config": vars(slow.cfg),
        "fast_config": vars(fast.cfg),
        "args": vars(args),
    }, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
