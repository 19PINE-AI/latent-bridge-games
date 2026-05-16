"""Stage C: Bridge supervised training via COCONUT-style curriculum.

The student is the fast model's latent-conditioned prediction (thought-vector buffer
populated, slow_text_suffix=None). The teacher is the same fast model's text-
conditioned prediction (slow_text_suffix=<emission>, thought_buffer=None). Both
forwards use the same Atari frame.

Curriculum stages:
  C0: bridge is a no-op (zero-init); fast model trains on text suffix only. This is
      effectively Stage A's continuation with the slow text in context — Stage A
      logits ≈ "teacher" logits, used to bootstrap the action head.
  C1: thought_buffer fed with the SECOND HALF of the emission's latent vectors.
      Supervision: KL(student || teacher) so the bridge learns to recover what
      the dropped text would have provided.
  C2: thought_buffer fed with ALL of the emission's latent vectors; slow_text_suffix
      is empty. Same KL supervision.

In all stages the slow model itself is unused during Stage C — we work from cached
trajectories produced by `scripts/run_text_bridge_baseline.py`. Bridge parameters
(cross-attn Q/K/V/O at fast-model layers 12, 24) are the only trainable weights;
the action head can also be unfrozen if Stage A wasn't sufficient.

Status: skeleton — the data path + curriculum mechanics + KL loss are wired, and
the optimizer steps the bridge xattn. Actual long-form training is GPU-bound and
will be run when GPU budget permits.

Usage:
    HF_HUB_OFFLINE=1 python -m src.training.stage_c_bridge \\
        --trace results/t_trajectories_MsPacman_seed0.pt \\
        --stage C2 --epochs 1 --batch-size 1 --grad-accum 4 \\
        --out checkpoints/stage_c/c2_mspacman.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.fast_model import FastModel, FastModelConfig
from src.bridge.ring_buffer import ThoughtBuffer
from src.training.stage_c_data import StageCDataset


def _collate(batch):
    """Custom collate — frames vary in size; keep them as a list."""
    return batch  # we process samples one at a time (batch_size=1)


def _build_thought_buffer(slow_vecs: torch.Tensor, stage: str,
                          capacity: int = 16, dim: int = 256,
                          device: str = "cuda") -> torch.Tensor:
    """Materialize a thought-buffer tensor for the curriculum stage.

    C1: take the second half of slow_vecs (the rest is dropped from text and replaced
        by what the bridge transmits).
    C2: take all of slow_vecs.
    Returns: [1, K, D_bridge] tensor on device.
    """
    if stage == "C2":
        used_vecs = slow_vecs
    elif stage == "C1":
        L = slow_vecs.shape[0]
        used_vecs = slow_vecs[L // 2:]
    else:
        raise ValueError(f"unknown stage {stage!r}")
    # Pack into ring buffer of capacity K (most-recent wins). Cap to last K vectors.
    used_vecs = used_vecs[-capacity:]
    if used_vecs.shape[0] < capacity:
        pad = torch.zeros(capacity - used_vecs.shape[0], dim, dtype=used_vecs.dtype)
        used_vecs = torch.cat([pad, used_vecs], dim=0)
    return used_vecs.unsqueeze(0).to(device, dtype=torch.bfloat16)


def _truncated_text(text: str, stage: str) -> str:
    """C1 also truncates the text in half — we drop the second half and rely on the
    latent half-injection to recover it. C2 drops all text. C0 keeps all text.
    """
    if stage == "C0" or text is None:
        return text
    if stage == "C2":
        return ""  # rely on latents only
    # C1: keep first half
    words = text.split()
    return " ".join(words[: len(words) // 2])


def _train_step(fast: FastModel, sample: dict, stage: str,
                kl_temp: float = 1.0) -> dict:
    """One Stage C training step on a single sample.

    Teacher (no grad): full text suffix, no thought buffer.
    Student (grad on bridge): truncated text + latent buffer per the stage.
    Loss: KL(student || teacher) over the legal-action logits.
    """
    frame = sample["frame"].numpy()
    legal = sample["legal_action_mask"]
    slow_text = sample["slow_text"]
    slow_vecs = sample["slow_vecs"]

    if slow_text is None or slow_vecs is None:
        # No emission active yet — nothing to align. Skip.
        return {"skipped": True, "reason": "no_emission"}

    # 1) Teacher: text-only, no bridge
    with torch.no_grad():
        teacher_logits = fast.predict_action(
            frame, thought_buffer=None,
            legal_action_mask=legal,
            slow_text_suffix=slow_text,
        ).float()

    # 2) Student: truncated text + latent buffer (requires grad)
    student_text = _truncated_text(slow_text, stage)
    tb = _build_thought_buffer(slow_vecs, stage=stage)
    student_logits = fast.predict_action(
        frame, thought_buffer=tb,
        legal_action_mask=legal,
        slow_text_suffix=student_text,
    ).float()

    # KL divergence over the legal-action distribution. Compute on legal-only logits
    # so that illegal positions (-inf) don't generate 0 * -inf = NaN. predict_action
    # uses the same legal-mask on both teacher and student, so the legal sets agree.
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
    ap.add_argument("--trace", nargs="+", required=True,
                    help="path(s) to T-condition trajectory .pt files")
    ap.add_argument("--stage", choices=("C0", "C1", "C2"), default="C2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/stage_c/bridge.pt")
    ap.add_argument("--stage-a-ckpt", default=None,
                    help="Stage A checkpoint to load action_head from (required for "
                         "meaningful KL training — without it teacher logits are uniform).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"Stage {args.stage} bridge training")
    print(f"Loading traces: {args.trace}")
    ds = StageCDataset(args.trace)
    print(f"  total samples: {len(ds)}")

    if args.max_samples and args.max_samples < len(ds):
        ds = torch.utils.data.Subset(
            ds, np.random.RandomState(args.seed).permutation(len(ds))[:args.max_samples].tolist()
        )
        print(f"  subsampled to: {len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_collate)

    print("Loading FastModel...")
    fast = FastModel(FastModelConfig()).load_pretrained()
    if args.stage_a_ckpt:
        ckpt = torch.load(args.stage_a_ckpt, map_location="cuda", weights_only=False)
        fast.action_head.load_state_dict(ckpt["action_head_state"])
        print(f"  loaded action_head from {args.stage_a_ckpt} "
              f"(val_acc={ckpt.get('val_acc', '?')})")
    # Freeze action_head during Stage C — bridge xattn is the only trainable part.
    for p in fast.action_head.parameters():
        p.requires_grad = False
    print(f"  trainable: {sum(p.numel() for p in fast.xattn_layers.parameters()):,} "
          f"(bridge xattn only; action_head frozen)")

    optimizer = torch.optim.AdamW(
        list(fast.xattn_layers.parameters()), lr=args.lr, weight_decay=0.0,
    )

    for epoch in range(1, args.epochs + 1):
        losses, n_train, n_skip = [], 0, 0
        t_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            sample = batch[0]
            out = _train_step(fast, sample, stage=args.stage, kl_temp=args.kl_temp)
            if out.get("skipped"):
                n_skip += 1
                continue
            loss = out["loss"]
            (loss / args.grad_accum).backward()
            if (n_train + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(fast.xattn_layers.parameters()), max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            n_train += 1
            if (step + 1) % 20 == 0:
                print(f"  [ep {epoch} step {step+1}/{len(loader)}] "
                      f"KL={np.mean(losses[-20:]):.4f}  trained={n_train}  "
                      f"skipped={n_skip}  elapsed={time.time()-t_start:.0f}s", flush=True)

        print(f"Epoch {epoch}: mean KL={np.mean(losses) if losses else 0:.4f}  "
              f"trained={n_train}  skipped={n_skip}  "
              f"({time.time()-t_start:.0f}s)")

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": args.stage,
        "xattn_state": {k: m.state_dict() for k, m in fast.xattn_layers.items()},
        "config": vars(fast.cfg),
        "args": vars(args),
    }, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
