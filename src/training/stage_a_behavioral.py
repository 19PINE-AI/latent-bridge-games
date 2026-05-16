"""Stage A: Fast-only behavioral cloning baseline.

Trains the action head (and optionally LoRA on Qwen3-8B attention) of MiniCPM-o to
predict actions from frames. No bridge involvement — `thought_buffer=None` at every
step, which is a no-op given the zero-init bridge.

Establishes the F (fast-only) baseline score and verifies the fast-side training
pipeline works before any bridge training.

Usage:
    HF_HUB_OFFLINE=1 python -m src.training.stage_a_behavioral \\
        --traj results/trajectories_MsPacman_eps1.0.pt \\
        --epochs 1 --batch-size 1 --grad-accum 8 \\
        --out checkpoints/stage_a/mspacman_action_head.pt

Design notes:
  - Batch size is 1 with grad accumulation because each MiniCPM-o forward pass
    processes a 90-token sequence + vision tower; bs=4+ may OOM.
  - We train only the action head + cross-attention layers' Q/K/V (the bridge cross-attn
    bias stays zero-init so the bridge contribution remains a no-op at Stage A — but
    its Q/K/V projections can pre-warm if `--train-xattn-qkv` is passed).
  - LoRA on backbone attention is added by `--lora` flag (peft).
  - Reward is unused here (this is behavioral cloning); we just predict the recorded action.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.fast_model import FastModel, FastModelConfig
from src.training.imitation_data import (
    ImitationDataset, N_GLOBAL_ACTIONS, legal_action_mask,
)


def _iter_dataset(loader: DataLoader) -> Iterable[dict]:
    for batch in loader:
        yield batch


def _train_one_epoch(fast: FastModel,
                     loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     grad_accum: int,
                     epoch: int,
                     log_every: int = 25) -> dict:
    fast.train()
    losses, accs, n = [], [], 0
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    trainable = [p for p in fast.action_head.parameters()]
    for step, batch in enumerate(_iter_dataset(loader)):
        frame_np = batch["frame"][0].numpy()  # batch size 1
        action_global = batch["action_global"][0].item()
        game = batch["game"][0]
        legal = batch["action_legal_mask"][0]

        logits = fast.predict_action(
            frame_np,
            thought_buffer=None,
            legal_action_mask=legal,
        )
        loss = F.cross_entropy(
            logits.float(),
            torch.tensor([action_global], device=logits.device, dtype=torch.long),
        )
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pred = logits.argmax(dim=-1).item()
            accs.append(int(pred == action_global))
            losses.append(float(loss.item()))
            n += 1

        if (step + 1) % log_every == 0:
            print(f"  [epoch {epoch} step {step+1:5d}/{len(loader)}]  "
                  f"loss={np.mean(losses[-log_every:]):.4f}  "
                  f"acc={np.mean(accs[-log_every:]):.3f}  "
                  f"games={game}  "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

    if (n % grad_accum) != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "mean_loss": float(np.mean(losses)),
        "mean_acc": float(np.mean(accs)),
        "n_samples": n,
        "elapsed_sec": time.time() - t_start,
    }


@torch.no_grad()
def _eval(fast: FastModel, loader: DataLoader) -> dict:
    fast.eval()
    losses, accs, n = [], [], 0
    for batch in _iter_dataset(loader):
        frame_np = batch["frame"][0].numpy()
        action_global = batch["action_global"][0].item()
        legal = batch["action_legal_mask"][0]
        logits = fast.predict_action(frame_np, thought_buffer=None,
                                     legal_action_mask=legal)
        loss = F.cross_entropy(
            logits.float(),
            torch.tensor([action_global], device=logits.device, dtype=torch.long),
        )
        pred = logits.argmax(dim=-1).item()
        accs.append(int(pred == action_global))
        losses.append(float(loss.item()))
        n += 1
    return {
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_acc": float(np.mean(accs)) if accs else 0.0,
        "n_samples": n,
    }


def _collate_keep_meta(batch):
    """Custom collate: stack tensors, keep `game` as a list of strings."""
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True,
                    help="path(s) to trajectory .pt files from collect_trajectories.py")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per-step batch size; MiniCPM-o is memory-hungry, default 1")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="effective batch = batch-size × grad-accum")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/stage_a/action_head.pt")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="limit dataset size for fast iteration (subsample uniformly)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Dataset ----
    print(f"Loading dataset from {args.traj}...")
    ds = ImitationDataset(args.traj)
    print(f"  total samples: {len(ds)}")

    if args.max_samples and args.max_samples < len(ds):
        ds = torch.utils.data.Subset(
            ds, np.random.permutation(len(ds))[:args.max_samples].tolist()
        )
        print(f"  subsampled to: {len(ds)}")

    n_val = int(len(ds) * args.val_fraction)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate_keep_meta,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate_keep_meta,
    )

    # ---- Model ----
    print("Loading FastModel (MiniCPM-o)...")
    fast = FastModel(FastModelConfig()).load_pretrained()

    # Stage A: train ONLY the action_head. Bridge xattn stays zero-init (no-op);
    # it is trained in Stage C. Including bridge xattn at Stage A makes the
    # optimization much harder (71M params, high-variance gradients) and tends to
    # diverge after the action head has warmed up.
    for p in fast.xattn_layers.parameters():
        p.requires_grad = False
    trainable_params = list(fast.action_head.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  trainable parameters: {n_trainable:,} (action_head only; "
          f"bridge xattn frozen at zero-init for Stage A)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    # ---- Train ----
    history = []
    best_val_acc = -1.0
    base, ext = os.path.splitext(args.out)
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} (train, {len(train_loader)} steps)")
        train_stats = _train_one_epoch(
            fast, train_loader, optimizer, args.grad_accum, epoch,
        )
        print(f"  train: loss={train_stats['mean_loss']:.4f}  "
              f"acc={train_stats['mean_acc']:.3f}  "
              f"({train_stats['elapsed_sec']:.0f}s)")

        print(f"Eval (val, {len(val_loader)} samples)...")
        val_stats = _eval(fast, val_loader)
        print(f"  val: loss={val_stats['mean_loss']:.4f}  acc={val_stats['mean_acc']:.3f}")

        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        # Checkpoint per epoch (action_head only — bridge xattn was frozen so no need
        # to persist its zero-init state).
        ep_path = f"{base}_ep{epoch}{ext}"
        torch.save({
            "epoch": epoch,
            "action_head_state": fast.action_head.state_dict(),
            "config": vars(fast.cfg),
            "history": history,
            "args": vars(args),
            "val_acc": val_stats["mean_acc"],
            "val_loss": val_stats["mean_loss"],
        }, ep_path)
        print(f"  wrote {ep_path}")

        # Track the best-val checkpoint at args.out
        if val_stats["mean_acc"] > best_val_acc:
            best_val_acc = val_stats["mean_acc"]
            torch.save({
                "epoch": epoch,
                "action_head_state": fast.action_head.state_dict(),
                "config": vars(fast.cfg),
                "history": history,
                "args": vars(args),
                "val_acc": val_stats["mean_acc"],
                "val_loss": val_stats["mean_loss"],
                "is_best": True,
            }, args.out)
            print(f"  new best val acc {best_val_acc:.3f}; updated {args.out}")

    print(f"\nDone. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
