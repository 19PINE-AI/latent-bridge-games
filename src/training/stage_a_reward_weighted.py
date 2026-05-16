"""Per-game Stage A action_head training via reward-weighted regression.

When there's no expert (e.g., Frostbite has no SB3 zoo checkpoint), train the action
head from existing T-condition trajectory data using a reward-weighted regression /
"upside-down RL" approach: weight each (frame, action) example by how much *future*
reward was earned after that decision.

Algorithm:
  For each tick t in each episode:
    - target = sum_{t'≥t} gamma^(t'-t) * reward_{t'}     (discounted future return)
    - weight = clip(target, 0, max_weight)               (treat positive returns as good)
  Train action_head with cross-entropy on `action_t` weighted by `weight`.

Notes:
  - Frames with target=0 (no future reward) contribute nothing.
  - This is NOT optimal RL — it's a cheap warm-start that learns "what actions
    correlated with scoring", regardless of the noise in the rest of the trajectory.
  - It's NOT self-distillation either (that would require model outputs as labels).
  - For games where T trajectories had any reward (Frostbite: mean 55, MsPacman: 276),
    this should produce a head better than zero-init.

Usage:
    HF_HUB_OFFLINE=1 python -m src.training.stage_a_reward_weighted \\
        --trace results/t_trajectories/Frostbite_seed*.pt \\
        --game Frostbite --gamma 0.99 --max-weight 100 --epochs 1 \\
        --out checkpoints/stage_a/frostbite_rwr.pt
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
from src.training.imitation_data import (
    legal_action_mask, GAME_ACTION_TO_GLOBAL, N_GLOBAL_ACTIONS,
)


def build_rwr_dataset(trace_paths: list[str], game: str, gamma: float = 0.99,
                      max_weight: float = 100.0) -> list[dict]:
    """Walk T trajectories; emit (frame, action_global, weight) per tick where weight
    is the clipped discounted future return."""
    if game not in GAME_ACTION_TO_GLOBAL:
        raise ValueError(f"unknown game {game!r}")
    local_to_global = GAME_ACTION_TO_GLOBAL[game]

    samples = []
    n_total = 0
    n_kept = 0
    n_positive_episodes = 0
    for p in trace_paths:
        blob = torch.load(p, weights_only=False)
        if blob.get("game") != game:
            continue
        trace = blob["trace"]
        # Compute discounted future return
        rewards = np.array([t["reward"] for t in trace], dtype=np.float32)
        T = len(rewards)
        future = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + gamma * running
            future[t] = running
        if future.max() > 0:
            n_positive_episodes += 1
        for t in range(T):
            n_total += 1
            w = float(min(max(future[t], 0.0), max_weight))
            if w <= 0:
                continue  # zero-weight sample contributes nothing
            local_action = int(trace[t]["action"])
            # action recorded in T trajectory uses GLOBAL space already
            # (we already mapped globally in run_text_bridge_baseline.py)
            samples.append({
                "frame": np.array(trace[t]["obs"], copy=True),
                "action_global": local_action,  # already global
                "weight": w,
            })
            n_kept += 1
    print(f"  built {n_kept}/{n_total} samples "
          f"(positive-reward episodes: {n_positive_episodes}/{len(trace_paths)})")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", nargs="+", required=True,
                    help="path(s) to T-condition trajectory .pt files")
    ap.add_argument("--game", required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max-weight", type=float, default=100.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/stage_a/rwr.pt")
    args = ap.parse_args()

    # Expand globs
    traces = []
    for pat in args.trace:
        matches = sorted(glob.glob(pat)) or [pat]
        traces.extend(matches)
    print(f"loading {len(traces)} trace file(s) for {args.game}...")
    samples = build_rwr_dataset(traces, args.game, gamma=args.gamma,
                                max_weight=args.max_weight)
    if not samples:
        print("ERROR: no positive-reward samples found. Cannot do reward-weighted "
              "regression on this trajectory set.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * args.val_fraction))
    val_idx = set(perm[:n_val].tolist())
    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]
    print(f"  train: {len(train_samples)}, val: {len(val_samples)}")

    print("Loading FastModel...")
    fast = FastModel(FastModelConfig()).load_pretrained()
    for p in fast.xattn_layers.parameters():
        p.requires_grad = False
    trainable = list(fast.action_head.parameters())
    print(f"  trainable: {sum(p.numel() for p in trainable):,} (action_head only)")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    legal = legal_action_mask(args.game)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Shuffle train samples per epoch
        train_perm = rng.permutation(len(train_samples))
        fast.train()
        losses, weights_used = [], []
        t_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, idx in enumerate(train_perm):
            s = train_samples[idx]
            frame = s["frame"]
            action_global = int(s["action_global"])
            weight = float(s["weight"])

            logits = fast.predict_action(frame, thought_buffer=None,
                                         legal_action_mask=legal)
            ce = F.cross_entropy(
                logits.float(),
                torch.tensor([action_global], device=logits.device, dtype=torch.long),
            )
            loss = ce * weight  # reward-weighted

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(ce.item()))
            weights_used.append(weight)
            if (step + 1) % 100 == 0:
                print(f"  [ep {epoch} step {step+1}/{len(train_perm)}] "
                      f"CE={np.mean(losses[-100:]):.4f} "
                      f"w_mean={np.mean(weights_used[-100:]):.1f} "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)

        # Validation
        fast.eval()
        val_losses, val_correct, val_n = [], 0, 0
        with torch.no_grad():
            for s in val_samples:
                logits = fast.predict_action(s["frame"], thought_buffer=None,
                                             legal_action_mask=legal)
                target = torch.tensor([int(s["action_global"])],
                                      device=logits.device, dtype=torch.long)
                ce = F.cross_entropy(logits.float(), target).item()
                val_losses.append(ce)
                if int(logits.argmax(dim=-1).item()) == int(s["action_global"]):
                    val_correct += 1
                val_n += 1
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = val_correct / max(1, val_n)
        print(f"  [ep {epoch}] train CE={np.mean(losses):.4f}  "
              f"val CE={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"({time.time()-t_start:.0f}s)", flush=True)

        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "action_head_state": fast.action_head.state_dict(),
                "config": vars(fast.cfg),
                "args": vars(args),
                "history": history,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, args.out)
            print(f"  new best val_loss {val_loss:.4f}; updated {args.out}", flush=True)

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
