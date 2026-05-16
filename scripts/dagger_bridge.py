"""DAgger-style on-policy bridge correction.

Fixes the offline-vs-online distribution-shift failure of Stage C: the bridge converges
to KL=0.004 on offline T-rollout frames but underperforms F at deployment because L
visits states T never visited.

Algorithm (iterative):
  Round r:
    1. Roll out L policy for `--episodes-per-round` episodes using current bridge.
       Save (frame, slow_text, slow_vecs) at every L-visited tick — this is now
       L-rollout data, NOT T-rollout.
    2. Train Stage C on the just-collected data for `--epochs-per-round` epochs.
       Teacher is fast model + text suffix; student is fast model + latent buffer.
       KL student||teacher loss, updates bridge xattn only.
    3. Save updated bridge as `bridge_v{r}.pt`.
  Repeat for `--rounds` rounds.

Each round shifts the bridge's training distribution toward L's deployment distribution.
Should close the offline-online gap that breaks plain Stage C.

Single-process design: loads fast + slow models once, keeps them resident across rounds.

Usage:
    HF_HUB_OFFLINE=1 python scripts/dagger_bridge.py --rounds 5 \\
        --episodes-per-round 5 --epochs-per-round 1 \\
        --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \\
        --bridge-init checkpoints/stage_c/c2_mspacman_v1.pt \\
        --out-dir checkpoints/dagger
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.atari_wrapper import AtariEnv
from src.bridge.ring_buffer import ThoughtBuffer
from src.models.fast_model import FastModel, FastModelConfig
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.imitation_data import legal_action_mask, global_to_local_action
from src.training.prompts import build_slow_model_messages


def collect_l_rollouts(fast: FastModel, slow: SlowModel, *,
                       game: str, episodes: int, base_seed: int, max_ticks: int,
                       slow_max_tokens: int = 64) -> list[dict]:
    """Run L policy for N episodes, return per-tick samples in Stage-C-dataset format.

    Each sample: dict(frame, slow_text, slow_vecs, action_global, legal_action_mask).
    The frames are NOT T-rollout — they are visited by L's own policy via the current
    bridge weights.
    """
    legal = legal_action_mask(game)
    bridge_dim = fast.cfg.bridge_dim
    samples: list[dict] = []
    fast.eval()

    for ep in range(episodes):
        seed = base_seed + ep
        thought_buf = ThoughtBuffer(
            capacity=16, dim=bridge_dim, device="cuda", dtype=torch.bfloat16,
        )
        env = AtariEnv(game_name=game, seed=seed)
        obs, _ = env.reset()
        latest_slow_text: str | None = None
        latest_slow_vecs: torch.Tensor | None = None  # CPU float32 [L, D]
        episode_reward = 0.0

        for t in range(max_ticks):
            # L policy: bridge active, argmax action
            with torch.no_grad():
                tb, _ = thought_buf.read(current_time=(t + 1) / 15.0)
                logits = fast.predict_action(
                    obs, thought_buffer=tb.unsqueeze(0),
                    legal_action_mask=legal,
                    slow_text_suffix=None,  # L mode = no text
                )
                global_action = int(logits.argmax(dim=-1).item())
            local_action = global_to_local_action(game, global_action)
            obs, reward, terminated, truncated, text_state = env.step(local_action)
            episode_reward += float(reward)

            # Record THIS frame + currently-active emission for Stage C
            if latest_slow_text is not None and latest_slow_vecs is not None:
                samples.append({
                    "frame": np.array(obs, copy=True),
                    "slow_text": latest_slow_text,
                    "slow_vecs": latest_slow_vecs,
                    "action_global": global_action,
                    "legal_action_mask": legal.clone(),
                })

            # Slow emission on text-state ticks
            if text_state is not None:
                messages = build_slow_model_messages(game, text_state,
                                                     prior_thought=latest_slow_text)
                with torch.no_grad():
                    emit_text, emit_vecs = slow.emit(
                        messages, frame=obs, max_new_tokens=slow_max_tokens,
                    )
                latest_slow_text = emit_text
                latest_slow_vecs = emit_vecs.detach().to("cpu", dtype=torch.float32)
                if emit_vecs.numel() > 0:
                    last_vec = emit_vecs[-1].to(thought_buf.device, dtype=thought_buf.dtype)
                    thought_buf.append(last_vec, timestamp=(t + 1) / 15.0)

            if terminated or truncated:
                break
        env.close()
        print(f"    L ep{ep}: score={episode_reward:.0f} ticks={t+1} samples_so_far={len(samples)}", flush=True)
    return samples


def _build_thought_buffer_for_step(vecs: torch.Tensor, capacity: int = 16,
                                   dim: int = 256, device: str = "cuda") -> torch.Tensor:
    """Materialize a [1, K, D_bridge] thought buffer from a single emission's vectors.
    Take the last `capacity` vectors; pad with zeros if fewer."""
    used = vecs[-capacity:]
    if used.shape[0] < capacity:
        pad = torch.zeros(capacity - used.shape[0], dim, dtype=used.dtype)
        used = torch.cat([pad, used], dim=0)
    return used.unsqueeze(0).to(device, dtype=torch.bfloat16)


def kl_step(fast: FastModel, sample: dict, kl_temp: float = 1.0) -> dict:
    """One DAgger training step. Same as Stage C2 _train_step but inlined here so
    DAgger script is self-contained.
    """
    frame = sample["frame"]
    legal = sample["legal_action_mask"]
    slow_text = sample["slow_text"]
    slow_vecs = sample["slow_vecs"]
    if slow_text is None or slow_vecs is None:
        return {"skipped": True}

    # Teacher
    with torch.no_grad():
        t_logits = fast.predict_action(
            frame, thought_buffer=None, legal_action_mask=legal,
            slow_text_suffix=slow_text,
        ).float()
    # Student (gradient flows)
    tb = _build_thought_buffer_for_step(slow_vecs)
    s_logits = fast.predict_action(
        frame, thought_buffer=tb, legal_action_mask=legal, slow_text_suffix=None,
    ).float()

    legal_idx = legal.nonzero(as_tuple=True)[0].to(s_logits.device)
    s = s_logits.index_select(-1, legal_idx) / kl_temp
    t = t_logits.index_select(-1, legal_idx) / kl_temp
    log_s = F.log_softmax(s, dim=-1)
    log_t = F.log_softmax(t, dim=-1)
    p_t = log_t.exp()
    kl = (p_t * (log_t - log_s)).sum(dim=-1).mean()
    return {"loss": kl, "skipped": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--episodes-per-round", type=int, default=5)
    ap.add_argument("--epochs-per-round", type=int, default=1)
    ap.add_argument("--max-ticks", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--slow-max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--fast-ckpt", required=True,
                    help="Stage A action_head checkpoint")
    ap.add_argument("--bridge-init", default=None,
                    help="Stage C bridge checkpoint to initialize from (optional)")
    ap.add_argument("--out-dir", default="checkpoints/dagger")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- Models ----
    print("Loading FastModel...", flush=True)
    fast = FastModel(FastModelConfig()).load_pretrained()
    if args.fast_ckpt:
        ckpt = torch.load(args.fast_ckpt, map_location="cuda", weights_only=False)
        fast.action_head.load_state_dict(ckpt["action_head_state"])
        print(f"  loaded action_head from {args.fast_ckpt} "
              f"(val_acc={ckpt.get('val_acc', '?')})", flush=True)
    if args.bridge_init:
        ckpt = torch.load(args.bridge_init, map_location="cuda", weights_only=False)
        for k, st in ckpt["xattn_state"].items():
            fast.xattn_layers[k].load_state_dict(st)
        print(f"  loaded bridge xattn from {args.bridge_init}", flush=True)
    # Freeze action_head for DAgger; bridge xattn is trainable
    for p in fast.action_head.parameters():
        p.requires_grad = False
    bridge_params = list(fast.xattn_layers.parameters())
    optimizer = torch.optim.AdamW(bridge_params, lr=args.lr, weight_decay=0.0)
    print(f"  bridge trainable: {sum(p.numel() for p in bridge_params):,}", flush=True)

    print("Loading SlowModel...", flush=True)
    slow = SlowModel(SlowModelConfig()).load_pretrained()

    # ---- DAgger loop ----
    for r in range(1, args.rounds + 1):
        print(f"\n=== ROUND {r}/{args.rounds} ===", flush=True)
        round_start = time.time()

        # 1) Collect L-rollouts
        print(f"  [collect] {args.episodes_per_round} L-rollout episodes...", flush=True)
        t_collect = time.time()
        samples = collect_l_rollouts(
            fast, slow,
            game=args.game,
            episodes=args.episodes_per_round,
            base_seed=args.seed + r * 100,
            max_ticks=args.max_ticks,
            slow_max_tokens=args.slow_max_tokens,
        )
        print(f"  [collect] done: {len(samples)} samples in {time.time() - t_collect:.0f}s", flush=True)

        if not samples:
            print("  [skip] no usable samples this round")
            continue

        # 2) Train bridge on the collected samples
        print(f"  [train] {args.epochs_per_round} epoch(s), {len(samples)} samples", flush=True)
        t_train = time.time()
        fast.train()
        for epoch in range(args.epochs_per_round):
            perm = np.random.permutation(len(samples))
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for step, idx in enumerate(perm):
                out = kl_step(fast, samples[idx])
                if out.get("skipped"):
                    continue
                loss = out["loss"]
                (loss / args.grad_accum).backward()
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(bridge_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.item()))
                if (step + 1) % 200 == 0:
                    print(f"    [r{r} e{epoch+1} step {step+1}/{len(samples)}] "
                          f"KL_recent={np.mean(losses[-200:]):.4f}", flush=True)
            print(f"    [r{r} e{epoch+1}] mean KL={np.mean(losses) if losses else 0:.4f} "
                  f"({time.time() - t_train:.0f}s)", flush=True)

        # 3) Save round checkpoint
        ckpt_path = os.path.join(args.out_dir, f"bridge_round{r}.pt")
        torch.save({
            "round": r,
            "xattn_state": {k: m.state_dict() for k, m in fast.xattn_layers.items()},
            "config": vars(fast.cfg),
            "args": vars(args),
        }, ckpt_path)
        print(f"  [save] {ckpt_path} (round took {time.time() - round_start:.0f}s)", flush=True)

    print("\nDAgger complete.")


if __name__ == "__main__":
    main()
