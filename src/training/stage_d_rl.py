"""Stage D: Online PPO with bridge in the loop.

This is the natural next step after Stage C v2: instead of training the bridge to
match the *text channel's* logits (KL-supervised offline), train it to maximize
*game reward* directly via on-policy RL. PPO is the standard choice.

Design:
  - Trainable params: slow.projection (33M, same as Stage C v2).
    Optionally also fast.action_head (74K) — but mode-collapse risk if both
    move together; the safer setting is to freeze the action head.
  - Rollout policy: L-mode (latent bridge active, no text suffix).
  - Reward: game score delta per fast tick. PPO clipped objective.
  - Value head: small MLP `4096 → 1` on fast model's last hidden state, separate
    from action_head. Trained with MSE against discounted returns.

Why bother after v2 succeeded:
  - v2 KL trained against the text-channel's behavior. The text-channel itself is
    not the optimal policy — it's just a strong reference. PPO can push past T.
  - v2 trains offline on random-rollout frames; PPO trains on its own rollouts
    (no distribution shift).
  - The MI diagnostic shows the bridge IS informative about future reward — PPO
    sharpens that signal.

What this file is for now:
  - A *skeleton* that establishes the env loop, rollout buffer, value head, and
    PPO update structure. Hyperparameters chosen from the standard Atari PPO
    recipe (Schulman et al. 2017 + the SB3 Atari defaults).
  - Not yet run on GPU. The v2 latent bridge already gives L=628 (vs T=408,
    F=256) on MsPacman, so this is upside, not a blocker.

Status: scaffold; full implementation deferred until v2 + bandwidth ablations
+ Seaquest H2 test are in. Marked TODO for the engineering pieces that need
real care.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOHyperparams:
    """Standard Atari PPO recipe (Schulman et al. 2017 + SB3 defaults)."""
    rollout_len: int = 128             # ticks per env per update
    minibatch_size: int = 16
    n_epochs: int = 4                  # update epochs per rollout
    lr: float = 2.5e-4
    gamma: float = 0.99                # discount factor
    gae_lambda: float = 0.95           # GAE λ
    clip_range: float = 0.1            # PPO clipping ε
    value_coef: float = 0.5            # value-loss weight
    entropy_coef: float = 0.01         # entropy bonus
    max_grad_norm: float = 0.5
    total_updates: int = 1000          # outer PPO iterations


class ValueHead(nn.Module):
    """Critic head: fast LLM hidden_dim → scalar value estimate."""
    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, T, D] → [B] value at last position"""
        return self.net(hidden[:, -1, :]).squeeze(-1)


class RolloutBuffer:
    """Standard PPO rollout buffer with GAE-Lambda advantage estimation."""
    def __init__(self, capacity: int, n_actions: int):
        self.capacity = capacity
        self.frames = []     # raw frames (uint8 HWC) for re-running fast model
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.bridge_tokens = []   # [N, hidden_dim] per tick (or None if reused)
        self.legal_masks = np.zeros((capacity, n_actions), dtype=bool)
        self.size = 0

    def add(self, frame, action, log_prob, value, reward, done, bridge_tokens, legal_mask):
        i = self.size
        self.frames.append(frame)
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.bridge_tokens.append(bridge_tokens)
        self.legal_masks[i] = legal_mask
        self.size += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Generalized Advantage Estimation."""
        advantages = np.zeros(self.size, dtype=np.float32)
        last_gae = 0.0
        next_value = last_value
        for t in reversed(range(self.size)):
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            next_value = self.values[t]
        returns = advantages + self.values[:self.size]
        return advantages, returns


def main():
    """Skeleton PPO main loop. Implements env→rollout→update structure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--stage-a-ckpt", required=True)
    ap.add_argument("--stage-c-v2-ckpt", required=True,
                    help="warm-start bridge from v2 Stage C")
    ap.add_argument("--total-updates", type=int, default=100)
    ap.add_argument("--rollout-len", type=int, default=128)
    ap.add_argument("--out", default="checkpoints/stage_d/ppo.pt")
    args = ap.parse_args()

    raise NotImplementedError(
        "Stage D PPO is scaffolded but not yet run. The structure here is correct "
        "(GAE+PPO+clipped objective + value head + bridge-only param updates), "
        "but several pieces need real care before launching:\n"
        "  - Stable LR schedule (PPO is sensitive)\n"
        "  - Reward clipping vs raw — for Atari conventions\n"
        "  - Bridge token recomputation efficiency (every rollout tick reruns slow)\n"
        "  - Vectorized env (currently single env)\n"
        "v2 Stage C already gives L=628 > T=408 on MsPacman; PPO is upside not blocker."
    )


if __name__ == "__main__":
    main()
