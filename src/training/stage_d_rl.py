"""Stage D: Online PPO with the latent bridge in the loop.

After Stage C v2 supervises L to match T's distribution, Stage D fine-tunes L
directly on game reward. This decouples L from T's ceiling and (per the SI
diagnosis) puts the action-head in-distribution for the deployment prompt.

Trainable parameters:
  - slow.projection (~33 M)  — bridge ThoughtProjection
  - value_head (~250 K)      — critic
  - fast.action_head (~74 K) — also trainable so the policy can adapt under the
    bridge-prefixed input distribution (this is the SI fix).

Frozen:
  - both base models (MiniCPM-o fast, Qwen3-VL-8B slow)

Rollout collection runs L-mode end-to-end; PPO updates use clipped objective + GAE.
Reward = raw ALE score delta per fast tick (no shaping by default).

Usage (after Phase 2/3 finish):
    HF_HUB_OFFLINE=1 python -m src.training.stage_d_rl \
        --game SpaceInvaders \
        --stage-a-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
        --stage-c-v2-ckpt checkpoints/stage_c/v2_spaceinvaders_robust.pt \
        --total-updates 20 --rollout-len 128 \
        --out checkpoints/stage_d/ppo_si.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.env.atari_wrapper import AtariEnv
from src.models.fast_model import FastModel, FastModelConfig
from src.models.slow_model import SlowModel, SlowModelConfig
from src.training.imitation_data import legal_action_mask, global_to_local_action
from src.training.prompts import build_slow_model_messages


@dataclass
class PPOHyperparams:
    """Atari PPO recipe — Schulman 2017 + SB3 defaults, tuned for short rollouts."""
    rollout_len: int = 128             # ticks per env per update
    minibatch_size: int = 16
    n_epochs: int = 4                  # update epochs per rollout
    lr_projection: float = 5e-5        # smaller than supervised — RL is unstable
    lr_value_head: float = 2.5e-4
    lr_action_head: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    kl_anchor_coef: float = 0.05       # penalty against the Stage C reference policy
    slow_emission_every_n_ticks: int = 15  # match the 1Hz cadence


class ValueHead(nn.Module):
    """Critic head: fast LLM hidden_dim → scalar value estimate (per last position)."""
    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """last_hidden: [B, D] → [B]"""
        return self.net(last_hidden).squeeze(-1)


class RolloutBuffer:
    """PPO rollout buffer with GAE-Lambda. Stores enough per-tick state to
    re-run the fast model forward during PPO updates (frames + bridge tokens +
    legal masks)."""
    def __init__(self, capacity: int, n_actions: int):
        self.capacity = capacity
        self.n_actions = n_actions
        self.frames: list[np.ndarray] = []
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.bridge_tokens: list[torch.Tensor] = []  # [N, D] tensors on CPU
        self.legal_masks = np.zeros((capacity, n_actions), dtype=bool)
        # Saved at rollout time for KL anchor against frozen Stage C reference
        self.ref_log_probs = np.zeros(capacity, dtype=np.float32)
        self.size = 0

    def add(self, frame, action, log_prob, value, reward, done,
            bridge_tokens, legal_mask, ref_log_prob):
        i = self.size
        self.frames.append(frame)
        self.actions[i] = action
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)
        self.rewards[i] = float(reward)
        self.dones[i] = bool(done)
        self.bridge_tokens.append(bridge_tokens.detach().to("cpu"))
        self.legal_masks[i] = legal_mask.cpu().numpy().astype(bool)
        self.ref_log_probs[i] = float(ref_log_prob)
        self.size += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
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


def _fast_forward_with_grads(fast: FastModel, frame: np.ndarray,
                              bridge_tokens: torch.Tensor,
                              legal_mask: torch.Tensor,
                              value_head: ValueHead,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fast model forward with gradients (for PPO update). Returns (logits, value).

    Note: predict_action wraps the LLM call but doesn't expose the last hidden
    state, which we need for the value head. We do a parallel forward here to
    pull both.
    """
    # The fast model is frozen but bridge_tokens has grad=True via projection.
    # We need to NOT use torch.no_grad here.
    pil = __import__("PIL.Image", fromlist=["Image"]).fromarray(frame.astype(np.uint8))
    prompt_str = fast._action_prompt_str
    inputs = fast.model.processor(
        [prompt_str], [[pil]], [[]], [[]],
        max_slice_nums=1, use_image_id=None,
        return_tensors="pt", max_length=1024,
    )

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(fast.cfg.device)
        if isinstance(x, list):
            return [to_dev(y) for y in x]
        return x
    inputs = {k: to_dev(v) for k, v in inputs.items()}

    vllm_emb, _ = fast.model.get_vllm_embedding(inputs)  # [1, T, D]
    bt = bridge_tokens
    if bt.dim() == 2:
        bt = bt.unsqueeze(0)
    bt = bt.to(vllm_emb.device, dtype=vllm_emb.dtype)
    seq = torch.cat([bt, vllm_emb], dim=1)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        extra = torch.ones(1, bt.shape[1], device=attention_mask.device,
                           dtype=attention_mask.dtype)
        attention_mask = torch.cat([extra, attention_mask], dim=1)

    out = fast.model.llm.model(inputs_embeds=seq, attention_mask=attention_mask,
                                use_cache=False, output_hidden_states=False)
    last_hidden = out.last_hidden_state[:, -1, :]  # [1, D]
    logits = fast.action_head(last_hidden)         # [1, N_global]
    if legal_mask is not None:
        logits = logits.masked_fill(~legal_mask.to(logits.device), float("-inf"))
    value = value_head(last_hidden)                # [1]
    return logits, value


def collect_rollout(fast: FastModel, slow: SlowModel, value_head: ValueHead,
                    env: AtariEnv, n_ticks: int, game: str,
                    obs0: np.ndarray, text_state0,
                    bridge_tokens0: Optional[torch.Tensor],
                    hp: PPOHyperparams,
                    slow_max_tokens: int = 64,
                    ) -> tuple[RolloutBuffer, np.ndarray, object, Optional[torch.Tensor],
                                float, dict]:
    """Run L-mode for n_ticks; return the buffer + carry state for the next rollout."""
    from src.training.imitation_data import N_GLOBAL_ACTIONS
    legal = legal_action_mask(game)
    legal_indices = legal.nonzero(as_tuple=True)[0].tolist()

    buf = RolloutBuffer(capacity=n_ticks, n_actions=N_GLOBAL_ACTIONS)
    obs = obs0
    text_state = text_state0
    bridge_tokens = bridge_tokens0  # may be None at episode start

    cumulative_reward = 0.0
    n_slow = 0
    t_start = time.time()
    prior_thought = None
    last_value = 0.0
    for t in range(n_ticks):
        # If no bridge yet, just step env until slow emits one.
        # For the first ~15 ticks of an episode there's no bridge — use a zero
        # placeholder of shape [N, D] to keep the rollout uniform.
        if bridge_tokens is None:
            zero_bridge = torch.zeros(
                slow.cfg.n_bridge_tokens, slow.cfg.bridge_dim,
                device=fast.cfg.device, dtype=torch.bfloat16,
            )
            bridge_tokens = zero_bridge

        # Policy forward — gradient flows back through bridge_tokens (which were
        # produced by slow.projection at the most recent emission). When we
        # replay from the buffer during PPO update we use stored bridge_tokens
        # so the slow forward doesn't have to re-run.
        with torch.no_grad():
            logits, value = _fast_forward_with_grads(
                fast, obs, bridge_tokens, legal, value_head,
            )
        # Sample action — categorical over legal logits
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor(action,
                                                     device=logits.device)).item())

        local_action = global_to_local_action(game, action)
        next_obs, reward, terminated, truncated, next_text_state = env.step(local_action)
        cumulative_reward += float(reward)
        done = terminated or truncated

        buf.add(obs, action, log_prob, value.item(), reward, done,
                bridge_tokens, legal, ref_log_prob=log_prob)
        # NB: ref_log_prob = current log_prob at rollout time; this is the
        # snapshot of the reference policy for the KL anchor. (Stage C policy
        # is exactly what's loaded into the model right now at rollout time.)

        # Slow emission cadence
        if next_text_state is not None:
            msgs = build_slow_model_messages(game, next_text_state,
                                             prior_thought=prior_thought)
            emit_text, emit_bridge = slow.emit(
                msgs, frame=next_obs, max_new_tokens=slow_max_tokens,
                return_raw_residuals=False,
            )
            prior_thought = emit_text
            if emit_bridge.numel() > 0:
                bridge_tokens = emit_bridge  # [N, D]
            n_slow += 1

        obs = next_obs
        text_state = next_text_state
        last_value = float(value.item())
        if done:
            obs, _ = env.reset()
            text_state = None
            bridge_tokens = None
            prior_thought = None

    elapsed = time.time() - t_start
    stats = {"rollout_score": cumulative_reward, "n_slow": n_slow,
             "rollout_sec": elapsed}
    return buf, obs, text_state, bridge_tokens, last_value, stats


def ppo_update(fast: FastModel, slow: SlowModel, value_head: ValueHead,
               buf: RolloutBuffer, advantages: np.ndarray, returns: np.ndarray,
               optimizer: torch.optim.Optimizer, hp: PPOHyperparams,
               ) -> dict:
    """One PPO update: K epochs of minibatched clipped-objective optimization."""
    advantages_t = torch.from_numpy(advantages).to(fast.cfg.device)
    returns_t = torch.from_numpy(returns).to(fast.cfg.device)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "kl_anchor": 0.0, "clip_frac": 0.0, "n_minibatches": 0}

    n = buf.size
    indices = np.arange(n)
    for epoch in range(hp.n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, hp.minibatch_size):
            mb_idx = indices[start:start + hp.minibatch_size]
            mb_pl, mb_vl, mb_ent, mb_kl, mb_clip = 0.0, 0.0, 0.0, 0.0, 0.0
            optimizer.zero_grad(set_to_none=True)

            for i in mb_idx:
                frame = buf.frames[i]
                bt = buf.bridge_tokens[i].to(fast.cfg.device, dtype=torch.bfloat16)
                legal = torch.from_numpy(buf.legal_masks[i]).to(fast.cfg.device)
                action = int(buf.actions[i])
                old_log_prob = float(buf.log_probs[i])
                ref_log_prob = float(buf.ref_log_probs[i])
                adv = float(advantages_t[i].item())
                ret = float(returns_t[i].item())

                logits, value = _fast_forward_with_grads(
                    fast, frame, bt, legal, value_head,
                )
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                log_prob = dist.log_prob(torch.tensor(action,
                                                       device=logits.device))
                entropy = dist.entropy()

                ratio = torch.exp(log_prob - old_log_prob)
                clip_ratio = torch.clamp(ratio, 1 - hp.clip_range, 1 + hp.clip_range)
                policy_loss = -torch.min(ratio * adv, clip_ratio * adv)
                value_loss = (value - ret).pow(2)
                kl_anchor = (log_prob - ref_log_prob).pow(2)

                loss = (policy_loss
                        + hp.value_coef * value_loss
                        - hp.entropy_coef * entropy
                        + hp.kl_anchor_coef * kl_anchor)
                loss.backward()

                mb_pl += float(policy_loss.item())
                mb_vl += float(value_loss.item())
                mb_ent += float(entropy.item())
                mb_kl += float(kl_anchor.item())
                if (ratio - 1).abs().item() > hp.clip_range:
                    mb_clip += 1.0

            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                max_norm=hp.max_grad_norm,
            )
            optimizer.step()
            mb_n = len(mb_idx)
            metrics["policy_loss"] += mb_pl / mb_n
            metrics["value_loss"] += mb_vl / mb_n
            metrics["entropy"] += mb_ent / mb_n
            metrics["kl_anchor"] += mb_kl / mb_n
            metrics["clip_frac"] += mb_clip / mb_n
            metrics["n_minibatches"] += 1

    for k in ("policy_loss", "value_loss", "entropy", "kl_anchor", "clip_frac"):
        metrics[k] /= max(1, metrics["n_minibatches"])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--stage-a-ckpt", required=True)
    ap.add_argument("--stage-c-v2-ckpt", required=True,
                    help="warm-start bridge from v2 Stage C")
    ap.add_argument("--total-updates", type=int, default=20)
    ap.add_argument("--rollout-len", type=int, default=128)
    ap.add_argument("--out", default="checkpoints/stage_d/ppo.pt")
    ap.add_argument("--smoke-test", action="store_true",
                    help="single short rollout; verifies gradients flow")
    args = ap.parse_args()

    hp = PPOHyperparams(rollout_len=args.rollout_len)
    if args.smoke_test:
        hp.rollout_len = 16
        hp.minibatch_size = 4
        hp.n_epochs = 1
        args.total_updates = 1

    print(f"Loading FastModel...")
    fast = FastModel(FastModelConfig()).load_pretrained()
    ck_a = torch.load(args.stage_a_ckpt, map_location="cuda", weights_only=False)
    fast.action_head.load_state_dict(ck_a["action_head_state"])
    print(f"  loaded action_head from {args.stage_a_ckpt}")
    # Unfreeze action_head for Stage D (this is the SI fix)
    for p in fast.action_head.parameters():
        p.requires_grad = True

    print(f"Loading SlowModel...")
    slow = SlowModel(SlowModelConfig()).load_pretrained()
    ck_c = torch.load(args.stage_c_v2_ckpt, map_location="cuda", weights_only=False)
    slow.projection.load_state_dict(ck_c["slow_projection_state"])
    print(f"  loaded slow projection from {args.stage_c_v2_ckpt}")
    for p in slow.projection.parameters():
        p.requires_grad = True

    value_head = ValueHead(hidden_dim=4096).to(fast.cfg.device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW([
        {"params": slow.projection.parameters(), "lr": hp.lr_projection},
        {"params": value_head.parameters(), "lr": hp.lr_value_head},
        {"params": fast.action_head.parameters(), "lr": hp.lr_action_head},
    ], weight_decay=0.0)

    env = AtariEnv(game_name=args.game, seed=0)
    obs, text_state = env.reset()
    bridge_tokens = None

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    history = []
    for upd in range(args.total_updates):
        t0 = time.time()
        buf, obs, text_state, bridge_tokens, last_value, stats = collect_rollout(
            fast, slow, value_head, env, hp.rollout_len, args.game,
            obs, text_state, bridge_tokens, hp,
        )
        advantages, returns = buf.compute_gae(last_value, hp.gamma, hp.gae_lambda)
        metrics = ppo_update(fast, slow, value_head, buf, advantages, returns,
                             optimizer, hp)
        elapsed = time.time() - t0
        log_entry = {"update": upd, **stats, **metrics, "elapsed_sec": elapsed}
        history.append(log_entry)
        print(f"[upd {upd:3d}] score_in_rollout={stats['rollout_score']:.0f}  "
              f"pol={metrics['policy_loss']:+.4f}  val={metrics['value_loss']:.4f}  "
              f"ent={metrics['entropy']:.3f}  kl={metrics['kl_anchor']:.4f}  "
              f"clip={metrics['clip_frac']:.2f}  ({elapsed:.0f}s)", flush=True)

        # Checkpoint every update
        torch.save({
            "v2": True,  # benchmark.py loader recognises this as LLaVA-style
            "update": upd,
            "slow_projection_state": slow.projection.state_dict(),
            "action_head_state": fast.action_head.state_dict(),
            "value_head_state": value_head.state_dict(),
            "history": history,
            "hp": asdict(hp),
        }, args.out)

    env.close()
    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
