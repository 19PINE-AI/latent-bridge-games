"""Collect trajectories from ALE games using either a random policy or an SB3 expert.

Writes a .pt file containing a list of episodes. Each episode is a dict of arrays:
  obs       : uint8 [T, 210, 160, 3]   raw RGB at 15Hz tick rate (downsampled from 60Hz)
  action    : int8  [T]                action taken at each tick (in the game's local action space)
  reward    : float32 [T]              per-tick reward (summed over 4 frames)
  done      : bool [T]                 True on terminal tick
  text_idx  : int32 [T_text]           tick indices at which text_state was emitted (1 / 15 ticks)
  text_state: list[dict] (length T_text)  per-emission text-state dict

Used downstream by:
  - Stage A behavioral cloning (obs, action) pairs
  - Stage C bridge supervision: feed text_state into slow model offline to generate
    text emissions, then construct (fast-ctx, slow-text, action) tuples.

Usage:
    # Random-policy data (cheap, low quality):
    python scripts/collect_trajectories.py --game MsPacman --episodes 50 --epsilon 1.0
    # SB3 expert data (requires a pretrained DQN/PPO checkpoint):
    python scripts/collect_trajectories.py --game MsPacman --episodes 50 \\
        --expert-policy /path/to/dqn_mspacman.zip --epsilon 0.05
"""
from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.atari_wrapper import AtariEnv


def _load_sb3_policy(policy_path: str):
    """Load a Stable-Baselines3 policy from a .zip checkpoint.

    The returned object exposes `.predict(obs, deterministic=True)`. SB3 standard Atari
    policies expect grayscale 84×84 frames stacked into a 4-channel input, so we also
    build the standard AtariPreprocessing wrapper for the policy's view of the env.
    """
    from stable_baselines3 import DQN, PPO
    # Try DQN first, fall back to PPO.
    for cls in (DQN, PPO):
        try:
            return cls.load(policy_path, device="auto")
        except Exception:
            continue
    raise ValueError(f"could not load SB3 policy from {policy_path}")


def _build_sb3_view_env(game: str, seed: int):
    """Build the *companion* gym env that the SB3 policy sees: standard Atari preprocessing
    (grayscale, 84x84, frame stack of 4). This env is stepped in lockstep with our
    AtariEnv so the policy sees what it was trained on while we record the raw RGB.
    """
    import gymnasium as gym
    import ale_py  # noqa: F401
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    env = make_atari_env(f"ALE/{game}-v5", n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


def collect_episode(env: AtariEnv, epsilon: float, rng: np.random.Generator,
                    max_ticks: int, sb3_policy=None, sb3_view_env=None) -> dict:
    obs, text0 = env.reset()
    sb3_obs = sb3_view_env.reset() if sb3_view_env is not None else None
    obses, actions, rewards, dones = [], [], [], []
    text_idx, text_states = [0], [text0]
    n_actions = env.action_space_size
    prev_action = 0
    for t in range(max_ticks):
        # Action selection.
        if sb3_policy is not None and rng.random() >= epsilon:
            action_pred, _ = sb3_policy.predict(sb3_obs, deterministic=True)
            action = int(action_pred[0])
        elif rng.random() < epsilon or sb3_policy is None:
            action = int(rng.integers(0, n_actions))
        else:
            action = prev_action
        prev_action = action

        next_obs, reward, terminated, truncated, text_state = env.step(action)
        if sb3_view_env is not None:
            # SB3's vec env uses frame-skip of its own; step it once with the same action.
            sb3_obs, _, sb3_done, _ = sb3_view_env.step(np.array([action]))
            if bool(sb3_done[0]):
                sb3_obs = sb3_view_env.reset()
        obses.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        if text_state is not None:
            text_idx.append(t + 1)
            text_states.append(text_state)
        obs = next_obs
        if terminated or truncated:
            break
    return {
        "obs": np.stack(obses, axis=0).astype(np.uint8),
        "action": np.array(actions, dtype=np.int8),
        "reward": np.array(rewards, dtype=np.float32),
        "done": np.array(dones, dtype=bool),
        "text_idx": np.array(text_idx, dtype=np.int32),
        "text_state": [vars(ts) if ts is not None else None for ts in text_states],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="prob of random action; 1.0 = pure random. With --expert-policy, "
                         "this is the exploration rate around the expert.")
    ap.add_argument("--expert-policy", default=None,
                    help="path to an SB3 .zip checkpoint (DQN or PPO). If provided, the "
                         "expert is queried with prob (1-epsilon).")
    ap.add_argument("--max-ticks", type=int, default=6750,  # 27000 / 4 = 15Hz ticks for 30 min
                    help="max 15Hz ticks per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"results/trajectories_{args.game}_eps{args.epsilon}.pt"
    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)

    sb3_policy = _load_sb3_policy(args.expert_policy) if args.expert_policy else None

    episodes = []
    t0 = time.time()
    for ep in range(args.episodes):
        env = AtariEnv(game_name=args.game, seed=args.seed + ep)
        rng = np.random.default_rng(args.seed + ep)
        sb3_view_env = _build_sb3_view_env(args.game, args.seed + ep) if sb3_policy else None
        episode = collect_episode(env, args.epsilon, rng, args.max_ticks,
                                  sb3_policy=sb3_policy, sb3_view_env=sb3_view_env)
        episodes.append(episode)
        env.close()
        if sb3_view_env is not None:
            sb3_view_env.close()
        if (ep + 1) % 5 == 0 or ep + 1 == args.episodes:
            total_ticks = sum(len(e["action"]) for e in episodes)
            total_reward = sum(float(e["reward"].sum()) for e in episodes) / len(episodes)
            elapsed = time.time() - t0
            print(f"[collect] {args.game} ep {ep+1}/{args.episodes}  "
                  f"avg_reward={total_reward:.1f}  total_ticks={total_ticks}  "
                  f"elapsed={elapsed:.1f}s", flush=True)

    torch.save(
        {
            "game": args.game,
            "epsilon": args.epsilon,
            "expert_policy": args.expert_policy,
            "episodes": episodes,
        },
        out,
    )
    print(f"[collect] wrote {out} ({os.path.getsize(out)/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
