"""Train a fresh PPO expert on MetaDrive (native 259-d state obs, continuous action).

Goal: a genuinely strong driving expert (route completion, low crash) to serve as the
Stage A teacher. The fast/slow models still use the top-down image wrapper; this expert
is mapped to the discrete action grid by collect_metadrive_expert at Stage A time.

Trains on the FAST native env (no rendering) so PPO is sample-efficient. Saves to
checkpoints/md_ppo/ and writes an honest eval (mean reward, route completion, crash rate)
to results/md_ppo_eval.json so we can GATE: only use it if it clearly beats the
built-in expert (~222) and the BC fast model F (~74).
"""
from __future__ import annotations
import os, sys, json, argparse, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def make_env(seed: int):
    import gymnasium as gym
    from metadrive.envs import MetaDriveEnv
    cfg = dict(use_render=False, num_scenarios=1000, start_seed=seed,
               horizon=1000, traffic_density=0.2, map=3,
               success_reward=10.0, out_of_road_penalty=5.0,
               crash_vehicle_penalty=5.0, driving_reward=1.0, speed_reward=0.5)
    env = MetaDriveEnv(cfg)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/md_ppo/ppo_metadrive")
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def env_fn(rank):
        def _f():
            return make_env(seed=rank * 1000)
        return _f

    venv = SubprocVecEnv([env_fn(i) for i in range(args.n_envs)])
    model = PPO("MlpPolicy", venv, verbose=1, n_steps=1024, batch_size=512,
                gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.0,
                learning_rate=3e-4, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]), device="cuda")
    ckpt_cb = CheckpointCallback(save_freq=max(1, 100_000 // args.n_envs),
                                 save_path=os.path.dirname(args.out),
                                 name_prefix="ppo_md")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=ckpt_cb, progress_bar=False)
    model.save(args.out)
    train_sec = time.time() - t0

    # ---- honest eval on held-out seeds ----
    eval_env = make_env(seed=900_000)
    rewards, completions, crashes, arrived = [], [], [], 0
    for ep in range(args.eval_episodes):
        obs, info = eval_env.reset(seed=900_000 + ep)
        tot = 0.0
        for t in range(1000):
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(act)
            tot += r
            if term or trunc:
                break
        rewards.append(tot)
        completions.append(float(info.get("route_completion", 0.0)))
        crashes.append(1.0 if info.get("crash") else 0.0)
        arrived += 1 if info.get("arrive_dest") else 0
    eval_env.close()
    venv.close()
    out = {
        "train_timesteps": args.timesteps, "train_sec": round(train_sec, 1),
        "eval_episodes": args.eval_episodes,
        "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        "mean_route_completion": float(np.mean(completions)),
        "crash_rate": float(np.mean(crashes)),
        "arrive_rate": arrived / args.eval_episodes,
        "ckpt": args.out + ".zip",
        "reference_builtin_expert_reward": 222.5,
        "reference_F_fast_model": 73.7,
    }
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/md_ppo_eval.json", "w"), indent=2)
    print("EVAL", json.dumps(out))


if __name__ == "__main__":
    main()
