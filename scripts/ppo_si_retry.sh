#!/bin/bash
# PPO retry on SI with anti-collapse hyperparameters:
#   entropy_coef 0.01 -> 0.1 (10× higher to prevent entropy crash)
#   action_head LR 2.5e-4 -> 5e-5 (smaller to slow the collapse mode)
#   clip_range 0.1 -> 0.2 (looser, more exploration)

set +e
REPO=/home/ubuntu/latent-bridge-games
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
kill_vllm() {
    local pids=$(pgrep -f "VLLM::EngineCore" || true)
    if [ -n "$pids" ]; then
        echo "[$(ts)] killing vLLM workers: $pids"
        kill $pids 2>/dev/null || true
        sleep 5
    fi
}

echo "[$(ts)] === PPO SI RETRY START ==="

echo "[$(ts)] === PPO (20 updates, entropy_coef=0.1, lr_action_head=5e-5) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_d_rl \
    --game SpaceInvaders --total-updates 20 --rollout-len 128 \
    --stage-a-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --stage-c-v2-ckpt checkpoints/stage_c/v2_spaceinvaders_robust.pt \
    --entropy-coef 0.1 \
    --lr-action-head 5e-5 \
    --clip-range 0.2 \
    --out checkpoints/stage_d/ppo_si_retry.pt \
    > /tmp/ppo_si_retry_full.log 2>&1

echo "[$(ts)] === Eval PPO retry checkpoint ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games SpaceInvaders --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --bridge-ckpt checkpoints/stage_d/ppo_si_retry.pt \
    --max-slow-tokens 64 \
    --out results/eval_ppo_si_retry.json \
    > /tmp/ppo_si_retry_eval.log 2>&1

echo "[$(ts)] === PPO SI RETRY COMPLETE ==="
