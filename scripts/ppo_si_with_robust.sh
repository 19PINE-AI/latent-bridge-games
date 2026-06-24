#!/bin/bash
# Stage D PPO on SpaceInvaders, starting from the robust Stage A + Stage C v2.
# Goal: push L from 15 (post-robust-SA) toward F's 107 or higher. Validates
# that PPO can recover an OOD-damaged policy under deployment distribution.

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
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

echo "[$(ts)] === PPO ON SI START ==="

# Smoke test first (16-tick rollout, 1 update — verifies gradients flow)
echo "[$(ts)] === Smoke test ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_d_rl \
    --game SpaceInvaders --total-updates 1 --rollout-len 16 \
    --stage-a-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --stage-c-v2-ckpt checkpoints/stage_c/v2_spaceinvaders_robust.pt \
    --smoke-test \
    --out checkpoints/stage_d/ppo_si_smoke.pt \
    > /tmp/ppo_si_smoke.log 2>&1
SMOKE_RC=$?
echo "[$(ts)] smoke test returned $SMOKE_RC"
if [ "$SMOKE_RC" -ne 0 ]; then
    echo "[$(ts)] === Smoke test FAILED — aborting full PPO ==="
    tail -20 /tmp/ppo_si_smoke.log
    exit 1
fi

# Full run — 20 PPO iterations × 128-tick rollouts, ~6h
echo "[$(ts)] === Full PPO (20 updates, 128-tick rollouts) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_d_rl \
    --game SpaceInvaders --total-updates 20 --rollout-len 128 \
    --stage-a-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --stage-c-v2-ckpt checkpoints/stage_c/v2_spaceinvaders_robust.pt \
    --out checkpoints/stage_d/ppo_si.pt \
    > /tmp/ppo_si_full.log 2>&1

# Eval the trained PPO bridge
echo "[$(ts)] === Eval PPO checkpoint ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games SpaceInvaders --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --bridge-ckpt checkpoints/stage_d/ppo_si.pt \
    --max-slow-tokens 64 \
    --out results/eval_ppo_si.json \
    > /tmp/ppo_si_eval.log 2>&1

echo "[$(ts)] === PPO ON SI COMPLETE ==="
