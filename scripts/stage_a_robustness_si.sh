#!/bin/bash
# Stage A robustness ablation for SpaceInvaders.
# Retrains the action head with --suffix-prob=0.5 (half of training samples get a
# fake slow-text suffix). If the OOD-brittleness diagnosis is correct, this should
# make T and L score > 0 on SpaceInvaders.

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

echo "[$(ts)] === STAGE A ROBUSTNESS (SI) START ==="

# 1) Re-train Stage A with suffix augmentation
echo "[$(ts)] === Step 1: Stage A retrain with --suffix-prob=0.5 ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_SpaceInvaders_sb3_dqn.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --suffix-prob 0.5 \
    --suffix-from-trace results/t_trajectories_v2_expert/SpaceInvaders_seed0.pt \
    --out checkpoints/stage_a/spaceinvaders_robust.pt \
    > /tmp/si_robust_stage_a.log 2>&1

# 2) Re-train Stage C v2 against the new robust action head
#    (using the existing expert T-trajectories which already gave +0.024 MI)
echo "[$(ts)] === Step 2: Stage C v2 against robust action head ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2_expert/SpaceInvaders_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --out checkpoints/stage_c/v2_spaceinvaders_robust.pt \
    > /tmp/si_robust_stage_c.log 2>&1

# 3) F/T/L eval with the robust checkpoints
echo "[$(ts)] === Step 3: F/T/L eval (robust head + bridge) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games SpaceInvaders --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --bridge-ckpt checkpoints/stage_c/v2_spaceinvaders_robust.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_spaceinvaders_robust.json \
    > /tmp/si_robust_eval.log 2>&1

echo "[$(ts)] === STAGE A ROBUSTNESS (SI) COMPLETE ==="
