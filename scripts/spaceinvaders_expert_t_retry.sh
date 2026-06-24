#!/bin/bash
# SpaceInvaders expert-T retry — test the diagnosis from the random-policy failure.
# Hypothesis: L=T=0 on SpaceInvaders is caused by Stage C KL training on random-policy
# T-trajectories under-representing the FIRE action. If we re-collect T-trajectories
# using the SB3-DQN expert (with low epsilon noise), the FIRE action will be properly
# represented in the training distribution, and L should score > 0.

set +e
LOG_DIR=/tmp
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

EXPERT=$(ls ${HOME}/.cache/huggingface/hub/models--sb3--dqn-SpaceInvadersNoFrameskip-v4/snapshots/*/dqn-SpaceInvadersNoFrameskip-v4.zip 2>/dev/null | head -1)
if [ -z "$EXPERT" ]; then
    echo "ERROR: SB3 SpaceInvaders DQN checkpoint not cached. Run the download step first."
    exit 1
fi

echo "[$(ts)] === SPACEINVADERS EXPERT-T RETRY START ==="
echo "[$(ts)] using SB3 expert: $EXPERT"

# 1) T-trajectories with expert policy (low epsilon for exploration)
echo "[$(ts)] === Step 1: T-trajectory collection (expert policy) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_text_bridge_baseline.py \
    --game SpaceInvaders --episodes 10 --ticks 750 --slow-max-tokens 96 --seed 0 \
    --action-policy sb3-expert --expert-policy "$EXPERT" --epsilon 0.1 \
    --out-dir results/t_trajectories_v2_expert \
    > $LOG_DIR/si_t_expert.log 2>&1

# 2) Stage C v2 on the expert T-trajectories
echo "[$(ts)] === Step 2: Stage C v2 (expert T-data) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2_expert/SpaceInvaders_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/spaceinvaders_sb3dqn.pt \
    --out checkpoints/stage_c/v2_spaceinvaders_expert.pt \
    > $LOG_DIR/si_stage_c_expert.log 2>&1

# 3) F/T/L eval against the expert-trained bridge
echo "[$(ts)] === Step 3: F/T/L eval (expert bridge) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games SpaceInvaders --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/spaceinvaders_sb3dqn.pt \
    --bridge-ckpt checkpoints/stage_c/v2_spaceinvaders_expert.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_spaceinvaders_expert.json \
    > $LOG_DIR/si_eval_expert.log 2>&1

# 4) MI diagnostic on expert bridge
echo "[$(ts)] === Step 4: MI diagnostic (expert bridge) ==="
CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
    --trace 'results/t_trajectories_v2_expert/SpaceInvaders_seed*.pt' \
    --bridge-ckpt checkpoints/stage_c/v2_spaceinvaders_expert.pt \
    --out results/mi_diagnostic_spaceinvaders_expert.json \
    > $LOG_DIR/si_mi_expert.log 2>&1

echo "[$(ts)] === SPACEINVADERS EXPERT-T RETRY COMPLETE ==="
