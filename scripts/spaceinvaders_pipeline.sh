#!/bin/bash
# SpaceInvaders end-to-end pipeline (Tier 2 second data point).
# Runs autonomously: SB3-expert collection → Stage A → T-trajectories →
# Stage C v2 → F/T/L eval.

set +e
LOG_DIR=/tmp
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

echo "[$(ts)] === SPACEINVADERS PIPELINE START ==="

# 1) Download SB3 expert
python3 -c "
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download('sb3/dqn-SpaceInvadersNoFrameskip-v4', 'dqn-SpaceInvadersNoFrameskip-v4.zip')
print(p)
" > $LOG_DIR/si_download.log 2>&1

# 2) SB3-expert trajectory collection (CPU)
echo "[$(ts)] === Step 1: SB3 expert collection ==="
CUDA_VISIBLE_DEVICES="" python3 scripts/collect_trajectories.py \
    --game SpaceInvaders --episodes 10 \
    --expert-policy $(ls /home/ubuntu/.cache/huggingface/hub/models--sb3--dqn-SpaceInvadersNoFrameskip-v4/snapshots/*/dqn-SpaceInvadersNoFrameskip-v4.zip 2>/dev/null | head -1) \
    --epsilon 0.1 --max-ticks 600 \
    --out results/trajectories_SpaceInvaders_sb3_dqn.pt \
    > $LOG_DIR/si_sb3_collect.log 2>&1

# 3) Stage A
echo "[$(ts)] === Step 2: Stage A behavioral cloning ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_SpaceInvaders_sb3_dqn.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --out checkpoints/stage_a/spaceinvaders_sb3dqn.pt \
    > $LOG_DIR/si_stage_a.log 2>&1

# 4) T-trajectories
echo "[$(ts)] === Step 3: T-trajectory collection ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_text_bridge_baseline.py \
    --game SpaceInvaders --episodes 10 --ticks 750 --slow-max-tokens 96 --seed 0 \
    --out-dir results/t_trajectories_v2 \
    > $LOG_DIR/si_t_collect.log 2>&1

# 5) Stage C v2
echo "[$(ts)] === Step 4: Stage C v2 ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2/SpaceInvaders_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/spaceinvaders_sb3dqn.pt \
    --out checkpoints/stage_c/v2_spaceinvaders.pt \
    > $LOG_DIR/si_stage_c.log 2>&1

# 6) Eval F/T/L
echo "[$(ts)] === Step 5: F/T/L eval ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games SpaceInvaders --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/spaceinvaders_sb3dqn.pt \
    --bridge-ckpt checkpoints/stage_c/v2_spaceinvaders.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_spaceinvaders.json \
    > $LOG_DIR/si_eval.log 2>&1

# 7) MI diagnostic
echo "[$(ts)] === Step 6: MI diagnostic ==="
CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
    --trace 'results/t_trajectories_v2/SpaceInvaders_seed*.pt' \
    --bridge-ckpt checkpoints/stage_c/v2_spaceinvaders.pt \
    --out results/mi_diagnostic_spaceinvaders.json \
    > $LOG_DIR/si_mi.log 2>&1

echo "[$(ts)] === SPACEINVADERS PIPELINE COMPLETE ==="
