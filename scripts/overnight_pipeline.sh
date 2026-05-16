#!/bin/bash
# Autonomous overnight pipeline.
#
# 1. (already running) Seaquest T-trajectory collection
# 2. Seaquest Stage C v2
# 3. Seaquest F/T/L head-to-head eval
# 4. MsPacman v2 bandwidth ablation: n_bridge_tokens=4 (Stage C + L eval)
# 5. MsPacman v2 bandwidth ablation: n_bridge_tokens=16 (Stage C + L eval)
# 6. MI diagnostic on v2 MsPacman + v2 Seaquest bridges
#
# Logs in /tmp/overnight_*.log. Each GPU step kills vLLM workers first
# (user explicitly authorized clearing GPU consumers in this session).

set +e

LOG_DIR=/tmp
REPO=/home/ubuntu/latent-bridge-games
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

kill_vllm() {
    local pids
    pids=$(pgrep -f "VLLM::EngineCore" || true)
    if [ -n "$pids" ]; then
        echo "[$(ts)] killing vLLM workers: $pids"
        kill $pids 2>/dev/null || true
        sleep 5
    fi
}

wait_for_seaquest_t_collection() {
    echo "[$(ts)] waiting for Seaquest T-collection to finish..."
    while pgrep -f "run_text_bridge_baseline.*Seaquest" > /dev/null; do
        sleep 60
    done
    echo "[$(ts)] Seaquest T-collection done"
}

echo "[$(ts)] === OVERNIGHT PIPELINE START ==="

# ----------------------------------------------------------------------
wait_for_seaquest_t_collection

# ----------------------------------------------------------------------
echo "[$(ts)] === Step 2: Seaquest Stage C v2 ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2/Seaquest_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/seaquest_sb3dqn.pt \
    --out checkpoints/stage_c/v2_seaquest.pt \
    > $LOG_DIR/overnight_stage_c_seaquest.log 2>&1

# ----------------------------------------------------------------------
echo "[$(ts)] === Step 3: Seaquest F/T/L eval ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games Seaquest --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/seaquest_sb3dqn.pt \
    --bridge-ckpt checkpoints/stage_c/v2_seaquest.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_seaquest.json \
    > $LOG_DIR/overnight_eval_seaquest.log 2>&1

# ----------------------------------------------------------------------
echo "[$(ts)] === Step 4: MsPacman bandwidth ablation N=4 ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LB_BRIDGE_N_TOKENS=4 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2/MsPacman_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --out checkpoints/stage_c/v2_mspacman_n4.pt \
    > $LOG_DIR/overnight_stage_c_mspacman_n4.log 2>&1

kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LB_BRIDGE_N_TOKENS=4 python3 -m src.eval.benchmark \
    --strategies L --games MsPacman --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --bridge-ckpt checkpoints/stage_c/v2_mspacman_n4.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_mspacman_n4.json \
    > $LOG_DIR/overnight_eval_mspacman_n4.log 2>&1

# ----------------------------------------------------------------------
echo "[$(ts)] === Step 5: MsPacman bandwidth ablation N=16 ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LB_BRIDGE_N_TOKENS=16 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2/MsPacman_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --out checkpoints/stage_c/v2_mspacman_n16.pt \
    > $LOG_DIR/overnight_stage_c_mspacman_n16.log 2>&1

kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LB_BRIDGE_N_TOKENS=16 python3 -m src.eval.benchmark \
    --strategies L --games MsPacman --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --bridge-ckpt checkpoints/stage_c/v2_mspacman_n16.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_mspacman_n16.json \
    > $LOG_DIR/overnight_eval_mspacman_n16.log 2>&1

# ----------------------------------------------------------------------
echo "[$(ts)] === Step 6: MI diagnostic ==="
CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
    --trace 'results/t_trajectories_v2/MsPacman_seed*.pt' \
    --bridge-ckpt checkpoints/stage_c/v2_mspacman.pt \
    --out results/mi_diagnostic_mspacman_n8.json \
    > $LOG_DIR/overnight_mi_mspacman_n8.log 2>&1

if [ -f checkpoints/stage_c/v2_mspacman_n4.pt ]; then
    CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
        --trace 'results/t_trajectories_v2/MsPacman_seed*.pt' \
        --bridge-ckpt checkpoints/stage_c/v2_mspacman_n4.pt \
        --out results/mi_diagnostic_mspacman_n4.json \
        > $LOG_DIR/overnight_mi_mspacman_n4.log 2>&1
fi
if [ -f checkpoints/stage_c/v2_mspacman_n16.pt ]; then
    CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
        --trace 'results/t_trajectories_v2/MsPacman_seed*.pt' \
        --bridge-ckpt checkpoints/stage_c/v2_mspacman_n16.pt \
        --out results/mi_diagnostic_mspacman_n16.json \
        > $LOG_DIR/overnight_mi_mspacman_n16.log 2>&1
fi
if [ -f checkpoints/stage_c/v2_seaquest.pt ]; then
    CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
        --trace 'results/t_trajectories_v2/Seaquest_seed*.pt' \
        --bridge-ckpt checkpoints/stage_c/v2_seaquest.pt \
        --out results/mi_diagnostic_seaquest.json \
        > $LOG_DIR/overnight_mi_seaquest.log 2>&1
fi

# ----------------------------------------------------------------------
echo "[$(ts)] === OVERNIGHT PIPELINE COMPLETE ==="
