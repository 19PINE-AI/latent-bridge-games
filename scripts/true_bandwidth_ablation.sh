#!/bin/bash
# TRUE bandwidth ablation: re-collect T-trajectories at N=4 and N=16,
# train Stage C on each, eval each. Compare to N=8 baseline.
#
# Earlier "ablation" only varied deployment-time N (training used cached
# N=8 trajectories regardless). This script varies BOTH training-time and
# deployment-time N consistently.

set +e
LOG_DIR=/tmp
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
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

echo "[$(ts)] === TRUE BANDWIDTH ABLATION START ==="

# Each N gets fresh T-trajectories saved with that N
for N in 4 16; do
    echo "[$(ts)] === True bandwidth ablation N=$N ==="

    # 1) Collect T-trajectories with N=$N residuals saved per emission
    kill_vllm
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LB_BRIDGE_N_TOKENS=$N python3 \
        scripts/run_text_bridge_baseline.py \
        --game MsPacman --episodes 10 --ticks 750 --slow-max-tokens 96 --seed 0 \
        --out-dir results/t_trajectories_v2_n${N} \
        > $LOG_DIR/true_ablation_collect_n${N}.log 2>&1

    # 2) Train Stage C v2 on these N=$N trajectories
    kill_vllm
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LB_BRIDGE_N_TOKENS=$N python3 \
        -m src.training.stage_c_v2 \
        --trace "results/t_trajectories_v2_n${N}/MsPacman_seed*.pt" \
        --epochs 1 --grad-accum 4 --lr 5e-5 \
        --stage-a-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
        --out checkpoints/stage_c/v2_mspacman_true_n${N}.pt \
        > $LOG_DIR/true_ablation_stage_c_n${N}.log 2>&1

    # 3) Eval (L only — F/T are baseline-independent of N)
    kill_vllm
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LB_BRIDGE_N_TOKENS=$N python3 \
        -m src.eval.benchmark \
        --strategies L --games MsPacman --seeds 0 1 2 --episodes 4 \
        --max-ticks 500 \
        --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
        --bridge-ckpt checkpoints/stage_c/v2_mspacman_true_n${N}.pt \
        --max-slow-tokens 64 \
        --out results/eval_v2_mspacman_true_n${N}.json \
        > $LOG_DIR/true_ablation_eval_n${N}.log 2>&1
done

echo "[$(ts)] === TRUE BANDWIDTH ABLATION COMPLETE ==="
