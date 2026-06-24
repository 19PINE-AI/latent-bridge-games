#!/bin/bash
# Resume Qbert pipeline from Step 3 (T-collection). Stage A already trained
# (val_acc 33.8%); previous run crashed on a prompt template bug.

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
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

echo "[$(ts)] === QBERT RESUME (skip Steps 1-2) ==="

echo "[$(ts)] === Step 3: T-trajectory collection ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_text_bridge_baseline.py \
    --game Qbert --episodes 10 --ticks 750 --slow-max-tokens 96 --seed 0 \
    --out-dir results/t_trajectories_v2 > /tmp/qbert_t_collect.log 2>&1

echo "[$(ts)] === Step 4: Stage C v2 ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2/Qbert_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/qbert_sb3dqn.pt \
    --out checkpoints/stage_c/v2_qbert.pt > /tmp/qbert_stage_c.log 2>&1

echo "[$(ts)] === Step 5: F/T/L eval ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games Qbert --seeds 0 1 2 --episodes 4 \
    --max-ticks 750 \
    --fast-ckpt checkpoints/stage_a/qbert_sb3dqn.pt \
    --bridge-ckpt checkpoints/stage_c/v2_qbert.pt \
    --max-slow-tokens 64 --out results/eval_v2_qbert.json \
    > /tmp/qbert_eval.log 2>&1

echo "[$(ts)] === Step 6: MI diagnostic ==="
CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
    --trace 'results/t_trajectories_v2/Qbert_seed*.pt' \
    --bridge-ckpt checkpoints/stage_c/v2_qbert.pt \
    --out results/mi_diagnostic_qbert.json > /tmp/qbert_mi.log 2>&1

echo "[$(ts)] === QBERT RESUME COMPLETE ==="
