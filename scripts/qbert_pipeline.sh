#!/bin/bash
# Qbert end-to-end pipeline. Isometric platformer with planned tile-jump
# traversal. Slow can plan sequence; fast handles jump timing.

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

EXPERT_REPO="sb3/dqn-QbertNoFrameskip-v4"
EXPERT_FILE="dqn-QbertNoFrameskip-v4.zip"
EXPERT=$(ls ${HOME}/.cache/huggingface/hub/models--sb3--dqn-QbertNoFrameskip-v4/snapshots/*/$EXPERT_FILE 2>/dev/null | head -1)
if [ -z "$EXPERT" ]; then
    echo "[$(ts)] downloading Qbert expert..."
    python3 -c "from huggingface_hub import hf_hub_download; \
                p=hf_hub_download('$EXPERT_REPO', '$EXPERT_FILE'); print(p)" \
                > /tmp/qbert_dl.log 2>&1
    EXPERT=$(ls ${HOME}/.cache/huggingface/hub/models--sb3--dqn-QbertNoFrameskip-v4/snapshots/*/$EXPERT_FILE | head -1)
fi
echo "[$(ts)] expert: $EXPERT"

echo "[$(ts)] === QBERT PIPELINE START ==="
echo "[$(ts)] === Step 1: SB3 expert collection ==="
CUDA_VISIBLE_DEVICES="" python3 scripts/collect_trajectories.py \
    --game Qbert --episodes 10 --expert-policy "$EXPERT" --epsilon 0.1 \
    --max-ticks 2500 --out results/trajectories_Qbert_sb3_dqn.pt \
    > /tmp/qbert_sb3_collect.log 2>&1

echo "[$(ts)] === Step 2: Stage A ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_Qbert_sb3_dqn.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --out checkpoints/stage_a/qbert_sb3dqn.pt > /tmp/qbert_stage_a.log 2>&1

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

echo "[$(ts)] === QBERT PIPELINE COMPLETE ==="
