#!/bin/bash
# Enduro end-to-end pipeline. RoadRunner-like profile: continuous directional
# context + long-horizon (daily car quota) + scrolling environment. Predicted
# L > T > F if F has trouble committing to a directional strategy.

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

EXPERT_REPO="qgallouedec/ppo-EnduroNoFrameskip-v4-3540983129"
EXPERT_FILE="ppo-EnduroNoFrameskip-v4.zip"
EXPERT=$(ls /home/ubuntu/.cache/huggingface/hub/models--qgallouedec--ppo-EnduroNoFrameskip-v4-3540983129/snapshots/*/$EXPERT_FILE 2>/dev/null | head -1)
if [ -z "$EXPERT" ]; then
    echo "[$(ts)] downloading Enduro expert..."
    python3 -c "from huggingface_hub import hf_hub_download; \
                p=hf_hub_download('$EXPERT_REPO', '$EXPERT_FILE'); print(p)" \
                > /tmp/enduro_dl.log 2>&1
    EXPERT=$(ls /home/ubuntu/.cache/huggingface/hub/models--qgallouedec--ppo-EnduroNoFrameskip-v4-3540983129/snapshots/*/$EXPERT_FILE | head -1)
fi
echo "[$(ts)] expert: $EXPERT"

echo "[$(ts)] === ENDURO PIPELINE START ==="
echo "[$(ts)] === Step 1: SB3 expert collection ==="
CUDA_VISIBLE_DEVICES="" python3 scripts/collect_trajectories.py \
    --game Enduro --episodes 10 --expert-policy "$EXPERT" --epsilon 0.1 \
    --max-ticks 3000 --out results/trajectories_Enduro_ppo.pt \
    > /tmp/enduro_sb3_collect.log 2>&1

echo "[$(ts)] === Step 2: Stage A ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_Enduro_ppo.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --out checkpoints/stage_a/enduro_ppo.pt > /tmp/enduro_stage_a.log 2>&1

echo "[$(ts)] === Step 3: T-trajectory collection ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_text_bridge_baseline.py \
    --game Enduro --episodes 10 --ticks 750 --slow-max-tokens 96 --seed 0 \
    --out-dir results/t_trajectories_v2 > /tmp/enduro_t_collect.log 2>&1

echo "[$(ts)] === Step 4: Stage C v2 ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2/Enduro_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/enduro_ppo.pt \
    --out checkpoints/stage_c/v2_enduro.pt > /tmp/enduro_stage_c.log 2>&1

echo "[$(ts)] === Step 5: F/T/L eval ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games Enduro --seeds 0 1 2 --episodes 4 \
    --max-ticks 750 \
    --fast-ckpt checkpoints/stage_a/enduro_ppo.pt \
    --bridge-ckpt checkpoints/stage_c/v2_enduro.pt \
    --max-slow-tokens 64 --out results/eval_v2_enduro.json \
    > /tmp/enduro_eval.log 2>&1

echo "[$(ts)] === Step 6: MI diagnostic ==="
CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
    --trace 'results/t_trajectories_v2/Enduro_seed*.pt' \
    --bridge-ckpt checkpoints/stage_c/v2_enduro.pt \
    --out results/mi_diagnostic_enduro.json > /tmp/enduro_mi.log 2>&1

echo "[$(ts)] === ENDURO PIPELINE COMPLETE ==="
