#!/bin/bash
# Scaling ablation: replace 8B-VL slow model with 30B-A3B-VL-FP8 and re-run T/L
# on MsPacman to test whether the latent advantage grows with slow-model capacity.
#
# Hypothesis (the bandwidth claim from docs/01_framing.md): if text serialization
# is the bottleneck, a bigger slow model has more reasoning per emission that
# text can't capture but the latent channel can. Predicted: L - T gap grows.
#
# Pre-req: Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 must already be cached locally.

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

# Tell slow_model.py to use the 30B-FP8 config via an env var hook.
export LB_USE_SCALING_SLOW=1

echo "[$(ts)] === SCALING ABLATION (30B-FP8) START ==="

# 1) T-trajectories with 30B slow on MsPacman
#    (must re-collect because 30B residuals are 2048-d vs 8B's 4096-d)
echo "[$(ts)] === Step 1: T-trajectory collection (30B slow) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_text_bridge_baseline.py \
    --game MsPacman --episodes 10 --ticks 750 --slow-max-tokens 96 --seed 0 \
    --out-dir results/t_trajectories_v2_30b \
    > /tmp/scaling_t_collect.log 2>&1

# 2) Stage C v2 against the 30B teacher
#    The trainable projection learns 2048 -> 4096 (vs the 8B's 4096 -> 4096)
echo "[$(ts)] === Step 2: Stage C v2 (30B teacher) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2_30b/MsPacman_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --out checkpoints/stage_c/v2_mspacman_30b.pt \
    > /tmp/scaling_stage_c.log 2>&1

# 3) F/T/L eval with 30B slow in the loop
echo "[$(ts)] === Step 3: F/T/L eval (30B slow) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games MsPacman --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --bridge-ckpt checkpoints/stage_c/v2_mspacman_30b.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_mspacman_30b.json \
    > /tmp/scaling_eval.log 2>&1

# 4) MI diagnostic on the 30B bridge
echo "[$(ts)] === Step 4: MI diagnostic (30B bridge) ==="
CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
    --trace 'results/t_trajectories_v2_30b/MsPacman_seed*.pt' \
    --bridge-ckpt checkpoints/stage_c/v2_mspacman_30b.pt \
    --out results/mi_diagnostic_mspacman_30b.json \
    > /tmp/scaling_mi.log 2>&1

echo "[$(ts)] === SCALING ABLATION COMPLETE ==="
