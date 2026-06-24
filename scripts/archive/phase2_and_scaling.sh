#!/bin/bash
# Phase 2D (Stage A robustness) + Phase 2B (S baseline) + Phase 2C (Oracle) +
# Phase 3B (30B-FP8 scaling). Chained sequentially since they all need GPU.
# Total ETA ~6 hours. Designed to run unattended overnight.

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

echo "[$(ts)] === MULTI-PHASE OVERNIGHT PIPELINE START ==="

# ========== Phase 2D: Stage A robustness on SpaceInvaders ==========
echo "[$(ts)] === Phase 2D / Step 1: Stage A robust retrain (SI, suffix-prob=0.5) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_SpaceInvaders_sb3_dqn.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --suffix-prob 0.5 \
    --suffix-from-trace results/t_trajectories_v2_expert/SpaceInvaders_seed0.pt \
    --out checkpoints/stage_a/spaceinvaders_robust.pt \
    > /tmp/si_robust_stage_a.log 2>&1

echo "[$(ts)] === Phase 2D / Step 2: Stage C v2 against robust head ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2_expert/SpaceInvaders_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --out checkpoints/stage_c/v2_spaceinvaders_robust.pt \
    > /tmp/si_robust_stage_c.log 2>&1

echo "[$(ts)] === Phase 2D / Step 3: F/T/L eval (robust SI) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games SpaceInvaders --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/spaceinvaders_robust.pt \
    --bridge-ckpt checkpoints/stage_c/v2_spaceinvaders_robust.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_spaceinvaders_robust.json \
    > /tmp/si_robust_eval.log 2>&1

# ========== Phase 2B: S baseline on MsPacman ==========
echo "[$(ts)] === Phase 2B: Slow-only S baseline (MsPacman) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_slow_only_baseline.py \
    --game MsPacman --episodes 3 --max-ticks 500 --slow-max-tokens 128 \
    --out results/eval_slow_only_mspacman.json \
    > /tmp/slow_only_mspacman.log 2>&1

# ========== Phase 2C: Oracle O baseline on MsPacman ==========
echo "[$(ts)] === Phase 2C: Oracle O baseline (MsPacman) ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_oracle_baseline.py \
    --game MsPacman --episodes 3 --max-ticks 500 --slow-max-tokens 512 \
    --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --out results/eval_oracle_mspacman.json \
    > /tmp/oracle_mspacman.log 2>&1

# ========== Phase 3B: 30B-FP8 scaling ablation ==========
echo "[$(ts)] === Phase 3B / Step 1: T-trajectory collection (30B-FP8) ==="
kill_vllm
LB_USE_SCALING_SLOW=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 scripts/run_text_bridge_baseline.py \
    --game MsPacman --episodes 10 --ticks 750 --slow-max-tokens 96 --seed 0 \
    --out-dir results/t_trajectories_v2_30b \
    > /tmp/scaling_t_collect.log 2>&1

echo "[$(ts)] === Phase 3B / Step 2: Stage C v2 (30B teacher) ==="
kill_vllm
LB_USE_SCALING_SLOW=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2_30b/MsPacman_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --out checkpoints/stage_c/v2_mspacman_30b.pt \
    > /tmp/scaling_stage_c.log 2>&1

echo "[$(ts)] === Phase 3B / Step 3: F/T/L eval (30B slow) ==="
kill_vllm
LB_USE_SCALING_SLOW=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python3 -m src.eval.benchmark \
    --strategies F T L --games MsPacman --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --bridge-ckpt checkpoints/stage_c/v2_mspacman_30b.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_mspacman_30b.json \
    > /tmp/scaling_eval.log 2>&1

echo "[$(ts)] === Phase 3B / Step 4: MI diagnostic (30B bridge) ==="
CUDA_VISIBLE_DEVICES="" python3 -m src.eval.mi_diagnostic \
    --trace 'results/t_trajectories_v2_30b/MsPacman_seed*.pt' \
    --bridge-ckpt checkpoints/stage_c/v2_mspacman_30b.pt \
    --out results/mi_diagnostic_mspacman_30b.json \
    > /tmp/scaling_mi.log 2>&1

echo "[$(ts)] === MULTI-PHASE OVERNIGHT PIPELINE COMPLETE ==="
