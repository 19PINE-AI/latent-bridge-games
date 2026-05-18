#!/bin/bash
# Phase 8: Apply Stage A robustness ablation to the games that ALREADY work
# (RoadRunner, MsPacman, Seaquest). Tests whether robust SA is universally
# beneficial or only matters for collapsed games.

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

run_robust() {
    local game=$1
    local sb3_traj=$2
    local t_glob=$3
    local max_ticks=$4
    local game_lower=${game,,}

    echo "[$(ts)] === [$game] Step 1: Robust Stage A (suffix-prob=0.5) ==="
    kill_vllm
    local suffix_src=$(ls $t_glob 2>/dev/null | head -1)
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_a_behavioral \
        --traj "$sb3_traj" \
        --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
        --suffix-prob 0.5 \
        --suffix-from-trace "$suffix_src" \
        --out checkpoints/stage_a/${game_lower}_robust.pt \
        > /tmp/${game_lower}_robust_stage_a.log 2>&1

    echo "[$(ts)] === [$game] Step 2: Stage C v2 against robust head ==="
    kill_vllm
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
        --trace "$t_glob" \
        --epochs 1 --grad-accum 4 --lr 5e-5 \
        --stage-a-ckpt checkpoints/stage_a/${game_lower}_robust.pt \
        --out checkpoints/stage_c/v2_${game_lower}_robust.pt \
        > /tmp/${game_lower}_robust_stage_c.log 2>&1

    echo "[$(ts)] === [$game] Step 3: F/T/L eval (robust) ==="
    kill_vllm
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
        --strategies F T L --games "$game" --seeds 0 1 2 --episodes 4 \
        --max-ticks $max_ticks \
        --fast-ckpt checkpoints/stage_a/${game_lower}_robust.pt \
        --bridge-ckpt checkpoints/stage_c/v2_${game_lower}_robust.pt \
        --max-slow-tokens 64 \
        --out results/eval_v2_${game_lower}_robust.json \
        > /tmp/${game_lower}_robust_eval.log 2>&1
}

echo "[$(ts)] === ROBUST STAGE A — WINNERS START ==="

run_robust RoadRunner results/trajectories_RoadRunner_sb3_dqn.pt \
    'results/t_trajectories_v2/RoadRunner_seed*.pt' 750
run_robust MsPacman   results/trajectories_MsPacman_sb3_dqn.pt \
    'results/t_trajectories_v2/MsPacman_seed*.pt' 500
run_robust Seaquest   results/trajectories_Seaquest_sb3_dqn.pt \
    'results/t_trajectories_v2/Seaquest_seed*.pt' 500

echo "[$(ts)] === ROBUST STAGE A — WINNERS COMPLETE ==="
