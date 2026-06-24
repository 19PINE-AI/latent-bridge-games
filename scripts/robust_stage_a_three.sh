#!/bin/bash
# Apply Stage A robustness ablation (suffix-prob=0.5) to the 3 collapsed games:
# River Raid, Q*bert, Enduro. If diagnosis is correct, all 3 should break their
# T=L collapses (like SpaceInvaders did).

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

# game → (sb3_traj_path, t_traj_pattern, eval_max_ticks)
run_robust() {
    local game=$1
    local sb3_traj=$2
    local t_glob=$3
    local max_ticks=$4
    local game_lower=${game,,}

    echo "[$(ts)] === [$game] Step 1: Robust Stage A (suffix-prob=0.5) ==="
    kill_vllm
    # Use the FIRST T-trajectory file as the suffix source (real slow emissions)
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

echo "[$(ts)] === ROBUST STAGE A — 3 GAMES START ==="

run_robust Riverraid results/trajectories_Riverraid_sb3_dqn.pt \
    'results/t_trajectories_v2/Riverraid_seed*.pt' 750
run_robust Qbert results/trajectories_Qbert_sb3_dqn.pt \
    'results/t_trajectories_v2/Qbert_seed*.pt' 750
run_robust Enduro results/trajectories_Enduro_ppo.pt \
    'results/t_trajectories_v2/Enduro_seed*.pt' 750

echo "[$(ts)] === ROBUST STAGE A — 3 GAMES COMPLETE ==="
