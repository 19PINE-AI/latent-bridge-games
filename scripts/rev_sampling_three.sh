#!/usr/bin/env bash
# Confirmatory sampling re-run (tau=1.0) of the three decoder-robust L>T winners,
# at their reported variants: MsPacman bare, RoadRunner bare, River Raid robust.
# Mirrors the Seaquest/bare sampling check (results/rev_seaquest/).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/rev_sampling logs

run () {
  local game=$1 fast=$2 bridge=$3 tag=$4
  echo "[$(date +%H:%M:%S)] === $tag ($game) sampling tau=1.0 ==="
  python3 -m src.eval.benchmark \
    --strategies F T L --games "$game" --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt "checkpoints/stage_a/$fast.pt" \
    --bridge-ckpt "checkpoints/stage_c/$bridge.pt" \
    --max-slow-tokens 64 \
    --action-policy sample --action-temperature 1.0 \
    --out "results/rev_sampling/${tag}_sample_t1.json"
}

run MsPacman   mspacman_sb3dqn_v2 v2_mspacman          mspacman_bare
run RoadRunner roadrunner_sb3dqn  v2_roadrunner        roadrunner_bare
run Riverraid  riverraid_robust   v2_riverraid_robust  riverraid_robust
echo "[$(date +%H:%M:%S)] === ALL THREE DONE ==="
