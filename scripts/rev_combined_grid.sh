#!/usr/bin/env bash
# Combined-channel (B = text suffix AND latent tokens together) decoder grid, to test whether
# "having both channels" beats best-T and best-L. Runs B-only at the SAME per-game decoder grid as
# the T/L sweep, so best-achievable(B) is selected from an identical decoder set (held-out, same
# protocol). Outputs results/rev_sampling/<tag>_B_<suffix>.json (B cell only; F/T/L already on disk).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/rev_sampling logs

runB () {  # tag fast bridge ; then decoder triples "policy temp suffix" ...
  local tag=$1 fast=$2 bridge=$3 game=$4; shift 4
  for trip in "$@"; do
    set -- $trip; local policy=$1 temp=$2 sfx=$3
    local out="results/rev_sampling/${tag}_B_${sfx}.json"
    if [ -f "$out" ]; then echo "[$(date +%H:%M:%S)] skip $out"; continue; fi
    echo "[$(date +%H:%M:%S)] === B $tag $policy temp=$temp ==="
    python3 -m src.eval.benchmark --strategies B --games "$game" --seeds 0 1 2 --episodes 4 \
      --max-ticks 500 --fast-ckpt "checkpoints/stage_a/$fast.pt" \
      --bridge-ckpt "checkpoints/stage_c/$bridge.pt" --max-slow-tokens 64 \
      --action-policy "$policy" --action-temperature "$temp" --out "$out"
  done
}

FULL=("argmax 1.0 greedy_reconfirm" "sample 0.3 sample_t03" "sample 0.5 sample_t05" "sample 0.7 sample_t07" "sample 1.0 sample_t1")
FULL15=("${FULL[@]}" "sample 1.5 sample_t15")
COARSE=("argmax 1.0 greedy_reconfirm" "sample 0.5 sample_t05" "sample 1.0 sample_t1")

runB mspacman_bare        mspacman_sb3dqn_v2   v2_mspacman         MsPacman      "${FULL[@]}"
runB roadrunner_bare      roadrunner_sb3dqn    v2_roadrunner       RoadRunner    "${FULL[@]}"
runB riverraid_robust     riverraid_robust     v2_riverraid_robust Riverraid     "${FULL[@]}"
runB seaquest_bare        seaquest_sb3dqn      v2_seaquest         Seaquest      "${FULL15[@]}"
runB qbert_robust         qbert_robust         v2_qbert_robust     Qbert         "${FULL15[@]}"
runB enduro_robust        enduro_robust        v2_enduro_robust    Enduro        "${COARSE[@]}"
runB spaceinvaders_robust spaceinvaders_robust v2_spaceinvaders_robust SpaceInvaders "${COARSE[@]}"

echo "[$(date +%H:%M:%S)] === COMBINED GRID DONE ==="
