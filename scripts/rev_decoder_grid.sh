#!/usr/bin/env bash
# Finer per-channel decoder search to find the TRUE optimum per (game, channel), feeding a
# held-out (leave-one-seed-out) decoder-selection analysis. Target grid per game:
#   {greedy, 0.3, 0.5, 0.7, 1.0, 1.5}.  We only run cells not already on disk:
#     - MsPacman/RoadRunner/RiverRaid: add 0.7  (their L/T peak at greedy or 0.5; 1.5 would be worse)
#     - Seaquest: add 0.7 and 1.5  (T and L still climbing toward 1.0)
#     - Q*bert: full grid (we only had paper means, no per-episode JSONs)
set -u
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/rev_sampling logs

run () {  # game fast bridge tag policy temp suffix
  local game=$1 fast=$2 bridge=$3 tag=$4 policy=$5 temp=$6 sfx=$7
  local out="results/rev_sampling/${tag}_${sfx}.json"
  if [ -f "$out" ]; then echo "[$(date +%H:%M:%S)] skip $out (exists)"; return; fi
  echo "[$(date +%H:%M:%S)] === $tag ($game) $policy temp=$temp ==="
  python3 -m src.eval.benchmark \
    --strategies F T L --games "$game" --seeds 0 1 2 --episodes 4 --max-ticks 500 \
    --fast-ckpt "checkpoints/stage_a/$fast.pt" --bridge-ckpt "checkpoints/stage_c/$bridge.pt" \
    --max-slow-tokens 64 --action-policy "$policy" --action-temperature "$temp" --out "$out"
}

# --- tau=0.7 for the four already-gridded games ---
run MsPacman   mspacman_sb3dqn_v2 v2_mspacman         mspacman_bare    sample 0.7 sample_t07
run RoadRunner roadrunner_sb3dqn  v2_roadrunner       roadrunner_bare  sample 0.7 sample_t07
run Riverraid  riverraid_robust   v2_riverraid_robust riverraid_robust sample 0.7 sample_t07
run Seaquest   seaquest_sb3dqn    v2_seaquest         seaquest_bare    sample 0.7 sample_t07
run Seaquest   seaquest_sb3dqn    v2_seaquest         seaquest_bare    sample 1.5 sample_t15

# --- Q*bert full grid (robust SA is its canonical variant) ---
QF=qbert_robust; QB=v2_qbert_robust; QT=qbert_robust
run Qbert $QF $QB $QT argmax 1.0 greedy_reconfirm
run Qbert $QF $QB $QT sample 0.3 sample_t03
run Qbert $QF $QB $QT sample 0.5 sample_t05
run Qbert $QF $QB $QT sample 0.7 sample_t07
run Qbert $QF $QB $QT sample 1.0 sample_t1
run Qbert $QF $QB $QT sample 1.5 sample_t15

# --- Enduro and SpaceInvaders (floor / near-collapse cells) to complete all 7 games ---
# coarser grid {greedy, 0.5, 1.0}: their scores are tiny/declining so fine resolution adds little,
# but >=3 decoders is enough for held-out selection + a best-achievable cell.
for spec in \
  "Enduro        enduro_robust        v2_enduro_robust        enduro_robust" \
  "SpaceInvaders spaceinvaders_robust v2_spaceinvaders_robust spaceinvaders_robust"; do
  set -- $spec
  run "$1" "$2" "$3" "$4" argmax 1.0 greedy_reconfirm
  run "$1" "$2" "$3" "$4" sample 0.5 sample_t05
  run "$1" "$2" "$3" "$4" sample 1.0 sample_t1
done

echo "[$(date +%H:%M:%S)] === DECODER GRID DONE ==="
