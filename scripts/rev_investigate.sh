#!/usr/bin/env bash
# Investigation before any paper reframe (user: "investigate more first").
#   (1) GREEDY CONTROL: re-run greedy F/T/L on the SAME checkpoints used for the sampling
#       sweep, to confirm the paper's greedy L>T reproduces (rules out checkpoint/code drift
#       as the explanation for the greedy-vs-sampling gap).
#   (2) tau=0.3: a low-temperature decoder between greedy and tau=0.5, to find whether there
#       is any stochastic regime where L>T survives.
# Reported variants: MsPacman bare, RoadRunner bare, River Raid robust, Seaquest bare.
set -u
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/rev_sampling logs

# game fast bridge tag
GAMES=(
  "MsPacman   mspacman_sb3dqn_v2 v2_mspacman          mspacman_bare"
  "RoadRunner roadrunner_sb3dqn  v2_roadrunner        roadrunner_bare"
  "Riverraid  riverraid_robust   v2_riverraid_robust  riverraid_robust"
  "Seaquest   seaquest_sb3dqn    v2_seaquest          seaquest_bare"
)

run () {  # game fast bridge tag policy temp suffix
  local game=$1 fast=$2 bridge=$3 tag=$4 policy=$5 temp=$6 sfx=$7
  echo "[$(date +%H:%M:%S)] === $tag ($game) $policy temp=$temp ==="
  python3 -m src.eval.benchmark \
    --strategies F T L --games "$game" --seeds 0 1 2 --episodes 4 --max-ticks 500 \
    --fast-ckpt "checkpoints/stage_a/$fast.pt" \
    --bridge-ckpt "checkpoints/stage_c/$bridge.pt" \
    --max-slow-tokens 64 \
    --action-policy "$policy" --action-temperature "$temp" \
    --out "results/rev_sampling/${tag}_${sfx}.json"
}

echo "########## PHASE 1: GREEDY CONTROL ##########"
for g in "${GAMES[@]}"; do run $g argmax 1.0 greedy_reconfirm; done

echo "########## PHASE 2: tau=0.3 ##########"
for g in "${GAMES[@]}"; do run $g sample 0.3 sample_t03; done

echo "[$(date +%H:%M:%S)] === INVESTIGATION DONE ==="
