#!/usr/bin/env bash
# Follow-up: does the L>T advantage survive a GENTLER stochastic decoder (tau=0.5)?
# tau=1.0 (rev_sampling_three.sh) broke/inverted the three showcase wins; tau=0.5 sits
# between greedy and tau=1.0 and tests whether there is any stochastic regime where L>T holds.
# Reported variants: MsPacman bare, RoadRunner bare, River Raid robust, Seaquest bare.
set -u
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/rev_sampling logs

run () {
  local game=$1 fast=$2 bridge=$3 tag=$4
  echo "[$(date +%H:%M:%S)] === $tag ($game) sampling tau=0.5 ==="
  python3 -m src.eval.benchmark \
    --strategies F T L --games "$game" --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 \
    --fast-ckpt "checkpoints/stage_a/$fast.pt" \
    --bridge-ckpt "checkpoints/stage_c/$bridge.pt" \
    --max-slow-tokens 64 \
    --action-policy sample --action-temperature 0.5 \
    --out "results/rev_sampling/${tag}_sample_t05.json"
}

run MsPacman   mspacman_sb3dqn_v2 v2_mspacman          mspacman_bare
run RoadRunner roadrunner_sb3dqn  v2_roadrunner        roadrunner_bare
run Riverraid  riverraid_robust   v2_riverraid_robust  riverraid_robust
run Seaquest   seaquest_sb3dqn    v2_seaquest          seaquest_bare
echo "[$(date +%H:%M:%S)] === ALL FOUR DONE ==="
