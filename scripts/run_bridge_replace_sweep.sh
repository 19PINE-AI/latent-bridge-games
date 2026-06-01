#!/usr/bin/env bash
# Bridge-replacement control across ALL Atari games (was only MsPacman).
# For each game, eval L with the trained bridge vs zeroed vs random-at-matched-norm,
# at the game's canonical (fast-ckpt, bridge-ckpt, decoder). Tests whether L's
# advantage is learned content (L_trained >> L_random) or just prepended-slot
# architecture (L_trained ~= L_random ~= L_zero), per game.
set -u
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD

LOG=results/br_sweep
mkdir -p "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/MASTER.log"; }
run_stage(){ local name="$1" marker="$2"; shift 2
  if [ -f "$marker" ]; then say "SKIP $name"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/$name.log" 2>&1; then touch "$marker"; say "OK $name"
  else say "FAIL $name (see $LOG/$name.log)"; fi   # don't halt: keep sweeping other games
}

# game : fast-ckpt : bridge-ckpt : decoder-extra
SPECS=(
 "MsPacman:mspacman_sb3dqn_v2:v2_mspacman:"
 "RoadRunner:roadrunner_sb3dqn:v2_roadrunner:"
 "Seaquest:seaquest_sb3dqn:v2_seaquest:"
 "Riverraid:riverraid_sb3dqn:v2_riverraid:"
 "Enduro:enduro_ppo:v2_enduro:"
 "SpaceInvaders:spaceinvaders_sb3dqn:v2_spaceinvaders:"
 "Qbert:qbert_robust:v2_qbert_robust:--action-policy sample --action-temperature 1.0"
)

# Optional: LB_BR_GAMES restricts this worker to a space-separated subset of games,
# so two workers can run disjoint partitions concurrently (markers prevent races).
FILTER="${LB_BR_GAMES:-}"

for spec in "${SPECS[@]}"; do
  IFS=':' read -r game fa br extra <<< "$spec"
  if [ -n "$FILTER" ] && ! grep -qw "$game" <<< "$FILTER"; then continue; fi
  FA=checkpoints/stage_a/$fa.pt
  BR=checkpoints/stage_c/$br.pt
  EV="--strategies L --games $game --seeds 0 1 2 --episodes 4 --max-ticks 750 \
      --fast-ckpt $FA --bridge-ckpt $BR --max-slow-tokens 64 $extra"
  run_stage ${game}_trained "$LOG/.${game}_trained" \
    python3 -m src.eval.benchmark $EV --bridge-replace none   --out results/br_${game}_trained.json
  run_stage ${game}_zero    "$LOG/.${game}_zero" \
    python3 -m src.eval.benchmark $EV --bridge-replace zero   --out results/br_${game}_zero.json
  run_stage ${game}_random  "$LOG/.${game}_random" \
    python3 -m src.eval.benchmark $EV --bridge-replace random --out results/br_${game}_random.json
done

say "BRIDGE_REPLACE_SWEEP_COMPLETE"; echo DONE >"$LOG/COMPLETE"
