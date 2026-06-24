#!/usr/bin/env bash
# Record F, T, AND L replay traces for every game with the CANONICAL checkpoints,
# then render individual F/T/L videos + a 3-way side-by-side. Previous demos had
# only F and L; this adds the text bridge (T) so the contrast is complete.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD

LOG=results/ftl_demos; mkdir -p "$LOG" traces/ftl demos
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/MASTER.log"; }
run(){ local name="$1" marker="$2"; shift 2
  if [ -f "$marker" ]; then say "SKIP $name"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/$name.log" 2>&1; then touch "$marker"; say "OK $name"
  else say "FAIL $name (see $LOG/$name.log)"; fi   # keep going on failure
}

# game : fast : bridge : lower : extra-env : extra-args
SPECS=(
 "MsPacman:mspacman_sb3dqn_v2:v2_mspacman:mspacman::"
 "RoadRunner:roadrunner_sb3dqn:v2_roadrunner:roadrunner::"
 "Seaquest:seaquest_sb3dqn:v2_seaquest:seaquest::"
 "Riverraid:riverraid_robust:v2_riverraid_robust:riverraid::"
 "Qbert:qbert_robust:v2_qbert_robust:qbert::"
 "SpaceInvaders:spaceinvaders_robust:v2_spaceinvaders_robust:spaceinvaders::"
 "Enduro:enduro_ppo:v2_enduro:enduro::"
 "MetaDrive:metadrive_plan_robust:v2_metadrive_plan:metadrive:SXSXSX:"
)

for spec in "${SPECS[@]}"; do
  IFS=':' read -r game fa br lower mdmap extra <<< "$spec"
  FA=checkpoints/stage_a/$fa.pt
  BR=checkpoints/stage_c/$br.pt
  TD=traces/ftl
  ENV=""
  TICKS=500
  if [ "$game" = "MetaDrive" ]; then ENV="env LB_MD_MAP=$mdmap SDL_VIDEODRIVER=dummy"; TICKS=400; fi

  run ${game}_FTL "$LOG/.${game}_rec" \
    $ENV python3 -m src.eval.benchmark --strategies F T L \
      --games $game --seeds 0 --episodes 1 --max-ticks $TICKS \
      --fast-ckpt $FA --bridge-ckpt $BR --max-slow-tokens 64 \
      --save-trace-dir $TD --out results/eval_ftl_demo_${lower}.json $extra

  for S in F T L; do
    run render_${game}_${S} "$LOG/.r_${game}_${S}" \
      python3 scripts/render_demo_mp4.py --trace-dir $TD/${S}_${game}_seed0 \
        --out demos/${lower}_${S}.mp4 --fps 12
  done
  run render_${game}_3way "$LOG/.r_${game}_3way" \
    python3 scripts/render_demo_mp4.py \
      --trace-dir $TD/F_${game}_seed0 $TD/T_${game}_seed0 $TD/L_${game}_seed0 \
      --labels F T L --out demos/${lower}_F_T_L.mp4 --fps 12
done

say "FTL_DEMOS_COMPLETE"; echo DONE >"$LOG/COMPLETE"
ls -la demos/*_T.mp4 demos/*_F_T_L.mp4 2>/dev/null | tee -a "$LOG/MASTER.log"
