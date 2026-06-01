#!/usr/bin/env bash
# MetaDrive bridge RETRAIN with expert-driven Stage B (the fix for the inert latent).
# Stage A checkpoints are reused (they cloned the 174-reward expert; not the problem).
# Stage B now drives with MetaDrive's built-in PPO expert (--action-policy md-expert) so
# the car actually drives -> slow model narrates real driving -> bridge has signal.
# Then retrain Stage C and re-eval F/T/L WITH the bridge-replace control.
# Gate to put in paper: L>F AND L(real) >> L(zero/random).
set -u
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SDL_VIDEODRIVER=dummy
export PYTHONPATH=$PWD

LOG=results/md_v2x
mkdir -p "$LOG" results/t_traj_md_expert
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/MASTER.log"; }
run_stage(){ local name="$1" marker="$2"; shift 2
  if [ -f "$marker" ]; then say "SKIP $name"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/$name.log" 2>&1; then touch "$marker"; say "OK $name"
  else say "FAIL $name (see $LOG/$name.log)"; echo "HALT=$name">"$LOG/HALT"; exit 1; fi
}

BARE=checkpoints/stage_a/metadrive_bare.pt
BRIDGE=checkpoints/stage_c/v2_metadrive_expert.pt
TRAJ='results/t_traj_md_expert/MetaDrive_seed*.pt'

# ---- Stage B: expert-driven trajectories (16 eps so the bridge sees varied driving) ----
run_stage stageB_expert "$LOG/.stageB" \
  python3 scripts/run_text_bridge_baseline.py \
    --game MetaDrive --episodes 16 --ticks 400 --slow-max-tokens 96 --seed 0 \
    --action-policy md-expert --epsilon 0.15 \
    --out-dir results/t_traj_md_expert

# ---- Stage C: retrain bridge on the expert-driven traces ----
run_stage stageC_expert "$LOG/.stageC" \
  python3 -m src.training.stage_c_v2 \
    --trace "$TRAJ" \
    --epochs 5 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt $BARE \
    --out $BRIDGE

EV="--games MetaDrive --seeds 0 1 2 --episodes 2 --max-ticks 400 \
  --fast-ckpt $BARE --bridge-ckpt $BRIDGE --max-slow-tokens 64"

# ---- Eval F/T/L (real bridge) ----
run_stage eval_FTL "$LOG/.eval_FTL" \
  python3 -m src.eval.benchmark --strategies F T L $EV \
    --out results/eval_md_expert_FTL.json

# ---- Bridge-replace controls: L with zeroed / random latent ----
run_stage eval_Lzero "$LOG/.eval_Lzero" \
  python3 -m src.eval.benchmark --strategies L $EV \
    --bridge-replace zero --out results/eval_md_expert_Lzero.json
run_stage eval_Lrandom "$LOG/.eval_Lrandom" \
  python3 -m src.eval.benchmark --strategies L $EV \
    --bridge-replace random --out results/eval_md_expert_Lrandom.json

say "V2EXPERT_COMPLETE"; echo DONE >"$LOG/COMPLETE"
