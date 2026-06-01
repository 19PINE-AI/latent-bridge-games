#!/usr/bin/env bash
# "Fix the teacher": retrain Stage C distilling from the ROBUST (suffix-aware) head, which
# has a HEALTHY text teacher (T=69.5 ~= F), instead of the bare head (T=17, broken/OOD).
# Uses the expert-driven Stage B traces (real driving). Then eval F/T/L on the robust head
# WITH bridge-replace {zero,random} controls. Gate: L>F AND L(real) >> L(zero/random).
set -u
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SDL_VIDEODRIVER=dummy
export PYTHONPATH=$PWD

LOG=results/md_fixt
mkdir -p "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/MASTER.log"; }
run_stage(){ local name="$1" marker="$2"; shift 2
  if [ -f "$marker" ]; then say "SKIP $name"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/$name.log" 2>&1; then touch "$marker"; say "OK $name"
  else say "FAIL $name (see $LOG/$name.log)"; echo "HALT=$name">"$LOG/HALT"; exit 1; fi
}

ROBUST=checkpoints/stage_a/metadrive_robust.pt
BRIDGE=checkpoints/stage_c/v2_metadrive_robustteacher.pt
TRAJ='results/t_traj_md_expert/MetaDrive_seed*.pt'

# ---- Stage C: distill from the ROBUST teacher on expert-driven traces ----
run_stage stageC_robust "$LOG/.stageC" \
  python3 -m src.training.stage_c_v2 \
    --trace "$TRAJ" --epochs 5 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt $ROBUST --out $BRIDGE

EV="--games MetaDrive --seeds 0 1 2 --episodes 2 --max-ticks 400 \
  --fast-ckpt $ROBUST --bridge-ckpt $BRIDGE --max-slow-tokens 64"

# ---- Greedy: F/T/L + controls ----
run_stage eval_FTL "$LOG/.eval_FTL" \
  python3 -m src.eval.benchmark --strategies F T L $EV --out results/eval_md_fixt_FTL.json
run_stage eval_Lzero "$LOG/.eval_Lzero" \
  python3 -m src.eval.benchmark --strategies L $EV --bridge-replace zero --out results/eval_md_fixt_Lzero.json
run_stage eval_Lrandom "$LOG/.eval_Lrandom" \
  python3 -m src.eval.benchmark --strategies L $EV --bridge-replace random --out results/eval_md_fixt_Lrandom.json

# ---- Sample: F/T/L + controls (the paper's decoder; latent can shift the distribution) ----
EVS="$EV --action-policy sample --action-temperature 1.0"
run_stage eval_FTL_s "$LOG/.eval_FTL_s" \
  python3 -m src.eval.benchmark --strategies F T L $EVS --out results/eval_md_fixt_FTL_sample.json
run_stage eval_Lzero_s "$LOG/.eval_Lzero_s" \
  python3 -m src.eval.benchmark --strategies L $EVS --bridge-replace zero --out results/eval_md_fixt_Lzero_sample.json
run_stage eval_Lrandom_s "$LOG/.eval_Lrandom_s" \
  python3 -m src.eval.benchmark --strategies L $EVS --bridge-replace random --out results/eval_md_fixt_Lrandom_sample.json

say "FIXTEACHER_COMPLETE"; echo DONE >"$LOG/COMPLETE"
