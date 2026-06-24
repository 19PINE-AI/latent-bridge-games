#!/usr/bin/env bash
# PLANNING-HEAVY MetaDrive: map=SXSXSX (straights + X-intersections), out_of_route_done=True.
# Hypothesis: survival now needs SLOW route decisions (take the correct exit), so the slow
# model's guidance should beat fast-only (T>F) and the latent should transfer (L>F, L>>L_zero).
# This distinguishes "fast model is enough for driving" from "the old task had no slow component".
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SDL_VIDEODRIVER=dummy
export LB_MD_MAP=SXSXSX
export PYTHONPATH=$PWD

LOG=results/md_plan
mkdir -p "$LOG" results/t_traj_md_plan
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/MASTER.log"; }
run_stage(){ local name="$1" marker="$2"; shift 2
  if [ -f "$marker" ]; then say "SKIP $name"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/$name.log" 2>&1; then touch "$marker"; say "OK $name"
  else say "FAIL $name (see $LOG/$name.log)"; echo "HALT=$name">"$LOG/HALT"; exit 1; fi
}

TRAJ=results/trajectories_MetaDrive_plan.pt
SA_BARE=checkpoints/stage_a/metadrive_plan_bare.pt
SA_ROB=checkpoints/stage_a/metadrive_plan_robust.pt
BRIDGE=checkpoints/stage_c/v2_metadrive_plan.pt
TR='results/t_traj_md_plan/MetaDrive_seed*.pt'

# 0. collect expert data on the planning map
run_stage collect "$LOG/.collect" \
  python3 scripts/collect_metadrive_expert.py --episodes 40 --seed 0 --max-ticks 600 --out $TRAJ

# A. Stage A bare + robust (robust = suffix-aware, the healthy-teacher head)
run_stage stageA_bare "$LOG/.sa_bare" \
  python3 -m src.training.stage_a_behavioral --traj $TRAJ \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 --out $SA_BARE
run_stage stageA_robust "$LOG/.sa_rob" \
  python3 -m src.training.stage_a_behavioral --traj $TRAJ \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 --suffix-prob 0.5 --out $SA_ROB

# B. Stage B: expert-driven slow-model trajectories (real driving through junctions)
run_stage stageB "$LOG/.sb" \
  python3 scripts/run_text_bridge_baseline.py \
    --game MetaDrive --episodes 16 --ticks 500 --slow-max-tokens 96 --seed 0 \
    --action-policy md-expert --epsilon 0.15 --out-dir results/t_traj_md_plan

# C. Stage C distilled from the ROBUST (healthy) teacher
run_stage stageC "$LOG/.sc" \
  python3 -m src.training.stage_c_v2 --trace "$TR" \
    --epochs 5 --grad-accum 4 --lr 5e-5 --stage-a-ckpt $SA_ROB --out $BRIDGE

# Eval on the ROBUST head (healthy teacher), both decoders, with controls
EV="--games MetaDrive --seeds 0 1 2 3 --episodes 2 --max-ticks 500 \
  --fast-ckpt $SA_ROB --bridge-ckpt $BRIDGE --max-slow-tokens 64"
run_stage eval_FTL "$LOG/.e_ftl" \
  python3 -m src.eval.benchmark --strategies F T L $EV --out results/eval_md_plan_FTL.json
run_stage eval_Lzero "$LOG/.e_lz" \
  python3 -m src.eval.benchmark --strategies L $EV --bridge-replace zero --out results/eval_md_plan_Lzero.json
run_stage eval_Lrand "$LOG/.e_lr" \
  python3 -m src.eval.benchmark --strategies L $EV --bridge-replace random --out results/eval_md_plan_Lrandom.json
EVS="$EV --action-policy sample --action-temperature 1.0"
run_stage eval_FTL_s "$LOG/.e_ftls" \
  python3 -m src.eval.benchmark --strategies F T L $EVS --out results/eval_md_plan_FTL_sample.json
run_stage eval_Lzero_s "$LOG/.e_lzs" \
  python3 -m src.eval.benchmark --strategies L $EVS --bridge-replace zero --out results/eval_md_plan_Lzero_sample.json

say "PLANNING_COMPLETE"; echo DONE >"$LOG/COMPLETE"
