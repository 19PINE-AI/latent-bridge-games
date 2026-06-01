#!/usr/bin/env bash
# MetaDrive diagnostics: is the latent bridge actually used, or does the head ignore it?
# Matches the original BARE eval exactly (bare ckpt, seeds 0 1 2, 2 eps, max-ticks 400, greedy)
# so cells are directly comparable. Resumable via markers.
set -u
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SDL_VIDEODRIVER=dummy
export PYTHONPATH=/home/ubuntu/latent-bridge-games

LOG=results/md_diag
mkdir -p "$LOG" results/md_traces
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/MASTER.log"; }

run_stage(){ local name="$1" marker="$2"; shift 2
  if [ -f "$marker" ]; then say "SKIP $name"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/$name.log" 2>&1; then touch "$marker"; say "OK $name"
  else say "FAIL $name (see $LOG/$name.log)"; echo "HALT=$name">"$LOG/HALT"; exit 1; fi
}

BARE=checkpoints/stage_a/metadrive_bare.pt
BRIDGE=checkpoints/stage_c/v2_metadrive.pt
COMMON="--games MetaDrive --seeds 0 1 2 --episodes 2 --max-ticks 400 \
  --fast-ckpt $BARE --bridge-ckpt $BRIDGE --max-slow-tokens 64"

# L with the trained latent (re-run, with traces) — should reproduce L=69.45 if deterministic
run_stage L_none "$LOG/.L_none" \
  python3 -m src.eval.benchmark --strategies L $COMMON \
    --bridge-replace none --save-trace-dir results/md_traces \
    --out results/eval_md_diag_L_none.json

# L with latent tokens ZEROED — if score == L_none, head is conditioning on something else
run_stage L_zero "$LOG/.L_zero" \
  python3 -m src.eval.benchmark --strategies L $COMMON \
    --bridge-replace zero \
    --out results/eval_md_diag_L_zero.json

# L with RANDOM tokens at matched norm — control for "any tokens vs trained tokens"
run_stage L_random "$LOG/.L_random" \
  python3 -m src.eval.benchmark --strategies L $COMMON \
    --bridge-replace random \
    --out results/eval_md_diag_L_random.json

# T with traces (for the bare-L == robust-T anomaly diff). Uses bare ckpt.
run_stage T_trace "$LOG/.T_trace" \
  python3 -m src.eval.benchmark --strategies T $COMMON \
    --save-trace-dir results/md_traces \
    --out results/eval_md_diag_T.json

say "DIAG_COMPLETE"; echo DONE >"$LOG/COMPLETE"
