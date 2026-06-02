#!/usr/bin/env bash
# Record MetaDrive F and L replay traces (planning map), then render MP4s matching
# the Atari demo format (gameplay + slow-guidance caption, plus F-vs-L side-by-side).
set -u
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=$PWD/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=$PWD/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SDL_VIDEODRIVER=dummy
export LB_MD_MAP=SXSXSX
export PYTHONPATH=$PWD

LOG=results/md_demo; mkdir -p "$LOG" traces demos
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/MASTER.log"; }
run(){ local name="$1" marker="$2"; shift 2
  if [ -f "$marker" ]; then say "SKIP $name"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/$name.log" 2>&1; then touch "$marker"; say "OK $name"
  else say "FAIL $name (see $LOG/$name.log)"; exit 1; fi
}

ROBUST=checkpoints/stage_a/metadrive_plan_robust.pt
BRIDGE=checkpoints/stage_c/v2_metadrive_plan.pt
EV="--games MetaDrive --seeds 0 --episodes 1 --max-ticks 400 \
    --fast-ckpt $ROBUST --bridge-ckpt $BRIDGE --max-slow-tokens 64 --save-trace-dir traces"

# F and L on the same seed (deterministic env -> directly comparable)
run trace_F "$LOG/.tF" python3 -m src.eval.benchmark --strategies F $EV
run trace_L "$LOG/.tL" python3 -m src.eval.benchmark --strategies L $EV

# Render: single L, single F, and side-by-side
run render_L "$LOG/.rL" python3 scripts/render_demo_mp4.py \
  --trace-dir traces/L_MetaDrive_seed0 --out demos/metadrive_L.mp4 --fps 12
run render_F "$LOG/.rF" python3 scripts/render_demo_mp4.py \
  --trace-dir traces/F_MetaDrive_seed0 --out demos/metadrive_F.mp4 --fps 12
run render_FL "$LOG/.rFL" python3 scripts/render_demo_mp4.py \
  --trace-dir traces/F_MetaDrive_seed0 traces/L_MetaDrive_seed0 --labels F L \
  --out demos/metadrive_F_vs_L.mp4 --fps 12

say "METADRIVE_DEMO_COMPLETE"; echo DONE >"$LOG/COMPLETE"
ls -la demos/metadrive_*.mp4 | tee -a "$LOG/MASTER.log"
