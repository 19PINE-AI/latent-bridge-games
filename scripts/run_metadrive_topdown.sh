#!/usr/bin/env bash
# MetaDrive latent-bridge pipeline using the TOP-DOWN (pygame) renderer.
# No xvfb, no GL: the wrapper uses TopDownMetaDrive + SDL_VIDEODRIVER=dummy.
# Resumable: each stage is skipped if its output marker already exists.
set -u
cd /home/ubuntu/latent-bridge-games

export LB_FAST_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/home/ubuntu/latent-bridge-games

LOG=results/md_pipeline
mkdir -p "$LOG" checkpoints/stage_a checkpoints/stage_c results/t_trajectories_v2_metadrive

stamp(){ date "+%Y-%m-%d %H:%M:%S"; }
say(){ echo "[$(stamp)] $*" | tee -a "$LOG/MASTER.log"; }

run_stage(){  # name  marker  cmd...
  local name="$1"; local marker="$2"; shift 2
  if [ -f "$marker" ]; then say "SKIP $name (marker $marker exists)"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/$name.log" 2>&1; then
    touch "$marker"; say "OK    $name"
  else
    say "FAIL  $name (see $LOG/$name.log)"; echo "PIPELINE_HALTED_AT=$name" >"$LOG/HALT"; exit 1
  fi
}

# ---- Stage 0: collect expert trajectories ----
run_stage collect "$LOG/.collect_done" \
  python3 scripts/collect_metadrive_expert.py \
    --episodes 40 --seed 0 --max-ticks 500 \
    --out results/trajectories_MetaDrive.pt

# ---- Stage A bare ----
run_stage stageA_bare "$LOG/.stageA_bare_done" \
  python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_MetaDrive.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --out checkpoints/stage_a/metadrive_bare.pt

# ---- Stage A robust (OOD suffix augmentation) ----
run_stage stageA_robust "$LOG/.stageA_robust_done" \
  python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_MetaDrive.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 --suffix-prob 0.5 \
    --out checkpoints/stage_a/metadrive_robust.pt

# ---- Stage B: slow-model T-trajectories ----
run_stage stageB "$LOG/.stageB_done" \
  python3 scripts/run_text_bridge_baseline.py \
    --game MetaDrive --episodes 12 --ticks 400 --slow-max-tokens 96 --seed 0 \
    --out-dir results/t_trajectories_v2_metadrive

# ---- Stage C: bridge training ----
run_stage stageC "$LOG/.stageC_done" \
  python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2_metadrive/MetaDrive_seed*.pt' \
    --epochs 4 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/metadrive_bare.pt \
    --out checkpoints/stage_c/v2_metadrive.pt

# ---- Eval: F/T/L on bare action head (greedy) ----
run_stage eval_bare_greedy "$LOG/.eval_bare_greedy_done" \
  python3 -m src.eval.benchmark \
    --strategies F T L --games MetaDrive --seeds 0 1 2 --episodes 2 --max-ticks 400 \
    --fast-ckpt checkpoints/stage_a/metadrive_bare.pt \
    --bridge-ckpt checkpoints/stage_c/v2_metadrive.pt --max-slow-tokens 64 \
    --out results/eval_v2_metadrive_bare_greedy.json

# ---- Eval: F/T/L on robust action head (greedy) ----
run_stage eval_robust_greedy "$LOG/.eval_robust_greedy_done" \
  python3 -m src.eval.benchmark \
    --strategies F T L --games MetaDrive --seeds 0 1 2 --episodes 2 --max-ticks 400 \
    --fast-ckpt checkpoints/stage_a/metadrive_robust.pt \
    --bridge-ckpt checkpoints/stage_c/v2_metadrive.pt --max-slow-tokens 64 \
    --out results/eval_v2_metadrive_robust_greedy.json

say "PIPELINE_COMPLETE"
echo "DONE" >"$LOG/COMPLETE"
