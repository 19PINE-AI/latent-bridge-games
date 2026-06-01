#!/bin/bash
# Autonomous MiniGrid pipeline: Stage A -> B -> C -> eval (greedy + sample).
# Loads BOTH models from local folders (LB_*_MODEL_PATH) to bypass the broken
# shared HF cache. Retries each model-loading stage on transient failure.
# Writes a final summary + DONE marker. Designed to run unattended.

cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
LOG=/tmp/minigrid_auto.log
STATUS=/tmp/minigrid_auto.status
: > "$LOG"
echo "PIPELINE_START $(date '+%F %T')" | tee "$STATUS"

ts(){ date '+%F %T'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG" "$STATUS" ; }

# --- 0. Wait for slow model download to finish (up to 60 min) ---
say "Step0: waiting for slow model download..."
for i in $(seq 1 120); do
  python3 - <<'PY' && break
import os,glob,sys
d="local_models/Qwen3-VL-8B-Thinking"
ok = os.path.isfile(d+"/config.json") and len(glob.glob(d+"/*.safetensors"))>=1
# also require the download log to report DL_DONE
done = os.path.exists("/tmp/dl_slow.log") and "DL_DONE" in open("/tmp/dl_slow.log").read()
sys.exit(0 if (ok and done) else 1)
PY
  sleep 30
done
say "Step0: slow model ready (or timed out): $(python3 -c "import os,glob;d='local_models/Qwen3-VL-8B-Thinking';print('cfg',os.path.isfile(d+'/config.json'),'shards',len(glob.glob(d+'/*.safetensors')))" 2>/dev/null)"

# Retry wrapper: run a command up to 3 times; success = marker file appears
run_retry(){
  local name="$1"; shift
  local marker="$1"; shift
  for attempt in 1 2 3; do
    say "$name: attempt $attempt"
    rm -f "$marker"
    "$@"
    if [ -f "$marker" ]; then say "$name: OK"; return 0; fi
    say "$name: attempt $attempt failed; sleeping 30s"
    sleep 30
  done
  say "$name: FAILED after 3 attempts"
  return 1
}

# --- 1. Stage A ---
stage_a(){
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_MiniGrid.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --out checkpoints/stage_a/minigrid_astar.pt >> "$LOG" 2>&1
  [ -f checkpoints/stage_a/minigrid_astar.pt ] && touch /tmp/mk_stage_a
}
run_retry "StageA" /tmp/mk_stage_a stage_a || { say "ABORT at StageA"; echo "PIPELINE_FAILED StageA" >> "$STATUS"; exit 1; }

# --- 2. Stage B: T-trajectory collection (slow model in loop) ---
stage_b(){
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_text_bridge_baseline.py \
    --game MiniGrid --episodes 12 --ticks 150 --slow-max-tokens 96 --seed 0 \
    --fast-ckpt checkpoints/stage_a/minigrid_astar.pt \
    --out-dir results/t_trajectories_v2_minigrid >> "$LOG" 2>&1
  ls results/t_trajectories_v2_minigrid/MiniGrid_seed*.pt >/dev/null 2>&1 && touch /tmp/mk_stage_b
}
run_retry "StageB" /tmp/mk_stage_b stage_b || { say "ABORT at StageB"; echo "PIPELINE_FAILED StageB" >> "$STATUS"; exit 1; }

# --- 3. Stage C: bridge training ---
stage_c(){
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2_minigrid/MiniGrid_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/minigrid_astar.pt \
    --out checkpoints/stage_c/v2_minigrid.pt >> "$LOG" 2>&1
  [ -f checkpoints/stage_c/v2_minigrid.pt ] && touch /tmp/mk_stage_c
}
run_retry "StageC" /tmp/mk_stage_c stage_c || { say "ABORT at StageC"; echo "PIPELINE_FAILED StageC" >> "$STATUS"; exit 1; }

# --- 4. Eval greedy ---
eval_greedy(){
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games MiniGrid --seeds 0 1 2 --episodes 4 --max-ticks 200 \
    --fast-ckpt checkpoints/stage_a/minigrid_astar.pt --bridge-ckpt checkpoints/stage_c/v2_minigrid.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_minigrid_greedy.json >> "$LOG" 2>&1
  [ -f results/eval_v2_minigrid_greedy.json ] && touch /tmp/mk_eval_greedy
}
run_retry "EvalGreedy" /tmp/mk_eval_greedy eval_greedy || say "WARN: EvalGreedy failed (continuing)"

# --- 5. Eval sample tau=1.0 ---
eval_sample(){
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games MiniGrid --seeds 0 1 2 --episodes 4 --max-ticks 200 \
    --fast-ckpt checkpoints/stage_a/minigrid_astar.pt --bridge-ckpt checkpoints/stage_c/v2_minigrid.pt \
    --max-slow-tokens 64 --action-policy sample --action-temperature 1.0 \
    --out results/eval_v2_minigrid_sample.json >> "$LOG" 2>&1
  [ -f results/eval_v2_minigrid_sample.json ] && touch /tmp/mk_eval_sample
}
run_retry "EvalSample" /tmp/mk_eval_sample eval_sample || say "WARN: EvalSample failed (continuing)"

# --- 6. Summary ---
say "Computing summary..."
python3 - >> "$LOG" 2>&1 <<'PY'
import json, numpy as np
from collections import defaultdict
def summ(path):
    try:
        d=json.load(open(path))
    except Exception as e:
        return f"  {path}: MISSING ({e})"
    by=defaultdict(list)
    for c in d["cells"]: by[c["strategy"]].append(c["score"])
    out=[f"  {path}:"]
    for s in ["F","T","L"]:
        if by[s]:
            a=np.array(by[s]); out.append(f"    {s}: n={len(a)} mean={a.mean():.1f} std={a.std(ddof=1) if len(a)>1 else 0:.1f}")
    return "\n".join(out)
print("=== MiniGrid RESULTS ===")
print(summ("results/eval_v2_minigrid_greedy.json"))
print(summ("results/eval_v2_minigrid_sample.json"))
PY
tail -20 "$LOG" | sed -n '/MiniGrid RESULTS/,$p' | tee -a "$STATUS"
echo "PIPELINE_DONE $(date '+%F %T')" | tee -a "$STATUS"
