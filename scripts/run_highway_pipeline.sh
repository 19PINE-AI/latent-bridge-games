#!/bin/bash
# Highway B->C->eval, after Stage A finishes. Dense reward (no shaping needed).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"
export LB_FAST_MODEL_PATH=${REPO}/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=${REPO}/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
S=/tmp/hw_pipe.status; L=/tmp/hw_pipe.log; : > "$S"; : > "$L"
say(){ echo "[$(date '+%H:%M:%S')] $*" >> "$S"; }
say "START pid=$$"

# wait for Stage A checkpoint + completion marker (up to 60 min)
say "waiting for Stage A..."
for i in $(seq 1 120); do
  if grep -q STAGE_A_EXIT /tmp/hw_stage_a.log 2>/dev/null && [ -f checkpoints/stage_a/highway_scripted.pt ]; then
    say "Stage A done"; break
  fi
  sleep 30
done
[ -f checkpoints/stage_a/highway_scripted.pt ] || { say "ABORT: no Stage A ckpt"; exit 1; }

retry(){ name="$1"; marker="$2"; shift 2; rm -f "$marker"
  for a in 1 2 3 4 5; do say "$name attempt $a"; "$@" >> "$L" 2>&1
    [ -f "$marker" ] && { say "$name OK"; return 0; }
    say "$name attempt $a FAILED sleep 20"; sleep 20; done
  say "$name GAVE_UP"; return 1; }

sb(){ python3 scripts/run_text_bridge_baseline.py --game Highway --episodes 12 \
  --ticks 200 --slow-max-tokens 96 --seed 0 --out-dir results/t_trajectories_v2_highway
  ls results/t_trajectories_v2_highway/Highway_seed*.pt >/dev/null 2>&1 && touch /tmp/mk_hb; }
retry StageB /tmp/mk_hb sb || { echo "FAILED StageB" >> "$S"; exit 1; }

sc(){ python3 -m src.training.stage_c_v2 --trace 'results/t_trajectories_v2_highway/Highway_seed*.pt' \
  --epochs 1 --grad-accum 4 --lr 5e-5 --stage-a-ckpt checkpoints/stage_a/highway_scripted.pt \
  --out checkpoints/stage_c/v2_highway.pt
  [ -f checkpoints/stage_c/v2_highway.pt ] && touch /tmp/mk_hc; }
retry StageC /tmp/mk_hc sc || { echo "FAILED StageC" >> "$S"; exit 1; }

eg(){ python3 -m src.eval.benchmark --strategies F T L --games Highway --seeds 0 1 2 --episodes 4 \
  --max-ticks 200 --fast-ckpt checkpoints/stage_a/highway_scripted.pt \
  --bridge-ckpt checkpoints/stage_c/v2_highway.pt --max-slow-tokens 64 \
  --out results/eval_v2_highway_greedy.json
  [ -f results/eval_v2_highway_greedy.json ] && touch /tmp/mk_heg; }
retry EvalGreedy /tmp/mk_heg eg || say "WARN greedy"

es(){ python3 -m src.eval.benchmark --strategies F T L --games Highway --seeds 0 1 2 --episodes 4 \
  --max-ticks 200 --fast-ckpt checkpoints/stage_a/highway_scripted.pt \
  --bridge-ckpt checkpoints/stage_c/v2_highway.pt --max-slow-tokens 64 \
  --action-policy sample --action-temperature 1.0 \
  --out results/eval_v2_highway_sample.json
  [ -f results/eval_v2_highway_sample.json ] && touch /tmp/mk_hes; }
retry EvalSample /tmp/mk_hes es || say "WARN sample"

python3 - >> "$S" 2>&1 <<'PY'
import json,numpy as np
from collections import defaultdict
for tag,p in [("greedy","results/eval_v2_highway_greedy.json"),("sample","results/eval_v2_highway_sample.json")]:
    try:
        d=json.load(open(p)); by=defaultdict(list)
        for c in d["cells"]: by[c["strategy"]].append(c["score"])
        line="[RESULT] "+tag+" "+" ".join(f"{s}={np.mean(by[s]):.2f}±{np.std(by[s],ddof=1) if len(by[s])>1 else 0:.2f}" for s in ["F","T","L"] if by[s])
    except Exception as e: line="[RESULT] "+tag+" MISSING "+str(e)
    open("/tmp/hw_pipe.status","a").write(line+"\n")
PY
say "HW_PIPE_DONE"
