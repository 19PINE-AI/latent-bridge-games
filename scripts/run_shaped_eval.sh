#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"
export LB_FAST_MODEL_PATH=${REPO}/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=${REPO}/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
S=/tmp/shaped.status; L=/tmp/shaped.log; : > "$S"; : > "$L"
say(){ echo "[$(date '+%H:%M:%S')] $*" >> "$S"; }
say "START pid=$$"
retry(){ name="$1"; marker="$2"; shift 2; rm -f "$marker"
  for a in 1 2 3 4 5; do say "$name attempt $a"; "$@" >> "$L" 2>&1
    [ -f "$marker" ] && { say "$name OK"; return 0; }
    say "$name attempt $a FAILED sleep 20"; sleep 20; done
  say "$name GAVE_UP"; return 1; }
eg(){ python3 scripts/eval_minigrid_shaped.py --action-policy argmax --seeds 0 1 2 --episodes 4 --max-ticks 160 \
  --fast-ckpt checkpoints/stage_a/minigrid_astar.pt --bridge-ckpt checkpoints/stage_c/v2_minigrid.pt \
  --out results/eval_minigrid_shaped_greedy.json
  [ -f results/eval_minigrid_shaped_greedy.json ] && touch /tmp/mk_sg; }
es(){ python3 scripts/eval_minigrid_shaped.py --action-policy sample --action-temperature 1.0 --seeds 0 1 2 --episodes 4 --max-ticks 160 \
  --fast-ckpt checkpoints/stage_a/minigrid_astar.pt --bridge-ckpt checkpoints/stage_c/v2_minigrid.pt \
  --out results/eval_minigrid_shaped_sample.json
  [ -f results/eval_minigrid_shaped_sample.json ] && touch /tmp/mk_ss; }
retry ShapedGreedy /tmp/mk_sg eg || say "WARN greedy"
retry ShapedSample /tmp/mk_ss es || say "WARN sample"
python3 - >> "$S" 2>&1 <<'PY'
import json
for tag,p in [("greedy","results/eval_minigrid_shaped_greedy.json"),("sample","results/eval_minigrid_shaped_sample.json")]:
    try:
        s=json.load(open(p))["summary"]
        line="[RESULT] "+tag+" "+" ".join(f"{k}:fc={s[k]['frac_closed_mean']:.3f}±{s[k]['frac_closed_std']:.3f},sr={s[k]['success_rate']:.2f}" for k in ["F","T","L"] if k in s)
    except Exception as e: line="[RESULT] "+tag+" MISSING "+str(e)
    open("/tmp/shaped.status","a").write(line+"\n")
PY
say "SHAPED_DONE"
