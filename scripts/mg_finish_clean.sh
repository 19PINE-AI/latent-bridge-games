#!/bin/bash
# Robust MiniGrid finish: Stage B -> C -> eval(greedy) -> eval(sample).
# Models loaded from local folders (bypasses HF cache). 5 retries/stage.
# Written via Write tool (not heredoc) to avoid shell-escaping corruption.
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

L=/tmp/mgc.log
S=/tmp/mgc.status
: > "$L"
: > "$S"
ts(){ date '+%F %T'; }
say(){ echo "[$(ts)] $*" >> "$S"; echo "[$(ts)] $*" >> "$L"; }
say "START pid=$$"

retry(){
  name="$1"; marker="$2"; shift 2
  rm -f "$marker"
  for a in 1 2 3 4 5; do
    say "$name attempt $a"
    "$@" >> "$L" 2>&1
    if [ -f "$marker" ]; then say "$name OK"; return 0; fi
    say "$name attempt $a FAILED; sleep 20"
    sleep 20
  done
  say "$name GAVE_UP"
  return 1
}

do_b(){
  python3 scripts/run_text_bridge_baseline.py --game MiniGrid --episodes 12 \
    --ticks 150 --slow-max-tokens 96 --seed 0 --out-dir results/t_trajectories_v2_minigrid
  ls results/t_trajectories_v2_minigrid/MiniGrid_seed*.pt >/dev/null 2>&1 && touch /tmp/mk_b
}
do_c(){
  python3 -m src.training.stage_c_v2 --trace 'results/t_trajectories_v2_minigrid/MiniGrid_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 --stage-a-ckpt checkpoints/stage_a/minigrid_astar.pt \
    --out checkpoints/stage_c/v2_minigrid.pt
  [ -f checkpoints/stage_c/v2_minigrid.pt ] && touch /tmp/mk_c
}
do_eg(){
  python3 -m src.eval.benchmark --strategies F T L --games MiniGrid --seeds 0 1 2 --episodes 4 \
    --max-ticks 200 --fast-ckpt checkpoints/stage_a/minigrid_astar.pt \
    --bridge-ckpt checkpoints/stage_c/v2_minigrid.pt --max-slow-tokens 64 \
    --out results/eval_v2_minigrid_greedy.json
  [ -f results/eval_v2_minigrid_greedy.json ] && touch /tmp/mk_eg
}
do_es(){
  python3 -m src.eval.benchmark --strategies F T L --games MiniGrid --seeds 0 1 2 --episodes 4 \
    --max-ticks 200 --fast-ckpt checkpoints/stage_a/minigrid_astar.pt \
    --bridge-ckpt checkpoints/stage_c/v2_minigrid.pt --max-slow-tokens 64 \
    --action-policy sample --action-temperature 1.0 \
    --out results/eval_v2_minigrid_sample.json
  [ -f results/eval_v2_minigrid_sample.json ] && touch /tmp/mk_es
}

retry StageB /tmp/mk_b do_b || { say "ABORT_B"; exit 1; }
retry StageC /tmp/mk_c do_c || { say "ABORT_C"; exit 1; }
retry EvalGreedy /tmp/mk_eg do_eg || say "WARN_greedy"
retry EvalSample /tmp/mk_es do_es || say "WARN_sample"

python3 -c "
import json,numpy as np
from collections import defaultdict
for tag,p in [('greedy','results/eval_v2_minigrid_greedy.json'),('sample','results/eval_v2_minigrid_sample.json')]:
    try:
        d=json.load(open(p)); by=defaultdict(list)
        for c in d['cells']: by[c['strategy']].append(c['score'])
        msg=tag+' '+' '.join(f'{s}={np.mean(by[s]):.1f}' for s in ['F','T','L'] if by[s])
    except Exception as e:
        msg=tag+' MISSING '+str(e)
    open('$S','a').write('[RESULT] '+msg+chr(10))
"
say "MG_DONE"
