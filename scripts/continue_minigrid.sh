#!/bin/bash
# Robust continuation: wait for Stage B output -> Stage C -> eval greedy -> eval sample.
# 5 retries per stage to ride out intermittent file/cache inconsistency.
cd /home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=/tmp/minigrid_cont.log
S=/tmp/minigrid_cont.status
: > "$LOG"; ts(){ date '+%F %T'; }; say(){ echo "[$(ts)] $*" | tee -a "$LOG" "$S"; }
say "CONTINUE_START"

# 1. Wait for Stage B output (up to 40 min)
say "waiting for Stage B trajectories..."
for i in $(seq 1 80); do
  n=$(ls results/t_trajectories_v2_minigrid/MiniGrid_seed*.pt 2>/dev/null | wc -l)
  if [ "$n" -ge 1 ] && ! pgrep -f "run_text_bridge_baseline.py --game MiniGrid" >/dev/null; then
    say "Stage B done: $n trajectory file(s)"; break
  fi
  sleep 30
done

retry(){ local name="$1" marker="$2"; shift 2
  for a in 1 2 3 4 5; do
    say "$name attempt $a"; rm -f "$marker"; "$@"
    [ -f "$marker" ] && { say "$name OK"; return 0; }
    say "$name attempt $a failed; sleep 30"; sleep 30
  done
  say "$name FAILED after 5"; return 1; }

stage_c(){ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
  --trace 'results/t_trajectories_v2_minigrid/MiniGrid_seed*.pt' --epochs 1 --grad-accum 4 --lr 5e-5 \
  --stage-a-ckpt checkpoints/stage_a/minigrid_astar.pt --out checkpoints/stage_c/v2_minigrid.pt >>"$LOG" 2>&1
  [ -f checkpoints/stage_c/v2_minigrid.pt ] && touch /tmp/mk_c; }
retry "StageC" /tmp/mk_c stage_c || { echo "FAILED StageC" >>"$S"; exit 1; }

eg(){ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
  --strategies F T L --games MiniGrid --seeds 0 1 2 --episodes 4 --max-ticks 200 \
  --fast-ckpt checkpoints/stage_a/minigrid_astar.pt --bridge-ckpt checkpoints/stage_c/v2_minigrid.pt \
  --max-slow-tokens 64 --out results/eval_v2_minigrid_greedy.json >>"$LOG" 2>&1
  [ -f results/eval_v2_minigrid_greedy.json ] && touch /tmp/mk_eg; }
retry "EvalGreedy" /tmp/mk_eg eg || say "WARN EvalGreedy failed"

es(){ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
  --strategies F T L --games MiniGrid --seeds 0 1 2 --episodes 4 --max-ticks 200 \
  --fast-ckpt checkpoints/stage_a/minigrid_astar.pt --bridge-ckpt checkpoints/stage_c/v2_minigrid.pt \
  --max-slow-tokens 64 --action-policy sample --action-temperature 1.0 \
  --out results/eval_v2_minigrid_sample.json >>"$LOG" 2>&1
  [ -f results/eval_v2_minigrid_sample.json ] && touch /tmp/mk_es; }
retry "EvalSample" /tmp/mk_es es || say "WARN EvalSample failed"

python3 - >>"$S" 2>&1 <<'PY'
import json, numpy as np
from collections import defaultdict
print("=== MiniGrid RESULTS ===")
for tag,p in [("greedy","results/eval_v2_minigrid_greedy.json"),("sample","results/eval_v2_minigrid_sample.json")]:
    try:
        d=json.load(open(p)); by=defaultdict(list)
        for c in d["cells"]: by[c["strategy"]].append(c["score"])
        print(tag+":", {s:f"{np.mean(by[s]):.1f}±{np.std(by[s],ddof=1) if len(by[s])>1 else 0:.1f} (n={len(by[s])})" for s in ["F","T","L"] if by[s]})
    except Exception as e: print(tag,"MISSING",e)
PY
say "CONTINUE_DONE"
