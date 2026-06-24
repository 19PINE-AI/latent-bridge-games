#!/bin/bash
# Full MetaDrive pipeline: persistent Xvfb -> collect expert -> Stage A (bare+robust)
# -> Stage B (T-traj) -> Stage C (converged) -> F/T/L eval (greedy+sample, bare+robust).
# Models loaded from local folders (bypass HF cache). Verify via files, retries per stage.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"

# --- persistent virtual display for MetaDrive/Panda3D GL ---
Xvfb :99 -screen 0 256x256x24 >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!
export DISPLAY=:99
sleep 3

export LB_FAST_MODEL_PATH=${REPO}/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=${REPO}/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

S=/tmp/md_pipe.status; L=/tmp/md_pipe.log; : > "$S"; : > "$L"
say(){ echo "[$(date '+%H:%M:%S')] $*" >> "$S"; }
say "START pid=$$ xvfb=$XVFB_PID DISPLAY=$DISPLAY"

retry(){ name="$1"; marker="$2"; shift 2; rm -f "$marker"
  for a in 1 2 3 4 5; do say "$name attempt $a"; "$@" >> "$L" 2>&1
    [ -f "$marker" ] && { say "$name OK"; return 0; }
    say "$name attempt $a FAILED sleep 20"; sleep 20; done
  say "$name GAVE_UP"; return 1; }

# 1. Collect expert trajectories (Stage A data)
collect(){ python3 scripts/collect_metadrive_expert.py --episodes 60 --seed 0 --max-ticks 500 \
    --out results/trajectories_MetaDrive.pt
  [ -f results/trajectories_MetaDrive.pt ] && touch /tmp/mk_md_data; }
retry CollectExpert /tmp/mk_md_data collect || { echo "FAIL collect" >> "$S"; kill $XVFB_PID; exit 1; }

# 2a. Stage A bare
sa_bare(){ python3 -m src.training.stage_a_behavioral --traj results/trajectories_MetaDrive.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --out checkpoints/stage_a/metadrive_bare.pt
  [ -f checkpoints/stage_a/metadrive_bare.pt ] && touch /tmp/mk_md_sa; }
retry StageA_bare /tmp/mk_md_sa sa_bare || { echo "FAIL stageA" >> "$S"; kill $XVFB_PID; exit 1; }

# 2b. Stage A robust (suffix-augmented, to counter OOD-brittleness)
sa_robust(){ python3 -m src.training.stage_a_behavioral --traj results/trajectories_MetaDrive.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 --suffix-prob 0.5 \
    --out checkpoints/stage_a/metadrive_robust.pt
  [ -f checkpoints/stage_a/metadrive_robust.pt ] && touch /tmp/mk_md_sar; }
retry StageA_robust /tmp/mk_md_sar sa_robust || say "WARN robust SA failed"

# 3. Stage B T-trajectories (slow model in loop)
sb(){ python3 scripts/run_text_bridge_baseline.py --game MetaDrive --episodes 16 \
    --ticks 400 --slow-max-tokens 96 --seed 0 --out-dir results/t_trajectories_v2_metadrive
  ls results/t_trajectories_v2_metadrive/MetaDrive_seed*.pt >/dev/null 2>&1 && touch /tmp/mk_md_b; }
retry StageB /tmp/mk_md_b sb || { echo "FAIL stageB" >> "$S"; kill $XVFB_PID; exit 1; }

# 4. Stage C converged (more epochs for low KL)
sc(){ python3 -m src.training.stage_c_v2 --trace 'results/t_trajectories_v2_metadrive/MetaDrive_seed*.pt' \
    --epochs 3 --grad-accum 4 --lr 5e-5 --stage-a-ckpt checkpoints/stage_a/metadrive_bare.pt \
    --out checkpoints/stage_c/v2_metadrive.pt
  [ -f checkpoints/stage_c/v2_metadrive.pt ] && touch /tmp/mk_md_c; }
retry StageC /tmp/mk_md_c sc || { echo "FAIL stageC" >> "$S"; kill $XVFB_PID; exit 1; }

# 5. Eval F/T/L, greedy + sample, BARE Stage A
eg_bare(){ python3 -m src.eval.benchmark --strategies F T L --games MetaDrive --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 --fast-ckpt checkpoints/stage_a/metadrive_bare.pt \
    --bridge-ckpt checkpoints/stage_c/v2_metadrive.pt --max-slow-tokens 64 \
    --out results/eval_v2_metadrive_bare_greedy.json
  [ -f results/eval_v2_metadrive_bare_greedy.json ] && touch /tmp/mk_md_eg; }
retry EvalBareGreedy /tmp/mk_md_eg eg_bare || say "WARN bare greedy failed"

es_bare(){ python3 -m src.eval.benchmark --strategies F T L --games MetaDrive --seeds 0 1 2 --episodes 4 \
    --max-ticks 500 --fast-ckpt checkpoints/stage_a/metadrive_bare.pt \
    --bridge-ckpt checkpoints/stage_c/v2_metadrive.pt --max-slow-tokens 64 \
    --action-policy sample --action-temperature 1.0 \
    --out results/eval_v2_metadrive_bare_sample.json
  [ -f results/eval_v2_metadrive_bare_sample.json ] && touch /tmp/mk_md_es; }
retry EvalBareSample /tmp/mk_md_es es_bare || say "WARN bare sample failed"

# 6. Eval with ROBUST Stage A (if it trained)
if [ -f checkpoints/stage_a/metadrive_robust.pt ]; then
  er_g(){ python3 -m src.eval.benchmark --strategies F T L --games MetaDrive --seeds 0 1 2 --episodes 4 \
      --max-ticks 500 --fast-ckpt checkpoints/stage_a/metadrive_robust.pt \
      --bridge-ckpt checkpoints/stage_c/v2_metadrive.pt --max-slow-tokens 64 \
      --out results/eval_v2_metadrive_robust_greedy.json
    [ -f results/eval_v2_metadrive_robust_greedy.json ] && touch /tmp/mk_md_erg; }
  retry EvalRobustGreedy /tmp/mk_md_erg er_g || say "WARN robust greedy failed"
fi

# summary
python3 - >> "$S" 2>&1 <<'PY'
import json, numpy as np
from collections import defaultdict
for tag,p in [("bare_greedy","results/eval_v2_metadrive_bare_greedy.json"),
              ("bare_sample","results/eval_v2_metadrive_bare_sample.json"),
              ("robust_greedy","results/eval_v2_metadrive_robust_greedy.json")]:
    try:
        d=json.load(open(p)); by=defaultdict(list)
        for c in d["cells"]: by[c["strategy"]].append(c["score"])
        line="[RESULT] "+tag+" "+" ".join(f"{s}={np.mean(by[s]):.1f}+-{np.std(by[s],ddof=1) if len(by[s])>1 else 0:.1f}" for s in ["F","T","L"] if by[s])
    except Exception as e: line="[RESULT] "+tag+" MISSING "+str(e)
    open("/tmp/md_pipe.status","a").write(line+"\n")
PY
say "MD_PIPE_DONE"
kill $XVFB_PID 2>/dev/null
