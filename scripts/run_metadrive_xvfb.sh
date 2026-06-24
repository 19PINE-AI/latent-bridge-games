#!/bin/bash
# MetaDrive end-to-end, every MetaDrive-touching stage wrapped in `xvfb-run -a`
# (the ONLY verified-working render path). Stages that don't touch MetaDrive
# (Stage A BC, Stage C bridge) read cached frames and run without xvfb.
# Self-heals corrupt main Stage A checkpoints from the per-epoch files.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"
export LB_FAST_MODEL_PATH=${REPO}/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=${REPO}/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Hardware GLX under Xvfb (VERIFIED 15.6 steps/sec). NVIDIA GLX libs now installed;
# this env var makes GLX use them instead of Mesa swrast (which was 38 s/step).
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export PYTHONPATH=${REPO}

S=/tmp/mdx.status; L=/tmp/mdx.log; : > "$S"; : > "$L"
say(){ echo "[$(date '+%H:%M:%S')] $*" >> "$S"; }
say "START pid=$$"

heal(){ python3 - >> "$L" 2>&1 <<'PY'
import os, shutil, glob
for base in ['metadrive_bare','metadrive_robust']:
    main=f'checkpoints/stage_a/{base}.pt'
    eps=sorted(glob.glob(f'checkpoints/stage_a/{base}_ep*.pt'))
    if eps and (not os.path.exists(main) or os.path.getsize(main)<10000):
        shutil.copyfile(eps[-1], main); print("healed", base, os.path.getsize(main))
PY
}

retry(){ name="$1"; marker="$2"; shift 2; rm -f "$marker"
  for a in 1 2 3 4 5; do say "$name attempt $a"; "$@" >> "$L" 2>&1
    [ -f "$marker" ] && { say "$name OK"; return 0; }
    say "$name attempt $a FAILED sleep 15"; sleep 15; done
  say "$name GAVE_UP"; return 1; }

# 1. Collect (MetaDrive -> xvfb)
collect(){ xvfb-run -a -s "-screen 0 256x256x24" python3 scripts/collect_metadrive_expert.py \
    --episodes 25 --seed 0 --max-ticks 200 --out results/trajectories_MetaDrive.pt
  [ -f results/trajectories_MetaDrive.pt ] && touch /tmp/mk_x_data; }
retry Collect /tmp/mk_x_data collect || { echo "FAIL collect" >> "$S"; exit 1; }

# 2a. Stage A bare (no MetaDrive -> no xvfb)
sa(){ python3 -m src.training.stage_a_behavioral --traj results/trajectories_MetaDrive.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --out checkpoints/stage_a/metadrive_bare.pt; heal
  [ -f checkpoints/stage_a/metadrive_bare.pt ] && [ $(stat -c%s checkpoints/stage_a/metadrive_bare.pt) -gt 10000 ] && touch /tmp/mk_x_sa; }
retry StageA_bare /tmp/mk_x_sa sa || { echo "FAIL stageA" >> "$S"; exit 1; }

# 2b. Stage A robust
sar(){ python3 -m src.training.stage_a_behavioral --traj results/trajectories_MetaDrive.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 --suffix-prob 0.5 \
    --out checkpoints/stage_a/metadrive_robust.pt; heal
  [ -f checkpoints/stage_a/metadrive_robust.pt ] && [ $(stat -c%s checkpoints/stage_a/metadrive_robust.pt) -gt 10000 ] && touch /tmp/mk_x_sar; }
retry StageA_robust /tmp/mk_x_sar sar || say "WARN robust SA failed"

# 3. Stage B (MetaDrive + slow model -> xvfb)
sb(){ xvfb-run -a -s "-screen 0 256x256x24" python3 scripts/run_text_bridge_baseline.py \
    --game MetaDrive --episodes 12 --ticks 200 --slow-max-tokens 96 --seed 0 \
    --out-dir results/t_trajectories_v2_metadrive
  ls results/t_trajectories_v2_metadrive/MetaDrive_seed*.pt >/dev/null 2>&1 && touch /tmp/mk_x_b; }
retry StageB /tmp/mk_x_b sb || { echo "FAIL stageB" >> "$S"; exit 1; }

# 4. Stage C (no MetaDrive -> no xvfb), converged 4 epochs
sc(){ python3 -m src.training.stage_c_v2 --trace 'results/t_trajectories_v2_metadrive/MetaDrive_seed*.pt' \
    --epochs 4 --grad-accum 4 --lr 5e-5 --stage-a-ckpt checkpoints/stage_a/metadrive_bare.pt \
    --out checkpoints/stage_c/v2_metadrive.pt
  [ -f checkpoints/stage_c/v2_metadrive.pt ] && touch /tmp/mk_x_c; }
retry StageC /tmp/mk_x_c sc || { echo "FAIL stageC" >> "$S"; exit 1; }

# 5. Eval (MetaDrive -> xvfb). $1 fast_ckpt $2 policy $3 temp $4 out $5 marker
evl(){ rm -f "$5"
  xvfb-run -a -s "-screen 0 256x256x24" python3 -m src.eval.benchmark --strategies F T L \
    --games MetaDrive --seeds 0 1 2 --episodes 2 --max-ticks 250 \
    --fast-ckpt "$1" --bridge-ckpt checkpoints/stage_c/v2_metadrive.pt \
    --max-slow-tokens 64 --action-policy "$2" --action-temperature "$3" --out "$4"
  [ -f "$4" ] && touch "$5"; }

retry EvalBareGreedy /tmp/mk_x_eg evl checkpoints/stage_a/metadrive_bare.pt argmax 1.0 results/eval_v2_metadrive_bare_greedy.json /tmp/mk_x_eg || say "WARN bg"
retry EvalBareSample /tmp/mk_x_es evl checkpoints/stage_a/metadrive_bare.pt sample 1.0 results/eval_v2_metadrive_bare_sample.json /tmp/mk_x_es || say "WARN bs"
if [ -f checkpoints/stage_a/metadrive_robust.pt ]; then
  retry EvalRobustGreedy /tmp/mk_x_erg evl checkpoints/stage_a/metadrive_robust.pt argmax 1.0 results/eval_v2_metadrive_robust_greedy.json /tmp/mk_x_erg || say "WARN rg"
  retry EvalRobustSample /tmp/mk_x_ers evl checkpoints/stage_a/metadrive_robust.pt sample 1.0 results/eval_v2_metadrive_robust_sample.json /tmp/mk_x_ers || say "WARN rs"
fi

python3 - >> "$S" 2>&1 <<'PY'
import json, numpy as np, os, re
from collections import defaultdict
for tag,p in [("bare_greedy","results/eval_v2_metadrive_bare_greedy.json"),
              ("bare_sample","results/eval_v2_metadrive_bare_sample.json"),
              ("robust_greedy","results/eval_v2_metadrive_robust_greedy.json"),
              ("robust_sample","results/eval_v2_metadrive_robust_sample.json")]:
    try:
        d=json.load(open(p)); by=defaultdict(list)
        for c in d["cells"]: by[c["strategy"]].append(c["score"])
        line="[RESULT] "+tag+" "+" ".join(f"{s}={np.mean(by[s]):.1f}+-{np.std(by[s],ddof=1) if len(by[s])>1 else 0:.1f}" for s in ["F","T","L"] if by[s])
    except Exception: line="[RESULT] "+tag+" MISSING"
    open("/tmp/mdx.status","a").write(line+"\n")
kl=re.findall(r'mean KL=([\d.]+)', open('/tmp/mdx.log').read()) if os.path.exists('/tmp/mdx.log') else []
open("/tmp/mdx.status","a").write("[STAGEC_KL] "+str(kl)+"\n")
PY
say "MDX_DONE"
