#!/bin/bash
# MetaDrive e2e via hardware EGL (baked into metadrive_wrapper.py, 120 steps/sec).
# No xvfb. collect -> StageA(bare+robust) -> B -> C(converged) -> eval(greedy+sample, bare+robust).
cd /home/ubuntu/latent-bridge-games
export PYTHONPATH=/home/ubuntu/latent-bridge-games
export LB_FAST_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/MiniCPM-o-4_5
export LB_SLOW_MODEL_PATH=/home/ubuntu/latent-bridge-games/local_models/Qwen3-VL-8B-Thinking
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
S=/tmp/mg.status; L=/tmp/mg.log; : > "$S"; : > "$L"
say(){ echo "[$(date '+%H:%M:%S')] $*" >> "$S"; }
say "START pid=$$"
heal(){ python3 - >> "$L" 2>&1 <<'PY'
import os,shutil,glob
for b in ['metadrive_bare','metadrive_robust']:
    m=f'checkpoints/stage_a/{b}.pt'; e=sorted(glob.glob(f'checkpoints/stage_a/{b}_ep*.pt'))
    if e and (not os.path.exists(m) or os.path.getsize(m)<10000): shutil.copyfile(e[-1],m); print("healed",b)
PY
}
retry(){ name="$1"; mk="$2"; shift 2; rm -f "$mk"
  for a in 1 2 3 4 5; do say "$name attempt $a"; "$@" >> "$L" 2>&1
    [ -f "$mk" ] && { say "$name OK"; return 0; }; say "$name attempt $a FAILED sleep 15"; sleep 15; done
  say "$name GAVE_UP"; return 1; }
collect(){ python3 scripts/collect_metadrive_expert.py --episodes 60 --seed 0 --max-ticks 500 --out results/trajectories_MetaDrive.pt
  [ -f results/trajectories_MetaDrive.pt ] && touch /tmp/mk_g_data; }
retry Collect /tmp/mk_g_data collect || { echo "FAIL collect">>"$S"; exit 1; }
sa(){ python3 -m src.training.stage_a_behavioral --traj results/trajectories_MetaDrive.pt --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 --out checkpoints/stage_a/metadrive_bare.pt; heal
  [ -f checkpoints/stage_a/metadrive_bare.pt ] && [ $(stat -c%s checkpoints/stage_a/metadrive_bare.pt) -gt 10000 ] && touch /tmp/mk_g_sa; }
retry StageA_bare /tmp/mk_g_sa sa || { echo "FAIL sa">>"$S"; exit 1; }
sar(){ python3 -m src.training.stage_a_behavioral --traj results/trajectories_MetaDrive.pt --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 --suffix-prob 0.5 --out checkpoints/stage_a/metadrive_robust.pt; heal
  [ -f checkpoints/stage_a/metadrive_robust.pt ] && [ $(stat -c%s checkpoints/stage_a/metadrive_robust.pt) -gt 10000 ] && touch /tmp/mk_g_sar; }
retry StageA_robust /tmp/mk_g_sar sar || say "WARN robust sa"
sb(){ python3 scripts/run_text_bridge_baseline.py --game MetaDrive --episodes 16 --ticks 400 --slow-max-tokens 96 --seed 0 --out-dir results/t_trajectories_v2_metadrive
  ls results/t_trajectories_v2_metadrive/MetaDrive_seed*.pt >/dev/null 2>&1 && touch /tmp/mk_g_b; }
retry StageB /tmp/mk_g_b sb || { echo "FAIL sb">>"$S"; exit 1; }
sc(){ python3 -m src.training.stage_c_v2 --trace 'results/t_trajectories_v2_metadrive/MetaDrive_seed*.pt' --epochs 4 --grad-accum 4 --lr 5e-5 --stage-a-ckpt checkpoints/stage_a/metadrive_bare.pt --out checkpoints/stage_c/v2_metadrive.pt
  [ -f checkpoints/stage_c/v2_metadrive.pt ] && touch /tmp/mk_g_c; }
retry StageC /tmp/mk_g_c sc || { echo "FAIL sc">>"$S"; exit 1; }
evl(){ rm -f "$5"; python3 -m src.eval.benchmark --strategies F T L --games MetaDrive --seeds 0 1 2 --episodes 4 --max-ticks 500 --fast-ckpt "$1" --bridge-ckpt checkpoints/stage_c/v2_metadrive.pt --max-slow-tokens 64 --action-policy "$2" --action-temperature "$3" --out "$4"; [ -f "$4" ] && touch "$5"; }
retry EvalBG /tmp/mk_g_eg evl checkpoints/stage_a/metadrive_bare.pt argmax 1.0 results/eval_v2_metadrive_bare_greedy.json /tmp/mk_g_eg || say "WARN bg"
retry EvalBS /tmp/mk_g_es evl checkpoints/stage_a/metadrive_bare.pt sample 1.0 results/eval_v2_metadrive_bare_sample.json /tmp/mk_g_es || say "WARN bs"
if [ -f checkpoints/stage_a/metadrive_robust.pt ]; then
  retry EvalRG /tmp/mk_g_erg evl checkpoints/stage_a/metadrive_robust.pt argmax 1.0 results/eval_v2_metadrive_robust_greedy.json /tmp/mk_g_erg || say "WARN rg"
  retry EvalRS /tmp/mk_g_ers evl checkpoints/stage_a/metadrive_robust.pt sample 1.0 results/eval_v2_metadrive_robust_sample.json /tmp/mk_g_ers || say "WARN rs"
fi
python3 - >> "$S" 2>&1 <<'PY'
import json,numpy as np,os,re
from collections import defaultdict
for tag,p in [("bare_greedy","results/eval_v2_metadrive_bare_greedy.json"),("bare_sample","results/eval_v2_metadrive_bare_sample.json"),("robust_greedy","results/eval_v2_metadrive_robust_greedy.json"),("robust_sample","results/eval_v2_metadrive_robust_sample.json")]:
    try:
        d=json.load(open(p)); by=defaultdict(list)
        for c in d["cells"]: by[c["strategy"]].append(c["score"])
        open("/tmp/mg.status","a").write("[RESULT] "+tag+" "+" ".join(f"{s}={np.mean(by[s]):.1f}+-{np.std(by[s],ddof=1) if len(by[s])>1 else 0:.1f}" for s in ["F","T","L"] if by[s])+"\n")
    except Exception: open("/tmp/mg.status","a").write("[RESULT] "+tag+" MISSING\n")
kl=re.findall(r'mean KL=([\d.]+)', open('/tmp/mg.log').read()) if os.path.exists('/tmp/mg.log') else []
open("/tmp/mg.status","a").write("[STAGEC_KL] "+str(kl)+"\n")
PY
say "MG_DONE"
