#!/bin/bash
# Wait for the Phase 2/3 chain to finish, then run Pong pipeline and PPO smoke.
# Intended to be backgrounded so the next set of experiments fires automatically.

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Wait for the chain's completion marker
MARKER='MULTI-PHASE OVERNIGHT PIPELINE COMPLETE'
echo "[$(ts)] follow-on: waiting for '$MARKER' in /tmp/phase2_pipeline.out"
while ! grep -q "$MARKER" /tmp/phase2_pipeline.out 2>/dev/null; do
    sleep 60
done
echo "[$(ts)] follow-on: chain finished; launching Pong pipeline"

bash scripts/pong_pipeline.sh
echo "[$(ts)] follow-on: Pong pipeline returned exit $?"

echo "[$(ts)] follow-on: launching PPO smoke test (MsPacman, 16-tick rollout)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_d_rl \
    --game MsPacman --total-updates 1 --rollout-len 16 \
    --stage-a-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --stage-c-v2-ckpt checkpoints/stage_c/v2_mspacman.pt \
    --out checkpoints/stage_d/ppo_smoke.pt \
    --smoke-test \
    > /tmp/ppo_smoke.log 2>&1
echo "[$(ts)] follow-on: PPO smoke returned exit $?"

echo "[$(ts)] follow-on: COMPLETE"
