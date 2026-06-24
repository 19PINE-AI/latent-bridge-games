#!/bin/bash
# Larger latency sweep: 12 episodes per cell, F mode, MsPacman, three
# vision_refresh_every values. Supersedes the n=2 spot check.

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
kill_vllm() {
    local pids=$(pgrep -f "VLLM::EngineCore" || true)
    if [ -n "$pids" ]; then
        echo "[$(ts)] killing vLLM workers: $pids"
        kill $pids 2>/dev/null || true
        sleep 5
    fi
}

echo "[$(ts)] === LATENCY SWEEP V2 START ==="

for VRF in 1 4 15; do
    echo "[$(ts)] === F MsPacman vision-refresh-every=$VRF ==="
    kill_vllm
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
        --strategies F --games MsPacman --seeds 0 1 2 --episodes 4 \
        --max-ticks 200 \
        --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
        --vision-refresh-every $VRF \
        --out results/eval_latency_vrf${VRF}_v2.json \
        > /tmp/latency_vrf${VRF}_v2.log 2>&1
done

echo "[$(ts)] === LATENCY SWEEP V2 COMPLETE ==="
