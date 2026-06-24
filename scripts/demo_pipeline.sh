#!/bin/bash
# Generate demo trace + MP4 for one episode of each (game, strategy) on a fixed seed.
# Produces 6 MP4s + 3 side-by-side comparisons in demos/

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

mkdir -p demos traces

echo "[$(ts)] === DEMO PIPELINE START ==="

# Per-game checkpoints
declare -A FAST_CKPT=(
    [MsPacman]=checkpoints/stage_a/mspacman_sb3dqn_v2.pt
    [Seaquest]=checkpoints/stage_a/seaquest_sb3dqn.pt
    [SpaceInvaders]=checkpoints/stage_a/spaceinvaders_sb3dqn.pt
)
declare -A BRIDGE_CKPT=(
    [MsPacman]=checkpoints/stage_c/v2_mspacman.pt
    [Seaquest]=checkpoints/stage_c/v2_seaquest.pt
    [SpaceInvaders]=checkpoints/stage_c/v2_spaceinvaders.pt
)

# Run F + L on each game, one episode each
for GAME in MsPacman Seaquest SpaceInvaders; do
    echo "[$(ts)] === $GAME F + L ==="
    kill_vllm
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
        --strategies F L --games "$GAME" --seeds 0 --episodes 1 \
        --max-ticks 500 \
        --fast-ckpt "${FAST_CKPT[$GAME]}" \
        --bridge-ckpt "${BRIDGE_CKPT[$GAME]}" \
        --max-slow-tokens 64 \
        --save-trace-dir traces/demo \
        --out results/eval_demo_${GAME,,}.json \
        > /tmp/demo_${GAME,,}.log 2>&1
done

echo "[$(ts)] === Rendering MP4s ==="
for GAME in MsPacman Seaquest SpaceInvaders; do
    echo "[$(ts)]   $GAME single panels"
    python3 scripts/render_demo_mp4.py \
        --trace-dir traces/demo/F_${GAME}_seed0 \
        --labels "F (fast only)" \
        --out demos/${GAME,,}_F.mp4 --fps 15 > /tmp/render_${GAME,,}_F.log 2>&1
    python3 scripts/render_demo_mp4.py \
        --trace-dir traces/demo/L_${GAME}_seed0 \
        --labels "L (latent bridge)" \
        --out demos/${GAME,,}_L.mp4 --fps 15 > /tmp/render_${GAME,,}_L.log 2>&1
    echo "[$(ts)]   $GAME side-by-side"
    python3 scripts/render_demo_mp4.py \
        --trace-dir traces/demo/F_${GAME}_seed0 traces/demo/L_${GAME}_seed0 \
        --labels "F (fast only)" "L (latent bridge)" \
        --out demos/${GAME,,}_F_vs_L.mp4 --fps 15 > /tmp/render_${GAME,,}_sxs.log 2>&1
done

echo "[$(ts)] === DEMO PIPELINE COMPLETE ==="
ls -la demos/
