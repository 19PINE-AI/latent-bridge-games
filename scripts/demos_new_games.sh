#!/bin/bash
# Generate demo MP4s for the 4 newer games (RoadRunner, RR, Enduro, Qbert) so the
# website + demos directory matches the result table.

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

# Per-game checkpoints
declare -A FAST_CKPT=(
    [RoadRunner]=checkpoints/stage_a/roadrunner_sb3dqn.pt
    [Riverraid]=checkpoints/stage_a/riverraid_sb3dqn.pt
    [Enduro]=checkpoints/stage_a/enduro_ppo.pt
    [Qbert]=checkpoints/stage_a/qbert_sb3dqn.pt
)
declare -A BRIDGE_CKPT=(
    [RoadRunner]=checkpoints/stage_c/v2_roadrunner.pt
    [Riverraid]=checkpoints/stage_c/v2_riverraid.pt
    [Enduro]=checkpoints/stage_c/v2_enduro.pt
    [Qbert]=checkpoints/stage_c/v2_qbert.pt
)

echo "[$(ts)] === NEW-GAMES DEMO PIPELINE START ==="

for GAME in RoadRunner Riverraid Enduro Qbert; do
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
for GAME in RoadRunner Riverraid Enduro Qbert; do
    LOWER=${GAME,,}
    echo "[$(ts)]   $GAME single + side-by-side"
    python3 scripts/render_demo_mp4.py \
        --trace-dir traces/demo/F_${GAME}_seed0 \
        --labels "F (fast only)" \
        --out demos/${LOWER}_F.mp4 --fps 15 > /tmp/render_${LOWER}_F.log 2>&1
    python3 scripts/render_demo_mp4.py \
        --trace-dir traces/demo/L_${GAME}_seed0 \
        --labels "L (latent bridge)" \
        --out demos/${LOWER}_L.mp4 --fps 15 > /tmp/render_${LOWER}_L.log 2>&1
    python3 scripts/render_demo_mp4.py \
        --trace-dir traces/demo/F_${GAME}_seed0 traces/demo/L_${GAME}_seed0 \
        --labels "F (fast only)" "L (latent bridge)" \
        --out demos/${LOWER}_F_vs_L.mp4 --fps 15 > /tmp/render_${LOWER}_sxs.log 2>&1
done

echo "[$(ts)] === NEW-GAMES DEMO PIPELINE COMPLETE ==="
ls -la demos/
