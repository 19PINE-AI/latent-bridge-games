#!/bin/bash
# Wait for BOTH (a) the bf16 30B download to complete AND (b) RR+RoadRunner to
# finish, then run the scaling ablation with the bf16 30B (FP8 didn't load).

set +e
REPO=/home/ubuntu/latent-bridge-games
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

DL_DIR=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking

echo "[$(ts)] scaling-bf16: waiting for bf16 download to finish (~60GB)"
# Heuristic: download is done when ALL .safetensors are present AND the
# huggingface-cli download process has exited
while pgrep -f "huggingface-cli download.*Qwen3-VL-30B-A3B-Thinking" >/dev/null 2>&1; do
    sleep 60
done
echo "[$(ts)] scaling-bf16: download process exited"

# Verify the expected files are present
echo "[$(ts)] scaling-bf16: verifying files..."
SHARDS=$(find $DL_DIR/snapshots -name "model-*.safetensors" 2>/dev/null | wc -l)
echo "[$(ts)] scaling-bf16: found $SHARDS safetensor shards"
if [ "$SHARDS" -lt 10 ]; then
    echo "[$(ts)] scaling-bf16: ERROR — expected >=10 shards for 60GB bf16 model"
    exit 1
fi

# Wait for RR + RoadRunner chain
RR_MARKER='rr+roadrunner: COMPLETE'
echo "[$(ts)] scaling-bf16: waiting for '$RR_MARKER' in /tmp/rr_roadrunner_watcher.out"
while ! grep -q "$RR_MARKER" /tmp/rr_roadrunner_watcher.out 2>/dev/null; do
    sleep 60
done
echo "[$(ts)] scaling-bf16: both prerequisites met; launching scaling"

bash scripts/scaling_ablation_30b.sh > /tmp/scaling_bf16.out 2>&1
echo "[$(ts)] scaling-bf16: returned $?"
echo "[$(ts)] scaling-bf16: COMPLETE"
