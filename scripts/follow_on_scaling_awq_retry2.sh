#!/bin/bash
# Re-run scaling-AWQ after enduro+qbert finishes (with the device_map fix).

set +e
REPO=/home/ubuntu/latent-bridge-games
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

MARKER='enduro+qbert: COMPLETE'
echo "[$(ts)] scaling-awq-retry2: waiting for '$MARKER' in /tmp/enduro_qbert_watcher.out"
while ! grep -q "$MARKER" /tmp/enduro_qbert_watcher.out 2>/dev/null; do
    sleep 60
done

echo "[$(ts)] scaling-awq-retry2: launching scaling ablation (AWQ)"
kill_vllm
bash scripts/scaling_ablation_30b.sh > /tmp/scaling_awq_retry2.out 2>&1
echo "[$(ts)] scaling-awq-retry2: returned $?"
echo "[$(ts)] scaling-awq-retry2: COMPLETE"
