#!/bin/bash
# Re-launch scaling ablation with AWQ-4bit 30B after the download finishes.
# (FP8 failed: transformers can't load block-scale-inv; bf16 failed: OOM
# alongside user's other workloads.)

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

DL_DIR=${HOME}/.cache/huggingface/hub/models--QuantTrio--Qwen3-VL-30B-A3B-Thinking-AWQ
echo "[$(ts)] scaling-awq: waiting for AWQ download (~15-18GB)"
while pgrep -f "huggingface-cli download.*Qwen3-VL-30B-A3B-Thinking-AWQ" >/dev/null; do
    sleep 30
done
SHARDS=$(find $DL_DIR/snapshots -name "*.safetensors" 2>/dev/null | wc -l)
echo "[$(ts)] scaling-awq: download done, $SHARDS shards"

echo "[$(ts)] scaling-awq: launching scaling ablation"
kill_vllm
bash scripts/scaling_ablation_30b.sh > /tmp/scaling_awq.out 2>&1
echo "[$(ts)] scaling-awq: returned $?"
echo "[$(ts)] scaling-awq: COMPLETE"
