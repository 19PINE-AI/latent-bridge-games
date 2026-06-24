#!/bin/bash
# After gap-closers chain finishes, retry the True Oracle with a smaller
# budget (256 tokens; 1024 OOM'd at 60+ GB user load + 35 GB joint models).

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

MARKER='GAP-CLOSERS CHAIN COMPLETE'
echo "[$(ts)] oracle-retry: waiting for '$MARKER' in /tmp/gap_closers.out"
while ! grep -q "$MARKER" /tmp/gap_closers.out 2>/dev/null; do
    sleep 60
done
echo "[$(ts)] oracle-retry: gap-closers done; launching Oracle (256 tokens)"

kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_true_oracle.py \
    --game MsPacman --episodes 3 --max-ticks 500 --oracle-tokens 256 \
    --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --out results/eval_true_oracle_mspacman.json \
    > /tmp/true_oracle_retry.log 2>&1
echo "[$(ts)] oracle-retry: returned $?"
echo "[$(ts)] oracle-retry: COMPLETE"
