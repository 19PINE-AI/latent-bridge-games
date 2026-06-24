#!/bin/bash
# Run Enduro + Qbert after the 30B-AWQ scaling completes.

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

MARKER='scaling-awq: COMPLETE'
echo "[$(ts)] enduro+qbert: waiting for '$MARKER' in /tmp/scaling_awq_watcher.out"
while ! grep -q "$MARKER" /tmp/scaling_awq_watcher.out 2>/dev/null; do
    sleep 60
done
echo "[$(ts)] enduro+qbert: scaling-awq done; launching Enduro"

bash scripts/enduro_pipeline.sh > /tmp/enduro_pipeline.out 2>&1
echo "[$(ts)] enduro+qbert: Enduro returned $?"

echo "[$(ts)] enduro+qbert: launching Qbert"
bash scripts/qbert_pipeline.sh > /tmp/qbert_pipeline.out 2>&1
echo "[$(ts)] enduro+qbert: Qbert returned $?"

echo "[$(ts)] enduro+qbert: COMPLETE"
