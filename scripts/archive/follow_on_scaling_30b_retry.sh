#!/bin/bash
# Wait for the Pong + PPO-smoke follow-on to finish, then re-launch the 30B-FP8
# scaling ablation with the load fix from commit 9b59d24.

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

MARKER='follow-on: COMPLETE'
echo "[$(ts)] scaling-retry: waiting for '$MARKER' in /tmp/followon.out"
while ! grep -q "$MARKER" /tmp/followon.out 2>/dev/null; do
    sleep 60
done
echo "[$(ts)] scaling-retry: Pong/PPO-smoke chain finished; launching 30B-FP8 scaling"

bash scripts/scaling_ablation_30b.sh > /tmp/scaling_30b_retry.out 2>&1
echo "[$(ts)] scaling-retry: returned exit $?"
echo "[$(ts)] scaling-retry: COMPLETE"
