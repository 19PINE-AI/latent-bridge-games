#!/bin/bash
# Wait for the 30B-FP8 scaling retry to finish, then run River Raid + RoadRunner.
# Both are bandwidth-claim evidence games (rich strategic state that text loses).

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

MARKER='scaling-retry: COMPLETE'
echo "[$(ts)] rr+roadrunner: waiting for '$MARKER' in /tmp/scaling_retry_watcher.out"
while ! grep -q "$MARKER" /tmp/scaling_retry_watcher.out 2>/dev/null; do
    sleep 60
done
echo "[$(ts)] rr+roadrunner: scaling-retry done; launching River Raid"

bash scripts/riverraid_pipeline.sh > /tmp/rr_pipeline.out 2>&1
echo "[$(ts)] rr+roadrunner: River Raid returned $?"

echo "[$(ts)] rr+roadrunner: launching RoadRunner"
bash scripts/roadrunner_pipeline.sh > /tmp/roadrunner_pipeline.out 2>&1
echo "[$(ts)] rr+roadrunner: RoadRunner returned $?"

echo "[$(ts)] rr+roadrunner: COMPLETE"
