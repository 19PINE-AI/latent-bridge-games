#!/bin/bash
# Phase 8 follow-on: run robust Stage A on the 3 winning games after the
# current master top-3 chain (demos -> robust × 3 collapsed -> PPO SI) finishes.

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

MARKER='MASTER TOP-3 CHAIN COMPLETE'
echo "[$(ts)] phase8: waiting for '$MARKER' in /tmp/master_top3.out"
while ! grep -q "$MARKER" /tmp/master_top3.out 2>/dev/null; do
    sleep 60
done
echo "[$(ts)] phase8: master chain done; launching robust Stage A on winners"

bash scripts/robust_stage_a_winners.sh > /tmp/robust_winners.out 2>&1
echo "[$(ts)] phase8: returned $?"
echo "[$(ts)] phase8: COMPLETE"
