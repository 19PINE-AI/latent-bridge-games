#!/bin/bash
# Master chain: demos (4 new games) -> robust Stage A (3 games) -> PPO on SI.
# ~16h total of sequential GPU work.

set +e
REPO=/home/ubuntu/latent-bridge-games
cd "$REPO"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === MASTER TOP-3 CHAIN START ==="

echo "[$(ts)] === A. Demo MP4s for 4 new games (~1h) ==="
bash scripts/demos_new_games.sh > /tmp/demos_new.out 2>&1
echo "[$(ts)] demos returned $?"

echo "[$(ts)] === B. Robust Stage A for 3 collapsed games (~9h) ==="
bash scripts/robust_stage_a_three.sh > /tmp/robust3.out 2>&1
echo "[$(ts)] robust3 returned $?"

echo "[$(ts)] === C. Stage D PPO on SI (~6h) ==="
bash scripts/ppo_si_with_robust.sh > /tmp/ppo_si.out 2>&1
echo "[$(ts)] ppo_si returned $?"

echo "[$(ts)] === MASTER TOP-3 CHAIN COMPLETE ==="
