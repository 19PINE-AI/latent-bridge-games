#!/bin/bash
# Gap-closer chain: cheap wins first (Oracle, latency), then PPO retry (long).
# 30B scaling stays infrastructure-blocked (GPU has 60GB held by user's other
# workloads; bf16 30B needs ~60GB more; FP8/AWQ also failed).

set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
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

echo "[$(ts)] === GAP-CLOSERS CHAIN START ==="

# 1. True Oracle on MsPacman (~30 min)
echo "[$(ts)] === [Oracle] True pre-computed oracle on MsPacman ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_true_oracle.py \
    --game MsPacman --episodes 3 --max-ticks 500 --oracle-tokens 1024 \
    --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --out results/eval_true_oracle_mspacman.json \
    > /tmp/true_oracle_mspacman.log 2>&1

# 2. Larger latency sweep (~30 min)
echo "[$(ts)] === [Latency] 12-episode sweep ==="
bash scripts/latency_sweep_v2.sh > /tmp/latency_v2.out 2>&1

# 3. PPO retry with anti-collapse hyperparams (~6h)
echo "[$(ts)] === [PPO] retry with entropy_coef=0.1 ==="
bash scripts/ppo_si_retry.sh > /tmp/ppo_si_retry.out 2>&1

echo "[$(ts)] === GAP-CLOSERS CHAIN COMPLETE ==="
