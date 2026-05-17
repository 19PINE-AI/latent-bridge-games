#!/bin/bash
# Pong end-to-end (Tier-1 game): tests the H2 strategic-complexity claim
# direction. Predicted: L ~ T because the fast model alone can saturate Pong.
# If L >> T even here, H2's "complexity matters" framing needs revision.

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

EXPERT=$(ls /home/ubuntu/.cache/huggingface/hub/models--sb3--dqn-PongNoFrameskip-v4/snapshots/*/dqn-PongNoFrameskip-v4.zip 2>/dev/null | head -1)
if [ -z "$EXPERT" ]; then
    echo "[$(ts)] downloading SB3 DQN Pong expert..."
    python3 -c "from huggingface_hub import hf_hub_download; \
                p=hf_hub_download('sb3/dqn-PongNoFrameskip-v4', 'dqn-PongNoFrameskip-v4.zip'); \
                print(p)" > /tmp/pong_dl.log 2>&1
    EXPERT=$(ls /home/ubuntu/.cache/huggingface/hub/models--sb3--dqn-PongNoFrameskip-v4/snapshots/*/dqn-PongNoFrameskip-v4.zip | head -1)
fi
echo "[$(ts)] SB3 expert: $EXPERT"

echo "[$(ts)] === PONG PIPELINE START ==="

# 1) SB3-expert collection (CPU)
echo "[$(ts)] === Step 1: SB3 expert collection ==="
CUDA_VISIBLE_DEVICES="" python3 scripts/collect_trajectories.py \
    --game Pong --episodes 10 \
    --expert-policy "$EXPERT" \
    --epsilon 0.1 --max-ticks 2500 \
    --out results/trajectories_Pong_sb3_dqn.pt \
    > /tmp/pong_sb3_collect.log 2>&1

# 2) Stage A behavioral cloning
echo "[$(ts)] === Step 2: Stage A ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_a_behavioral \
    --traj results/trajectories_Pong_sb3_dqn.pt \
    --epochs 3 --batch-size 1 --grad-accum 8 --lr 1e-4 --val-fraction 0.1 \
    --out checkpoints/stage_a/pong_sb3dqn.pt \
    > /tmp/pong_stage_a.log 2>&1

# 3) T-trajectories
echo "[$(ts)] === Step 3: T-trajectory collection ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_text_bridge_baseline.py \
    --game Pong --episodes 10 --ticks 750 --slow-max-tokens 96 --seed 0 \
    --out-dir results/t_trajectories_v2 \
    > /tmp/pong_t_collect.log 2>&1

# 4) Stage C v2
echo "[$(ts)] === Step 4: Stage C v2 ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2/Pong_seed*.pt' \
    --epochs 1 --grad-accum 4 --lr 5e-5 \
    --stage-a-ckpt checkpoints/stage_a/pong_sb3dqn.pt \
    --out checkpoints/stage_c/v2_pong.pt \
    > /tmp/pong_stage_c.log 2>&1

# 5) F/T/L eval
echo "[$(ts)] === Step 5: F/T/L eval ==="
kill_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.eval.benchmark \
    --strategies F T L --games Pong --seeds 0 1 2 --episodes 4 \
    --max-ticks 1500 \
    --fast-ckpt checkpoints/stage_a/pong_sb3dqn.pt \
    --bridge-ckpt checkpoints/stage_c/v2_pong.pt \
    --max-slow-tokens 64 \
    --out results/eval_v2_pong.json \
    > /tmp/pong_eval.log 2>&1

echo "[$(ts)] === PONG PIPELINE COMPLETE ==="
