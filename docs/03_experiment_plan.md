# Experiment Plan

## Centerpiece task

**Atari Ms. Pac-Man and Frostbite via the Arcade Learning Environment (ALE).**

Both games require both reflexes (dodge ghosts / jump on moving ice floes) and long-horizon
planning (route through maze / temperature management + igloo return). Both have
well-established RL baselines and instantly recognizable visuals. Frostbite is also a
standard hard-exploration benchmark, providing additional signal.

**Tick rate:** Atari runs at 60Hz. We downsample to **15 Hz nominal** for the fast model
(every 4 frames; 67ms ideal tick). Measured cold-path latency is **~170ms cold / ~140ms
warm** at `max_slice_nums=1` without torch.compile, and ~120-130ms with compile —
realistically a **~7-8 Hz effective rate** until further optimization (vision-tower
caching or lower-res input). The ALE env's frame-skip absorbs this: actions are still
emitted every 4 60Hz frames as far as the game state machine is concerned; the wall-clock
just runs slower than realtime during data generation. The slow model emits at **1-2 Hz**
(every ~15-30 fast ticks).

## Four-strategy comparison

| Strategy | Fast model | Slow model | Channel | Trainable |
|---|---|---|---|---|
| **F** (fast-only) | MiniCPM-o 4.5 | — | none | LoRA on action head |
| **S** (slow-offline) | — | Qwen3-VL-8B-Thinking | — | none (zero-shot prompting) |
| **T** (text bridge) | MiniCPM-o 4.5 | Qwen3-VL-8B-Thinking | text prompts | LoRA on fast |
| **L** (latent bridge) | MiniCPM-o 4.5 | Qwen3-VL-8B-Thinking | continuous vectors | LoRA on both + bridge |

**Why same-size on both sides:** the thesis is about the *channel* (latent vs text), not a
capability gap between the two endpoints. Using two ~8-9B models with the slow side just
operating at a slower tick + with a CoT "thinking" budget isolates the bridge as the
load-bearing architectural choice and avoids the confound "the big model was just better."
Same-size also fits comfortably in 96GB with headroom for PPO at Stage D. Capability
scaling is addressed separately as a single ablation (see below).

**S is a control showing slow-only is too slow:** the slow model produces actions at its
emission rate (1-2Hz), with the game advancing 15+ frames per action. Score will be low
because the game requires faster reactions. This rules out "just use the big model in
real time" as an alternative.

**T is the TML-comparable baseline:** slow model emits text strategic guidance every ~1s;
fast model consumes it as extra context. No LoRA on slow model — pure prompting.

**L is the proposal:** v2 architecture (post-2026-05-16) is the LLaVA-style latent-as-token
bridge — the slow model produces N=8 latent tokens in the fast model's input embedding
space (4096-d), prepended to the fast model's input sequence so all 36 LLM layers attend
through full causal attention. **The v1 design (mid-layer cross-attn into a 256-d ring
buffer at LLM depths 12 & 24) was empirically shown to fail** despite converging to
KL=0.004 on offline data (see `docs/05_status.md` for details). The v2 redesign was
motivated by analysis of how LLaVA / BLIP-2 / MiniCPM-o themselves succeed at multimodal
coupling: text and visual tokens both enter at the LLM's input embedding and share the
attention stack. v2 gives the latent bridge the same architectural privileges. Only the
slow model's `ThoughtProjection` is trainable (~32M params); everything else is frozen.

We also include **O (oracle)**: pre-computed slow-model analysis injected as fast-model
context. Cheating, but bounds the upper limit of what *any* fast/slow coupling can buy.

### Capability-scaling ablation (one configuration)

In addition to the four-strategy main sweep, we run **one** scaling configuration on
Frostbite (Tier 3) only, with Qwen3-30B-A3B-Thinking as the slow model under both T and L.
This is *not* the main result. Its purpose is to show whether the latent-vs-text gap
*grows* when the slow model has more reasoning depth to compress — directly evidencing
the bandwidth claim in the framing doc. Run as `T-30B` and `L-30B` cells; 3 seeds × 30
episodes per cell. Expected: the absolute scores go up under both, but the L–T delta is
*at least as large* as in the 8B–8B configuration, and probably larger.

## Training stages

### Stage A: Behavioral baseline (Week 1)
Fast-only LoRA on MiniCPM-o for action prediction. Imitation learning from random play
seeded with epsilon-greedy exploration, ~100K transitions per game. Establishes F
performance and verifies the fast-side pipeline works.

### Stage B: Text-bridge baseline (Week 2, early)
T condition. No additional training — uses Stage A's fast model with slow-model text
output appended as context. Establishes T performance and provides supervision data for
Stage C.

### Stage C: Bridge supervised training (Week 2)
Run T to generate trajectory data: at each fast tick, record (fast-model context, slow-model
text-thought-so-far, action-taken, reward). Add cross-attention LoRA layers to MiniCPM-o
and a projection LoRA on Qwen3-VL-8B-Thinking that emits continuous thought vectors
from intermediate residual stream.

COCONUT curriculum:
- **Stage C0:** Bridge carries text tokens (replicates T).
- **Stage C1:** First half of slow-model emissions replaced with their corresponding latent
  vectors (LoRA-projected intermediate residual stream). Bridge loss = MSE between
  latent-conditioned fast-model logits and C0 logits at the same positions.
- **Stage C2:** All slow-model emissions replaced with latents. Same supervision.
- **Stage C3 (optional):** Bidirectional — fast model emits perception summaries that
  slow model's LoRA reads.

### Stage D: Joint LoRA co-training on game reward (Week 3)
Both LoRAs fine-tuned via online policy gradient (PPO or GRPO) on game score. Both base
models frozen throughout. Bridge architecture from Stage C2.

### Optional Stage E: Oracle upper bound (Week 3, parallel)
Pre-compute slow-model analyses on a frozen game-state trajectory. Inject directly into
fast model's context as text. Measure score ceiling.

## Evaluation harness

### Primary metrics
- **Mean episode score** (3 seeds × 30 episodes per game per condition)
- **Mean episode length** (survival time)
- **Score at 10K training steps** (sample efficiency)
- **Phase-transition lift L over T** as function of game complexity tier

### Game complexity tiers (for the phase-transition sweep)
- **Tier 1 (low strategic load):** Pong, Breakout, Beam Rider — fast model alone should
  saturate
- **Tier 2 (medium):** Ms. Pac-Man, Space Invaders — moderate planning helps
- **Tier 3 (high):** Frostbite, Montezuma's Revenge (subset), Pitfall — long-horizon
  planning is critical

Prediction: L >> T on Tier 3, L ≈ T on Tier 1.

### Latency / on-clock fraction
Wall-clock per-tick latency for each strategy. Confirms that F, T, L all meet the 67ms
tick budget; S does not.

### Bridge information content (analysis)
Mutual information between thought vectors and (a) optimal action, (b) post-30-tick
reward, on a held-out trajectory set. Diagnostic — confirms the bridge isn't collapsed.

## Four-week timeline

### Week 1 (Days 1-7): Stand up + Stage A
- Day 1-2: Repo setup, model download, verify joint bf16 inference fits
- Day 3-4: ALE environment wrapper, MiniCPM-o action-head LoRA
- Day 5-7: Stage A training on Ms. Pac-Man + Frostbite; verify F-baseline score

**Gate:** F-baseline reaches at least 50% of published DQN scores. If not, debug.

### Week 2 (Days 8-14): Stage B + Stage C0-C2
- Day 8-9: Implement T (slow-model text prompting + fast-model context injection)
- Day 10-11: Generate ~5K trajectories with T; measure T performance
- Day 12-14: Stage C bridge curriculum C0 → C1 → C2

**Gate:** Stage C2's latent-bridge logits match C0's text-bridge logits within 2% KL on
held-out trajectories.

### Week 3 (Days 15-21): Stage D + ablations + Stage E
- Day 15-17: Stage D joint LoRA RL with PPO on game score
- Day 18-19: Run full four-strategy benchmark on 8 games × 3 seeds × 30 episodes
- Day 20-21: Ablations (bridge bandwidth, slow emission rate, staleness tolerance)

**Gate:** L beats T by ≥10% mean score on Tier 3 games, p < 0.05 across seeds.

### Week 4 (Days 22-28): Eval + paper
- Day 22-23: Generate demo video (3-way side-by-side: F, T, L)
- Day 24-26: Write paper draft
- Day 27-28: Polish, citations, README, release prep

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Joint 8B+9B inference doesn't fit | Very low | ~34GB weights; mitigation N/A |
| Same-size slow model too weak to show L–T gap on Tier 3 | Medium | Run the 30B-A3B scaling ablation on Frostbite to disambiguate |
| 30B-A3B scaling ablation doesn't fit alongside 9B fast | Low | MoE → 60GB weights; quantize to AWQ-4bit if needed (saves ~30GB) |
| Fast-only saturates on chosen games | Medium | Pre-screen on Tier 1; if saturated, shift to Tier 3 only |
| Slow model emission too slow | Medium | Reduce emission rate to 0.5Hz; rely on staleness-tolerant buffer |
| Bridge collapses to constant | Medium | MI diagnostic; if collapsed, add reconstruction auxiliary loss |
| RL training unstable | High | Fall back to Stage C supervised only; rely on imitation if PPO diverges |
| Demo doesn't look impressive | Low | Pre-screen 3 games; pick the one with cleanest visual story |

## Resource budget

- **Compute:** RTX Pro 6000 96GB. Continuous use for ~4 weeks.
- **Storage:** ~500GB (model weights, trajectories, checkpoints)
- **API calls:** ~$200 of Gemini API (API keys available in env; use Gemini 3 Flash model) for evaluation judge + scripted-data generation
- **HF/wandb:** free tiers sufficient
