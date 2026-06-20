# Experiment Plan

> **Status update (2026-05-16):** The plan originally targeted Frostbite as the
> Tier-3 game. After Frostbite Stage A failed (no SB3-zoo expert; RWR self-bootstrap on
> random-policy trajectories produced val_acc ≈ uniform random), **Seaquest** was
> substituted as the Tier-3 game (SB3 DQN expert available, 18-action ALE-canonical
> space, oxygen+diver-collection planning load). SpaceInvaders was added as a second
> Tier-2 data point. Frostbite remains future work (needs scratch-trained DQN or a
> scripted heuristic). See `docs/06_results.md` for current headline numbers.

## Centerpiece task

**Atari Ms. Pac-Man (Tier 2) and Seaquest (Tier 3) via the Arcade Learning Environment
(ALE).** SpaceInvaders (Tier 2) added as a second data point.

All games require both reflexes (dodge ghosts / dodge bombs / surface for oxygen) and
long-horizon planning (route through maze / clear invader rows / diver-collection +
oxygen budget). All have SB3-zoo expert checkpoints. Frostbite remains the intended
Tier-3 stretch target but lacks an SB3 expert and so currently sits in future work.

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
capability gap between the two endpoints. Using the 9B fast (MiniCPM-o 4.5) and the 8B slow
(Qwen3-VL-8B-Thinking) at matched scale — both frozen, the slow side just operating at a
slower tick + with a CoT "thinking" budget — isolates the 33M-param bridge as the only
trained part and avoids the confound "the big model was just better." Same-size also fits
comfortably in 96GB with headroom for PPO at Stage D. Capability scaling is addressed
separately as a single ablation (see below).

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
33M-param bridge (the slow model's `ThoughtProjection`) is trainable; both base models are
frozen.

We also include **O (oracle)**: pre-computed slow-model analysis injected as fast-model
context. Cheating, but bounds the upper limit of what *any* fast/slow coupling can buy.

### Capability-scaling ablation (one configuration)

In addition to the four-strategy main sweep, we run **one** cross-scale configuration with
Qwen3-30B-A3B-Thinking as the slow model under both T and L. This is *not* the main result.
Its purpose is to check whether the picture survives a stronger slow model: under the
current spine (latent helps iff slow reasoning helps), a more capable slow model that
raises T over F should also raise L. Run as `T-30B` and `L-30B` cells; 3 seeds × 30
episodes per cell.

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
- **L−F vs T−F** across games (predictor: latent helps iff slow reasoning helps)

### Game coverage (for the cross-game sweep)
The games span a range of strategic load, used to test where slow reasoning helps at all:
- **Low strategic load:** Pong, Breakout, Beam Rider — fast model alone should saturate
- **Medium:** Ms. Pac-Man, Space Invaders (SpaceInvaders surfaced a methodology finding —
  random-policy T-trajectories fail on reward-asymmetric games; Stage A OOD-brittleness)
- **High:** Seaquest, RoadRunner, RiverRaid, Qbert; Frostbite, Montezuma's Revenge
  (subset), Pitfall (future work; need scratch-trained experts)

**Current spine (predictor):** the latent helps *iff* the slow model's reasoning helps
(T > F). L−F tracks T−F at r = 0.93 (n = 8; 0.96 over 16 cells); MetaDrive is the controlled
negative. The original "phase transition by complexity tier" / bandwidth prediction is
**retired** — the latent's greedy edge over text is decoder-specific. Tuning the decoder per
channel on held-out seeds ("best-achievable"), the latent is never significantly worse than
text and is significantly better on 2 of 7 games (MsPacman, RoadRunner), 5 ties; a single
fixed greedy decoder makes it look like 4 of 7 (26–82 %), over-crediting the latent.
Combining both channels interferes (hurts 3 of 7; RoadRunner −96 %), so we couple via
exactly one channel. See `docs/06_results.md` "Hypothesis status" for the full revision.

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
- Day 20-21: Ablations (latent token-count N, decoder sweep, slow emission rate, staleness tolerance)

**Gate:** on at least one game where the slow model raises T over F, the best-achievable
latent decoder matches or beats text (never significantly worse), p < 0.05 across seeds.

### Week 4 (Days 22-28): Eval + paper
- Day 22-23: Generate demo video (3-way side-by-side: F, T, L)
- Day 24-26: Write paper draft
- Day 27-28: Polish, citations, README, release prep

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Joint 8B+9B inference doesn't fit | Very low | ~34GB weights; mitigation N/A |
| Same-size slow model too weak to raise T over F | Medium | Run the 30B-A3B cross-scale ablation to disambiguate |
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
