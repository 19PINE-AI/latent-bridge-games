# Framing

## Thesis

Real-time interactive AI systems face a structural dilemma: reasoning models (Claude 4.7,
GPT-5, Qwen3-VL-Thinking) deliberate excellently but cannot operate at 100-200ms tick
rates; small streaming models hit the tick rate but lack reasoning depth. Current
production systems address this with either (a) monolithic streaming models with shallow
thinking (Gemini Live, OpenAI Realtime), or (b) two-model splits communicating via text
prompts (Thinking Machines Lab's interaction model + asynchronous background reasoner).

We argue the text-channel split — the most architecturally promising option — is
bandwidth-limited. Text serialization carries hundreds of bits per call where a continuous
latent channel could carry hundreds of thousands. This bottleneck shows up empirically as
slow redirection of reasoning when the environment changes rapidly, shallow handling of
substantive user interruptions, and inability to maintain tight slow-fast coupling under
high-event-density conditions.

This project investigates whether a **learned continuous-valued bridge** between a frozen
real-time multimodal model and a frozen reasoning model can outperform the text-channel
default, demonstrated on video games requiring both fast reflexes and long-horizon
planning.

## Why games

Games are the cleanest empirical domain for this question:

1. **Visual self-evidence.** The fast/slow split is visible to viewers: reactive gameplay
   is the fast model's contribution, strategic planning is the slow model's. A side-panel
   showing the slow model's thoughts makes the architecture legible without prior
   knowledge.

2. **Objective evaluation.** Score, survival time, and level completion are automatically
   measurable across architectures, with no judge bias or subjective listener ratings.

3. **Universal recognition.** Ms. Pac-Man and Frostbite are universally familiar; the
   demo communicates instantly without setup.

4. **Established baselines.** Decades of RL work on Atari give comparison reference points
   for what fast-only agents achieve.

5. **Clean fast/slow separability.** Reflex-only games (Pong, Breakout) and strategy-only
   games (chess at slow time controls) don't justify the architecture; games where both
   matter (Ms. Pac-Man's stochastic ghosts, Frostbite's temperature timer) are the
   precise empirical target.

## Hypotheses

**H1 (capability).** On games requiring both reactive and strategic components, a
MiniCPM-o 4.5 + Qwen3-VL-8B-Thinking system coupled via a learned latent bridge achieves
significantly higher scores than (a) MiniCPM-o 4.5 alone, (b) the same pair coupled via a
text channel. We make this claim with both endpoints at the same ~8-9B scale to isolate
the channel from the capability gap; a scaling ablation with Qwen3-30B-A3B-Thinking on
the slow side tests whether the latent advantage grows with slow-model reasoning depth.

**H2 (phase transition).** The latent-vs-text gap is small on games with low strategic
load and grows with strategic complexity. There exists an identifiable complexity
threshold above which text-channel splits saturate while latent bridges continue to scale.

**H3 (training tractability).** A COCONUT-style staged curriculum (text → mixed → latent),
trained only on LoRA adapters with both base models frozen, recovers most of a unified
jointly-trained upper bound's capability — making the architecture deployable when the
slow model cannot be retrained.

## Contributions if all hypotheses hold

1. A reproducible cross-model latent-bridge architecture for non-physical real-time
   interaction.
2. A phase-transition empirical result distinguishing where text-channel splits fail.
3. A training recipe — COCONUT-curriculum + dual LoRA — runnable on a single 96GB GPU.
4. Open-source release: code, LoRA checkpoints, eval harness, game-environment wrappers.

## Scope

**In scope:**
- Atari-class games (ALE / Gymnasium) at 10-60Hz tick rates
- MiniCPM-o 4.5 as the frozen fast model
- Qwen3-VL-8B-Thinking as the frozen slow model (with Qwen3-30B-A3B-Thinking as a single
  scaling-ablation configuration on Frostbite)
- LoRA adapters on both for bridge endpoints
- Four-strategy comparison: fast-only, slow-only-offline, text-bridge, latent-bridge

**Out of scope (for this paper):**
- Pretraining either base model
- Voice/audio scenarios (deferred — see follow-up)
- Physical robotics (covered by Helix, π0, FiS-VLA elsewhere)
- 3D games / modern AAA-class environments
- Multi-agent settings

## Non-goals

- We do NOT claim to invent a new model architecture from scratch.
- We do NOT claim to beat TML on absolute latency.
- We do NOT claim general superiority of latent over text channels across all tasks; the
  claim is bounded to tight-coupling regimes.

## Failure modes the design must guard against

- The fast model is already too good at the chosen games (no headroom for slow model to help)
- The bridge collapses to constant output (slow model's projections become uninformative)
- Slow model emission rate is too low to keep the buffer fresh
- Evaluation is too noisy at the 100-task game-seed granularity to separate conditions
