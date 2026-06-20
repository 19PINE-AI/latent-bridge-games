# Framing

## Thesis

The long-term target is general computer use (GCU): an agent that must act continuously
while reasoning over longer horizons. Real-time games are the hardest case of this —
the agent must act every few tens of milliseconds while planning over seconds. No single
open multimodal LLM does both: a reasoning VLM (Qwen3-VL-8B-Thinking) is ~1.5s too slow
for the ~15Hz control loop, while a reactive VLM (MiniCPM-o 4.5) lacks deliberation. The
fast/slow split is the fix. Thinking Machines Lab's "Interaction Models" make this split
explicit, coupling the two via shared text/context; closed systems (Gemini Live, OpenAI
Realtime) are streaming-only and appear here only as related work, not motivation.

We test an **open alternative** to text coupling: a **learned continuous latent bridge**
that projects slow-model residuals into the fast model's input-embedding space (LLaVA-style).
This project investigates whether such a bridge between a frozen real-time multimodal model
and a frozen reasoning model can match or outperform the text-channel default, demonstrated
on video games requiring both fast reflexes and long-horizon planning.

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

**H1 (best-achievable capability).** On games requiring both reactive and strategic
components, a MiniCPM-o 4.5 (9B) + Qwen3-VL-8B-Thinking (8B) system coupled via a learned
latent bridge is never significantly worse than the same pair coupled via a text channel,
and is significantly better on some games — under the honest comparison that tunes the
action decoder per channel on held-out seeds ("best-achievable"). Both endpoints are at
the same ~8-9B scale to isolate the channel from the capability gap; the 33M-param bridge
is the only trained component, both base models frozen. Caveat: the latent's apparent
greedy advantage is decoder-specific (it vanishes at every fixed sampling temperature),
so the headline must be stated under the best-achievable decoder, not a single fixed
greedy decoder.

**H2 (latent helps iff slow reasoning helps).** The latent helps a task exactly when slow
reasoning helps it (T>F). The per-game latent gain L−F tracks the slow-reasoning gain T−F
at Pearson r≈0.93 (n=8). MetaDrive (driving sim) is the controlled negative where neither
helps. (A latent token-count (N) ablation probes how few latent tokens suffice; this is
NOT a "bandwidth" claim — that thesis is retired.)

**H3 (training tractability).** A COCONUT-style staged curriculum (text → mixed → latent),
training only the bridge with both base models frozen, recovers most of a unified
jointly-trained upper bound's capability — making the architecture deployable when neither
base model can be retrained.

**Single-channel rule.** Combining both channels does not help; it interferes (never beats
the better single channel, and hurts on some games). Couple via exactly one channel.

## Contributions if all hypotheses hold

1. A reproducible cross-model latent-bridge architecture for non-physical real-time
   interaction.
2. An empirical result that the latent helps iff slow reasoning helps (L−F tracks T−F,
   r≈0.93), under a best-achievable per-channel decoder — with the latent never
   significantly worse than text and better on some games.
3. A training recipe — COCONUT-curriculum + bridge-only training — runnable on a single
   96GB GPU (NVIDIA RTX Pro 6000).
4. Open-source release: code, bridge checkpoints, eval harness, game-environment wrappers.

## Scope

**In scope:**
- Atari-class games (ALE / Gymnasium) at 10-60Hz tick rates
- MiniCPM-o 4.5 as the frozen fast model
- Qwen3-VL-8B-Thinking as the frozen slow model
- A 33M-param latent bridge as the only trained component (both base models frozen)
- Four-strategy comparison: fast-only (F), slow-only-offline (T), text-bridge, latent-bridge (L)

**Out of scope (for this paper):**
- Pretraining either base model
- Voice/audio scenarios (deferred — see follow-up)
- Physical robotics (covered by Helix, π0, FiS-VLA elsewhere)
- 3D games / modern AAA-class environments
- Multi-agent settings

## Non-goals

- We do NOT claim to invent a new model architecture from scratch.
- We do NOT claim to beat TML on absolute latency.
- We do NOT claim general superiority of latent over text channels across all tasks; under
  the best-achievable decoder the latent is never significantly worse than text and better
  only on some games, and combining both channels interferes.
- We do NOT claim a "bandwidth" advantage. The bandwidth/capacity-ceiling story is retired
  (a 30B slow model does not widen the gap; longer text does not close it). Likewise the
  continuous-vs-categorical hypothesis is retired (emission statistics do not predict
  sign(L−T)).

## Failure modes the design must guard against

- The fast model is already too good at the chosen games (no headroom for slow model to help)
- The bridge collapses to constant output (slow model's projections become uninformative)
- Slow model emission rate is too low to keep the buffer fresh
- Evaluation is too noisy at the 100-task game-seed granularity to separate conditions
