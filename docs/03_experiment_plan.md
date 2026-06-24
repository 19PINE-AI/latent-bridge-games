# Experiment plan

> The plan originally targeted Frostbite as the Tier-3 game; after its Stage A
> failed (no SB3-zoo expert), **Seaquest** was substituted and **SpaceInvaders**
> added as a second Tier-2 point. Frostbite remains future work. Current headline
> numbers live in `06_results.md`.

## Task

Atari via the Arcade Learning Environment (ALE): Ms. Pac-Man (Tier 2), Seaquest
(Tier 3), SpaceInvaders (Tier 2), and the wider cross-game sweep. Each game needs
both reflexes (dodge ghosts/bombs, surface for oxygen) and long-horizon planning
(route the maze, clear rows, manage the oxygen/diver budget), and each has an
SB3-zoo expert checkpoint for Stage A.

**Tick rate.** ALE runs at 60 Hz; the fast model acts every 4 frames (15 Hz
nominal, 67 ms budget). Measured warm latency is ~140 ms (~7–8 Hz effective) at
`max_slice_nums=1`; the env's frame-skip absorbs this during data generation, so
actions are still emitted every 4 frames as far as the game is concerned. The slow
model emits at 1–2 Hz.

## Four-strategy comparison

| Strategy | Fast | Slow | Channel |
|---|---|---|---|
| **F** fast-only | MiniCPM-o 4.5 | — | none |
| **S** slow-offline | — | Qwen3-VL-8B-Thinking | — (zero-shot) |
| **T** text bridge | MiniCPM-o 4.5 | Qwen3-VL-8B-Thinking | text suffix |
| **L** latent bridge | MiniCPM-o 4.5 | Qwen3-VL-8B-Thinking | continuous vectors |

**Why same-size on both sides.** The thesis is about the *channel* (latent vs
text), not a capability gap between the endpoints. Matched 9B fast / 8B slow, both
frozen — the slow side just operating at a slower tick with a CoT budget — isolates
the 33M-param bridge as the only trained part and avoids the confound "the big model
was just better." **S** is a control showing slow-only is too slow to react; **T**
is the text-coupling baseline (pure prompting, no LoRA on the slow model); **L** is
the proposal. **O (oracle)** — pre-computed slow analysis injected as fast context —
bounds the upper limit of what *any* fast/slow coupling can buy.

**L architecture.** v2 is the LLaVA-style latent-as-token bridge: the slow model
produces N=8 latent tokens in the fast model's 4096-d input-embedding space,
prepended so all 36 LLM layers attend through full causal attention. The v1 design
(mid-layer cross-attn into a 256-d ring buffer at LLM depths 12 & 24) was
empirically shown to fail despite converging to KL≈0.004 offline — see
`05_status.md`. Only the bridge (the slow model's `ThoughtProjection`) is trainable.

**Capability-scaling ablation.** One cross-scale configuration swaps the slow model
for Qwen3-30B-A3B-Thinking under T and L. This is *not* the main result — it checks
whether the picture survives a stronger slow model: under the predictor (latent
helps iff slow reasoning helps), a more capable slow model that raises T over F
should also raise L.

## Training stages

- **Stage A — behavioral baseline.** Fast-only action head trained by imitation from
  an SB3 expert (ε-greedy exploration). Establishes F and verifies the fast pipeline.
- **Stage B — text-bridge baseline.** T condition; no extra training. Also generates
  the trajectory supervision (fast context, slow text-so-far, action, reward) for C.
- **Stage C — bridge supervised training.** `KL(student‖teacher)` over the 18-way
  action distribution. COCONUT curriculum: C0 (text only) → C1 (first-half emissions
  replaced by latents) → C2 (all latents); optional C3 is bidirectional. Only the
  slow model's `ThoughtProjection` trains; both base models stay frozen.
- **Stage D — online RL.** PPO on game reward with the bridge in the loop; both base
  models frozen. Targets the offline→online distribution shift Stage C can't fix.
- **Stage E (optional) — oracle upper bound.** Pre-computed slow analyses injected as
  fast context; measures the score ceiling.

## Evaluation

- **Metrics** — mean episode score (seeds × episodes per cell, 95 % bootstrap CIs),
  mean episode length, and the predictor **L−F vs T−F** across games.
- **Latency / on-clock fraction** — confirms F, T, L meet the 67 ms tick budget; S
  does not.
- **Bridge information content** — mutual information between the bridge vectors and
  (optimal action, post-30-tick reward) on held-out trajectories; confirms the
  bridge isn't collapsed.

**Game coverage.** Low strategic load (Pong — fast model should saturate), medium
(Ms. Pac-Man, SpaceInvaders — which surfaced the Stage A OOD-brittleness finding),
high (Seaquest, RoadRunner, RiverRaid, Qbert). MetaDrive is the cross-domain
controlled negative; Frostbite/Montezuma/Pitfall are future work (need scratch-trained
experts).

**Current spine (predictor).** The latent helps *iff* the slow model's reasoning
helps (T > F): L−F tracks T−F at r = 0.93 (n = 8; 0.96 over 16 cells), with MetaDrive
the controlled negative. The original "phase transition by complexity tier" /
bandwidth prediction is **retired** — the latent's greedy edge over text is
decoder-specific. Tuned per channel on held-out seeds ("best-achievable"), the
latent is never significantly worse than text and significantly better on 2 of 7
games (MsPacman, RoadRunner); combining both channels interferes (RoadRunner −96 %),
so we couple via exactly one. See `06_results.md` for the full hypothesis revision.
