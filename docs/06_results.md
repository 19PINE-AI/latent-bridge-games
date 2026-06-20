# Results

This document summarizes the empirical findings from the v1 → v2 redesign cycle and
points at the headline numbers, by experiment.

## Headline — the predictor: the latent helps iff slow reasoning helps

The current spine of the paper is a *predictor*, not a bandwidth claim: **the latent
bridge helps exactly when slow reasoning helps the task (T > F).** Across the sweep,
L − F tracks T − F at Pearson **r = 0.93** (n = 8 reported-variant cells; r = 0.96
over all 16 cells). The bridge-replacement control confirms the mechanism — the
learned latent content tracks T > F, and on a controlled negative where slow doesn't
help (MetaDrive driving sim, T ≤ F) the latent is inert.

**Decoder-sensitivity caveat (read before any L > T number below).** The per-game
F/T/L tables below are the *greedy* (argmax) view. The latent's edge over text under
greedy decoding is **decoder-specific**: it vanishes at every fixed sampling
temperature — the thin per-emission latent signal only sharpens the greedy argmax.
The honest comparison tunes the action decoder per channel on held-out seeds
("best-achievable"): under that protocol the latent is **never significantly worse
than text, and significantly better on 2 of 7 games (MsPacman, RoadRunner)**; the
other 5 are ties. A single fixed greedy decoder makes it look like **4 of 7
(26–82 %)** but over-credits the latent. Every "L > T on N-of-7" statement carries
this caveat.

**Combining channels interferes.** Running text + latent in one forward pass never
beats the better single channel and hurts 3 of 7 games (RoadRunner −96 %). Rule:
couple via exactly one channel.

## Per-game greedy view (with the decoder caveat above)

### MsPacman (Tier 2; 12 episodes per cell)
| Strategy | Mean ± Std | Median | Best | Latency (ms) |
|---|---|---|---|---|
| F (fast only) | 256 ± 24 | 250 | 290 | 44 |
| T (text bridge) | 408 ± 88 | 385 | 600 | 50 |
| **L (v2 latent bridge)** | **628 ± 341** | **550** | **1740** | 124 |

L vs T: **+54 % mean / +43 % median**. L vs F: +145 %.

### Seaquest (Tier 3; 12 episodes per cell)
| Strategy | Mean ± Std | Median | Best | Latency (ms) |
|---|---|---|---|---|
| F (fast only) | 41.7 ± 19.1 | 40.0 | 80 | 78 |
| T (text bridge) | 63.3 ± 11.1 | 60.0 | 80 | 87 |
| **L (v2 latent bridge)** | **80.0 ± 0.0** | **80.0** | 80 | 84 |

L vs T: **+26 % mean**. L vs F: +92 %. **L is fully deterministic** (std = 0 across 12
seeds) — locked into an exploit-style policy at the local maximum of 8 kills × 10 pts.

### SpaceInvaders (Tier 2; 12 episodes per cell)
| Strategy | Mean ± Std | Median | Best | Latency (ms) | Bridge MI minus baseline |
|---|---|---|---|---|---|
| F (fast only) | 105 ± 0 | 105 | 105 | 34 | — |
| T (text bridge, original prompt) | **0 ± 0** | 0 | 0 | 61 | — |
| T (text bridge, aggressive prompt) | **0 ± 0** | 0 | 0 | 41 | — |
| L (latent, random-T) | **0 ± 0** | 0 | 0 | 54 | I(b;a)=−0.004, I(b;r)=−0.003 |
| L (latent, expert-T) | **0 ± 0** | 0 | 0 | 80 | **I(b;a)=+0.024, I(b;r)=+0.012** |
| **F (robust Stage A)** | **107 ± 60** | 92 | — | 55 | — |
| **T (robust Stage A)** | **18 ± 18** | 10 | — | 80 | — |
| **L (robust Stage A)** | **15 ± 0** | 15 | — | 93 | — |

**Stage A robustness ablation (2026-05-17): confirms the OOD-brittleness diagnosis.**
We retrained Stage A with `--suffix-prob=0.5` (half of training samples receive a
random slow-style text suffix; library sourced from the actual expert-T emissions).
Val accuracy traded down slightly (29.6 % vs 32.9 % on bare prompts) but **T and L
both broke the 0 collapse** (T = 18 ± 18, L = 15 ± 0). The recovery is small — L is
deterministic at 15, far below F's 107 — so suffix augmentation alone is not enough
to make L surpass F on this game; PPO under the deployment distribution (Stage D)
is the natural next step. But the diagnosis is now empirically validated end-to-end:
the bottleneck on SpaceInvaders was Stage A OOD brittleness, not bridge architecture,
data policy, or slow content.

Both T and L collapse to zero score, and the expert-T retry (re-collecting
T-trajectories using the SB3-DQN expert with ε=0.1 instead of a random policy)
*did not* recover the score. F alone fires and scores 105; any setup that places the
slow model in the loop suppresses FIRE.

**Initial diagnosis (later refuted)** — we first hypothesized this was a data
methodology issue: random-policy T-trajectories under-represent FIRE (~17 % of
random-uniform actions vs ~55 % under expert), so the bridge learns a passive
distribution. The expert-T retry was designed to test this exactly.

**Result of the expert-T retry** — same outcome (L = 0) but **MI(bridge; action) flips
from −0.004 to +0.024 nats** and MI(bridge; reward_sign) from −0.003 to +0.012.
The bridge under expert-T data carries the *most* information of any game in this
sweep, yet the deployed policy still scores zero. So the data fix worked at the
representation level — the bridge does learn structure — but the deployed effect on
the fast model is still passivity.

**Second hypothesis (also refuted)** — we then rewrote the SI slow-model prompt to
be aggressively shoot-oriented ("CRITICAL: the only way to score is to FIRE… never
abandon firing to dodge unless a bomb is directly above the cannon"). Ran T eval
again with no other changes. **Result: T = 0 ± 0 across 12 episodes** — same outcome.
So the bottleneck is not the slow model's guidance *content* either.

**Final diagnosis (consistent with all observations)** — Stage A trained on
SB3-DQN-collected trajectories with a **bare game-state prompt** (no slow-model
suffix). At deployment, F uses the same bare prompt (matches training); T appends
a text suffix; L prepends bridge tokens. Both T and L therefore present an
**out-of-distribution prompt** to a frozen action head. The head degrades.
- On MsPacman (9 actions, every direction collects dots) and Seaquest (18 actions,
  every direction does something useful), degraded action selection still scores —
  the slow's guidance contributes enough beyond the noise to give L > T > F.
- On SI (6 actions, only the 3 FIRE-containing actions score), the same degradation
  manifests as a non-FIRE-biased policy → zero score.

This is a single coherent story across all our experiments: random-T, expert-T, and
aggressive-prompt all produce L = T = 0 on SI not because of bridge architecture,
data policy, or slow content, but because the action head was never exposed to a
suffix-augmented prompt during Stage A and the action space is too unforgiving of
the resulting policy drift to give us a non-zero readout.

**Why the bridge methodology works at all then** — for symmetric-reward games the
distribution shift induced by adding a bridge / suffix is *small enough* that the
slow's contextual information improves the policy net of the shift cost. The bridge
mechanism is sound; it is bounded above by the worst-case OOD robustness of the
frozen Stage A head.

Future remedies (not run): (a) co-train Stage A with a mix of bare and
suffix-augmented prompts so the action head is in-distribution for T/L; or (b) use
Stage D PPO to fine-tune the action head on game reward under the deployment
prompt distribution. Either should recover SI.

### Q*bert (Tier 2-3; isometric platformer — 12 episodes per cell)
| Variant | F | T | L |
|---|---|---|---|
| Bare Stage A | 25 ± 0 | 0 ± 0 | 0 ± 0 |
| **Robust Stage A (suffix-prob=0.5)** | 25 ± 0 | **125 ± 0** | **50 ± 0** |

Robust Stage A broke the T/L collapse; under greedy decoding T > L on Q*bert
(T=125 vs L=50). Under tuned-decoder ("best-achievable") evaluation Q*bert is a
tie like most games.

**Note — the "continuous-vs-categorical" hypothesis is retired.** We originally
read Q*bert as evidence that L > T only when the slow's reasoning is
*continuous-rich* (fuel/distances/coordinates) and T ≥ L when it is categorical
("jump UP-RIGHT to tile row 3, col 2"). That hypothesis does **not** survive: the
slow model's emission statistics do not predict sign(L − T) — lexical diversity
correlates with the gap at only r = +0.05 (n.s.). The sign of L − T is governed by
whether slow reasoning helps the task (the predictor, T > F), not by whether its
content is continuous or discrete.

### Enduro (Tier 2; scrolling racing — 12 episodes per cell)
| Variant | F | T | L |
|---|---|---|---|
| Bare Stage A | 3.2 ± 2.5 | 0 ± 0 | 7.8 ± 8.7 |
| Robust Stage A | 0.8 ± 1.0 | 4.9 ± 5.6 | **5.8 ± 2.5** |

Robust Stage A on Enduro modestly recovers T from 0 to 5; L holds at ~6
with **+18 % L > T**. Absolute scores are tiny because the SB3 PPO expert
never learned to play well (avg_reward=0 on collection), bounding Stage A's
ceiling.

### Enduro (Tier 2; scrolling racing — 12 episodes per cell)
| Strategy | Mean ± Std | Median | Comment |
|---|---|---|---|
| F (fast only) | 3.2 ± 2.5 | 3.5 | Stage A val_acc 49.2 % |
| T (text bridge) | **0 ± 0** | 0 | Suffix collapses policy |
| L (latent bridge) | **7.8 ± 8.7** | 3.5 | +144 % over F, infinity over T |

Enduro was queued as a RoadRunner-similar candidate. SB3 PPO expert was weak
(0 avg reward on collection); Stage A still trained to 49.2 % val_acc on
the partial trajectories. The L > T direction is consistent with RoadRunner
but absolute scores are tiny. The robust-Stage-A retry (in the Q*bert
section above) gave a modest +18 % L over T.

### RoadRunner (Tier 3; 12 episodes per cell)
| Strategy | Mean ± Std | Median | Best |
|---|---|---|---|
| F (fast only, bare Stage A) | **0 ± 0** | 0 | 0 |
| T (text bridge) | 475 ± 160 | 500 | — |
| **L (latent bridge)** | **608 ± 29** | 600 | — |

**Greedy L > T by +28 %** (d = 1.16), reproducible at L = 608. RoadRunner is one of
the two games (with MsPacman) where the latent stays significantly above text under
the tuned-decoder protocol, so the L > T direction here is robust. The *magnitude* is
not: an earlier run scored L = 967, but the F = 0 baseline (a bare-Stage-A artifact;
see the robust-SA section below) makes the absolute score run-to-run-unstable under
FP nondeterminism. **967 is a noted high-water mark only — never the canonical value,
and "+59 %" should not be reported.** RoadRunner is **not** the largest L − T gap:
River Raid's greedy +82 % is larger (and is a tie under tuned decoders).

The bare-Stage-A inverse pattern is the demo's visual draw — F can't play (score = 0)
yet the slow's directional context ("head right to escape the Coyote") unlocks a
high-scoring policy. But note F = 0 is itself a Stage A bare-prompt artifact: under
robust Stage A all three channels tie near the ceiling (F = 958, T = 1000, L = 925;
see below). Stage A val_acc on RoadRunner was the highest of any game (58.5 %), so the
action head is competent.

MI diagnostic: I(b;a) − baseline = −0.05 (negative on the training trajectory)
yet deployed L = 608 — interesting discrepancy. The bridge's value at deployment
isn't captured by static action-prediction on the bare training distribution; it
emerges from the joint slow-fast computation.

### River Raid (Tier 3; 12 episodes per cell)
| Strategy | Mean ± Std | Median | Bridge MI minus baseline |
|---|---|---|---|
| F (fast only) | **1067 ± 84** | 1060 | — |
| T (text bridge) | 383 ± 57 | 390 | — |
| L (latent bridge) | 360 ± 0 | 360 | I(b;a) = −0.0006, I(b;r) = +0.003 |

River Raid has 4 interacting objectives (fuel / dodging / shooting / path) and
continuous long-horizon state. Predicted L > T > F.

**Observed pattern: another Stage A OOD-brittleness case.** F plays well (Stage A
val_acc 31.5 %); but the slow-text suffix and bridge tokens both *degrade* the
policy to a fraction of F's score. The pattern matches the SpaceInvaders failure
exactly: when the action head was trained on the bare-prompt distribution and
the game requires precision (dodging + targeting), even a small distribution
shift from the suffix flips a working policy into a failing one. MI on the
bridge is near zero, confirming nothing useful was encoded.

**Implication:** under bare Stage A, OOD-brittleness pre-empts any L-vs-T reading —
F is much higher than T, and L is bounded above by T's KL anchor. The L vs T
comparison only becomes meaningful *after* the Stage A distribution mismatch is fixed
(via Stage A robustness training, à la the SI fix, or Stage D PPO under deployment).

**Update (robust Stage A retry, 2026-05-18):** retraining Stage A with
`--suffix-prob=0.5` (the same fix that broke SI's L=0 collapse) recovers River
Raid dramatically:

| Variant | F | T | L | L vs T |
|---|---|---|---|---|
| RR bare Stage A | 1067 ± 84 | 383 ± 57 | 360 ± 0 | collapse |
| **RR robust Stage A** | 1033 ± 19 | **337 ± 77** | **612 ± 297** | **+82 %** |

Greedy L over T by **+82 %** — the largest greedy L − T gap on any game we've tested
(though under the tuned-decoder protocol River Raid is a **tie**, so this gap is
decoder-specific and does not count toward the "significantly better" tally). The
OOD-brittleness diagnosis is now confirmed *with the same fix recipe* on two
distinct games (SI + RR): once Stage A is unblocked, the greedy L > T direction
returns as predicted.

### Pong (Tier 1; reactive-only — 12 episodes per cell)
| Strategy | Mean | Median | Comment |
|---|---|---|---|
| F (fast only) | −21 ± 0 | −21 | Loss floor (CPU 21-0) |
| T (text bridge) | −21 ± 0 | −21 | Same |
| L (latent bridge) | −21 ± 0 | −21 | Same |

Stage A val_acc was only 25.1 % (1.5 × random for 6 actions), so the action head
cannot play Pong even at amateur level. The slow model cannot rescue a broken
reactive policy — its strategic guidance ("move paddle to match ball y") cannot
compensate for the head not learning the basic reflex. This is the **other**
Tier-1 failure mode (vs the predicted "F saturates and bridges don't help
because there's nothing to plan for"): here F fails *and* T/L can't fix it.

The published H2 prediction was L ≈ T on Tier 1. Empirically all three are
*identical at the loss floor*, which is consistent with L ≈ T but for a different
reason than predicted. The clean conclusion for the paper: **the bridge
mechanism's contribution is gated by the fast model's baseline competence; on
games where the action head fails the imitation step, no bridge architecture
can fix it.** This is the upstream analogue of the SpaceInvaders finding
(which was Stage A OOD-brittleness; this is Stage A under-fitting).

### Cross-game summary
- **Predictor (the spine):** the latent helps iff slow reasoning helps (T > F). L − F
  tracks T − F at r = 0.93 (n = 8) / r = 0.96 (n = 16). MetaDrive is the controlled
  negative (T ≤ F → latent inert).
- **Best-achievable (tuned-decoder) tally:** the latent is never significantly worse
  than text; significantly better on **2 of 7** games (MsPacman, RoadRunner); the
  other 5 tie. A single fixed greedy decoder inflates this to **4 of 7 (26–82 %)** —
  that view over-credits the latent and must carry the decoder-sensitivity caveat.
- **Combining channels interferes** — text + latent in one pass never beats the better
  single channel and hurts 3 of 7 (RoadRunner −96 %); couple via exactly one channel.
- **Stage A OOD-brittleness gates everything (SpaceInvaders, River Raid, bare
  RoadRunner):** when only one or two actions carry reward, adding a bridge/suffix to a
  bare-prompt-trained head drives the policy off-distribution and can collapse T/L to
  0. Robust Stage A (suffix-prob=0.5) fixes it on the collapsed games but hurts the
  ones that already worked — it is targeted, not universal.
- The honest interpretation is: **the latent matches or modestly beats text under the
  best decoder for each, the direction is predicted by whether slow helps, and the
  approach is bounded above by the OOD robustness of the frozen Stage A head.**

## What changed from v1 (the design that failed)

v1: mid-layer cross-attention into a 256-d ring buffer, injected at LLM depths 12 & 24
only; 71 M trainable params on the fast side.

  - v1 L: 225 ± 85 — **worse than F**, bimodal with 4/12 catastrophic episodes (70–160).
  - KL converged to 0.004 *on the training distribution* but **failed at deployment**.
  - Three architectural variants (ungated / gated / gated + head-tune) all failed
    or mode-collapsed.

v2: LLaVA-style latent-as-token bridge — slow model emits 8 tokens at the fast LLM's
full input embedding width (4096-d) with LayerNorm-bookended ThoughtProjection;
tokens are **prepended to the fast model's input sequence** so all 36 LLM layers
attend through full causal attention. Only the slow model's `ThoughtProjection`
(~33.6 M MLP params) is trainable; everything else frozen.

**Two changes that fixed it:**
1. Bridge tokens live in the LLM's input-embedding space (vs 256-d arbitrary).
2. All 36 layers attend over them (vs cross-attn at 2 layers).

These are exactly the architectural privileges text tokens have inherited from the
LLM's pretraining. The cross-attn approach was asking the model to learn a new
modality from scratch with ~5K training samples; the LLaVA approach lets the model
reuse its existing sequence-processing pipeline.

## Hypothesis status

| H | Claim | Status |
|---|---|---|
| Predictor | The latent helps iff slow reasoning helps (T > F) | ✅ **CONFIRMED**. L − F tracks T − F at r = 0.93 (n = 8) / r = 0.96 (n = 16); MetaDrive is the controlled negative. This is the paper's spine. |
| Best-achievable L ≥ T | Under a per-channel tuned decoder, the latent is never significantly worse than text | ✅ **CONFIRMED**, and significantly better on 2 of 7 (MsPacman, RoadRunner). The greedy-only 4-of-7 view over-credits the latent. |
| Frozen-base coupling | COCONUT-style frozen-base + adapter recovers most of the unified bound | ✅ **CONFIRMED**. Only the slow ThoughtProjection (~33.6 M params) is trained; fast model entirely frozen. |
| ~~Bandwidth / capacity ceiling~~ | ~~Text is bandwidth-limited; the gap grows with strategic complexity~~ | ❌ **RETIRED** (see Latent token-count ablation below). 30B slow doesn't widen the gap; longer text doesn't close it; deploy-only N = 16 scores best; used capacity ≪ nominal. No capacity ceiling. |

## Stage A (behavioral cloning) results

| Game | n_train | val_acc (best) | random baseline | × random |
|---|---|---|---|---|
| MsPacman | 4852 | 32.1 % | 11.1 % (1/9) | 2.9 × |
| Seaquest | 4733 | 24.2 % | 5.6 % (1/18) | 4.3 × |
| SpaceInvaders | 4634 | 32.9 % | 16.7 % (1/6) | 2.0 × |

Seaquest's per-action signal is stronger relative to random (4.3 × vs 2.9 ×) because
the larger 18-action space gives the slow expert more to exploit; in absolute terms
MsPacman's 9-action accuracy is higher.

## Stage C v2 (KL bridge training)

| Game | Mean KL (epoch 1) | Final KL (last steps) |
|---|---|---|
| MsPacman | 0.026 | ~0.005 |
| Seaquest | 0.024 | ~0.006 |
| SpaceInvaders | 0.023 | ~0.005 |

KL convergence is similar across all three games — yet deployment outcomes diverge
dramatically. This re-confirms the v1 lesson: **training KL is necessary but does
not predict deployment behavior** when the trajectory distribution diverges from the
reward-relevant policy.

KL convergence on training distribution is a *necessary* but **not sufficient**
condition for deployment success — confirmed empirically by v1, where KL=0.004
preceded deployment catastrophe. v2 succeeds in deployment *also* because the
architectural privileges match the LLM's pretraining.

## Latent token-count (N) ablation

This was originally run as a "bandwidth ablation" to test a capacity-ceiling thesis.
**That thesis is retired: there is no capacity ceiling.** More latent tokens at
deploy is *not* monotonically better (N = 8 beats N = 16 when both are trained), the
30B slow model does not widen the gap, and longer text does not close it — the latent
uses far less than its nominal capacity. What follows is the token-count sweep, read
purely as an N hyperparameter study (not as evidence for or against bandwidth).

### Phase 1: deployment-time N scan (fixed projection)

We trained Stage C v2 with `LB_BRIDGE_N_TOKENS=4/8/16` env var, but because Stage C
uses *cached* residuals from T-trajectory files (always saved with N=8) and the seed
is fixed, all three projections came out **bit-identical**. So this scan actually
measured: *given a projection trained on 8 tokens, how does eval-time token count affect L*?

| Deploy N | L mean ± std | L median | Note |
|---|---|---|---|
| 4 | 232 ± 49 | 190 | Projection under-utilized at deploy; bridge collapses below F |
| **8** | **628 ± 341** | **550** | Matches training |
| **16** | **720 ± 117** | **770** | Same projection; more tokens at deploy still helps + lower variance |

Result: more inference-time tokens helps even when the projection was trained on fewer.
Variance also drops substantially at N=16 (117 vs 341).

### Phase 2: matched train + deploy N

| N | KL converged | L mean ± std | L median | Comment |
|---|---|---|---|---|
| 4 | 0.020 | 296 ± 63 | 260 | Above F=256 but below T=408 |
| **8** | 0.026 | **628 ± 341** | **550** | **Sweet spot** — best mean |
| 16 | 0.021 | 259 ± 71 | 290 | Worse than F! — more tokens dilute |

**Finding: N is Goldilocks, not monotonic.** Both N=4 and N=16 underperform N=8.
This is the opposite of what a capacity-ceiling ("more bandwidth is better") story
predicts, and is one of the reasons that story is retired: the deploy-only N=16 run
(Phase 1) actually scores *best* of all, so the limiting factor is the training
distribution, not channel capacity.

Why N=16 hurts:
- The slow model generates ~96 tokens per emission. The "last 8" positions contain
  its *conclusion* (after `<think>`). The "last 16" dilutes with more reasoning tokens.
- Bigger bridge prefix = the fast LLM's attention budget is split across more tokens.
- The projection has to learn more variation with the same training data.

**Reconciliation with Phase 1:**
- Phase 1 N=16 deployment-only (with N=8-trained projection): L = 720. This works
  because the projection learned the high-info 8-position distribution AND received
  extra tokens at inference.
- Phase 2 N=16 train+deploy: L = 259. Training the projection on lower-info positions
  *and* deploying at that N gives the worst-of-both-worlds.

The practical recipe: **train on N=8, deploy with N=8-16**. The training distribution
matters more than the deployment N.

## MI diagnostic

Practical mutual-information estimator (binned plug-in) on the trained MsPacman v2
bridge:

| Y | I(bridge; Y) | Shuffled baseline | I − baseline |
|---|---|---|---|
| sign(future-30-tick reward) | 0.024 nats | 0.014 nats | **+0.010** |
| action taken | 0.087 nats | 0.085 nats | +0.002 (≈0) |

Action MI is near baseline as expected — the T-trajectory actions were **random**,
so no projection of the slow's residuals could predict them. The future-reward MI is
positive: the bridge **does** carry information about reward outcomes, confirming the
bridge isn't collapsed.

## Latency

Profile of `FastModel.predict_action` (Ms. Pac-Man, max_slice_nums=1):

| Phase | Time |
|---|---|
| Processor (image+text tokenize) | 14 ms |
| Host→device transfer | 42 ms |
| `get_vllm_embedding` (vision tower) | 80 ms |
| `llm.model` 36-layer forward | 52 ms |
| Action head | <1 ms |
| **Total cold** | **197 ms** |
| With `torch.compile` on llm.model | **170 ms** (−13 ms) |
| With v2 bridge prepended (8 extra tokens) | +5 ms |
| **v2 L target** | **~125 ms** (achieved) |
| **15 Hz target** | **67 ms** (not yet) |

Remaining latency work (queued task): vision-token caching across consecutive frames
(Atari changes ~1 px/tick) should cut ~50 ms.

### Vision-token caching benchmark (F MsPacman)

Spot check (n=2 per cell, isolated GPU):
| `vision_refresh_every` | Latency (ms) | Score |
|---|---|---|
| 1 | 33 | 180 |
| 4 | 20 (−39 %) | 110 |
| 15 | 17 (−48 %) | 140 |

Larger sweep (n=12 per cell; 3 seeds × 4 eps; competing GPU workload):
| `vision_refresh_every` | Latency (ms) | Score |
|---|---|---|
| 1 | 157 | 183 ± 4 |
| 4 | 101 (−36 %) | 110 ± 0 |
| 15 | 77 (−51 %) | 140 ± 0 |

The *absolute* latencies differ between the two runs (isolated vs competing
GPU) but the *relative* speedup is the same and the score pattern reproduces:
vrf=1 highest score, vrf=4 lowest, vrf=15 middle. The 110/140 score
non-monotonicity at vrf=4 vs vrf=15 reproduces with std=0 — it's a real
artifact of how perception staleness interacts with this particular policy,
not noise.

The vision tower (SigLIP + resampler) is the dominant cost in the fast tick. Caching
it across N consecutive ticks roughly halves per-tick latency. The score numbers are
noisy at n=2 per cell (110 vs 140 is sample variance), but the latency speedups are
consistent and large. Combined with `torch.compile` on the LLM forward, the warm
tick path is **well under the 67 ms (15 Hz) target** when vision is cached. The
optimal `vision_refresh_every` is a per-game hyperparameter trading perception
latency for action latency; for action-heavy games like Atari we expect cache
windows of 2-4 to be the sweet spot.

## Robust Stage A on winning games — Phase 8 (2026-05-19)

We applied the `--suffix-prob=0.5` Stage A retraining to the three games where
L > T already worked, to test whether robust SA is **universally beneficial**
or **targeted to OOD-collapsed games**.

| Game | Bare F / T / L | Robust F / T / L | Direction |
|---|---|---|---|
| RoadRunner | 0 / 475 / **608** | 958 / **1000** / 925 | F recovered; T slightly > L |
| MsPacman | 256 / 408 / **628** | 325 / 61 / 60 | **L collapsed** |
| Seaquest | 42 / 63 / **80** | 20 / 0 / 0 | **L collapsed** |

**Outcome: robust SA hurts the winners.** MsPacman and Seaquest lost their
L > T results entirely — Stage A val_acc dropped (32 % → 33 %; 24 % → 19 %),
Stage C KL rose, and the resulting L is no longer above F. The recipe that
recovered SpaceInvaders and River Raid (where T/L had collapsed to 0)
**hurts games where T/L was already winning**.

**Lesson**: robust SA is *targeted*, not universal. Use it when:
- T/L collapse to 0 or near-zero (the Stage A OOD-brittleness diagnosis)
- F's bare-prompt accuracy is robustly above the random floor

Don't use it when:
- T/L already works with bare Stage A
- The slight bare-prompt accuracy drop dominates the suffix-robustness gain

**Surprise on RoadRunner**: F jumped from 0 to 958 under robust SA. This
implies the original "F = 0" wasn't a fundamental policy-stuckness — it was
Stage A overfitting to the exact bare-prompt structure. The robust head's
prompt-augmentation regularization fixed F. Robust L = 925 (bare L = 608),
so under robust SA all three strategies tie near the ceiling.

The RoadRunner robust headline becomes: **F = 958, T = 1000, L = 925** — T
beats L slightly, and the F = 0 → L = 608 inversion is no longer the
demo's centerpiece. The bare-Stage-A RoadRunner remains the most visually
striking result in the demos directory (and is honest — that *is* what
happens with bare Stage A on RoadRunner).

## Stage D PPO on SpaceInvaders — mode collapse (driver works; tuning needed)

Ran 20 PPO updates from the robust Stage A + Stage C v2 SI checkpoints
(starting baseline: F=107, T=18, L=15). PPO driver started cleanly:

| Update | Rollout score | Entropy | KL anchor | Note |
|---|---|---|---|---|
| 0 | 0 | 0.796 | 3.28 | Warm start; large KL deviation as expected |
| 6 | **50** | 0.789 | 0.43 | First positive rollout |
| 7-19 | 0 | → 0.000 | → 0.000 | **Policy collapsed to single deterministic action** |

Final eval: F = T = L = 0 across 12 episodes. The PPO checkpoint is worse
than the robust-Stage-A baseline it started from.

**Diagnosis**: classic PPO collapse on sparse-reward games. The entropy
coefficient (0.01, standard Atari PPO default) was too small to prevent the
policy from collapsing onto a low-variance attractor once it found one. SI's
reward structure (rare FIRE-hit events) amplifies this: a few updates with
slightly more probability mass on NOOP-like actions get rewarded by the value
loss (lower variance prediction), and the policy implodes onto that action.

The driver itself works: gradients flow through `slow.projection` +
`action_head` + `value_head`; the KL anchor against the Stage C reference
fires; the rollout buffer + GAE compute correct advantages. Future PPO runs
should:
1. Raise `entropy_coef` from 0.01 → 0.05-0.1 for sparse-reward games
2. Add reward shaping (intrinsic motivation or step-bonus) for SI
3. Smaller LR on `action_head` (currently 2.5e-4) to slow the collapse mode

Not run; PPO is engineering-heavy enough that one calibrated re-run is
worth more than three uncalibrated ones.

## Phase 9 gap-closers — partial (2026-05-19)

Three gap-closer experiments queued; one completed, two infrastructure-blocked.

**Latency sweep v2** (n=12 per cell): completed. Numbers above. Replaces the
prior n=2 spot check; latencies are higher absolutely (157/101/77 vs prior
33/20/17 ms) due to competing user workloads on the GPU, but relative
speedup and score-vs-vrf pattern reproduce with std=0 — robust finding.

**True pre-computed Oracle** (1024-token slow budget, MsPacman, n=3):
infrastructure-blocked. OOM during slow.emit() when MiniCPM-o (18 GB) +
Qwen3-VL-8B (17 GB) + 1024-token generation activations couldn't fit
alongside the user's concurrent 60 GB of GPU workloads (LoRA fine-tune +
judge + textworld training, none of which we have authorization to kill).
A 256-token retry was queued but the user's workloads grew further (70 GB);
even that didn't fit.

**Stage D PPO retry** (entropy_coef=0.1, lr_action_head=5e-5, clip_range=0.2,
SI): infrastructure-blocked at slow-model load — same OOM root cause as Oracle.

Both await a window where the user's other GPU workloads release memory.
The driver and hyperparameter changes are committed; once GPU has ~40 GB
free, either experiment is a single command away. The paper can honestly
flag these as "future work pending GPU availability."

## Negative findings worth keeping in the paper

1. **v1 cross-attention bridge** — the design we tried first — does not work in
   deployment despite excellent offline KL. This is a publishable negative result
   because the literature on adapter-based fast/slow coupling rarely tests deployment.
2. **Cross-game action head transfer fails** — MsPacman Stage A head deployed on
   Frostbite produces deterministic 80-score episodes. Per-game adapters are required.
3. **RWR self-bootstrap from random-policy trajectories** for missing-expert games
   (Frostbite) fails — val_acc ≈ uniform random. Reward-weighted regression on
   random labels cannot disambiguate causal actions.
4. **Joint action_head + bridge KL training mode-collapses**. The KL loss has a
   degenerate solution where both teacher and student output the same deterministic
   distribution. Do not unfreeze action_head during KL training without entropy
   regularization or labeled CE supervision.
5. **Stage A action-head OOD brittleness collapses reward-asymmetric games when a
   bridge/suffix is added at deployment.** Convergent finding from three
   SpaceInvaders interventions (all gave L = T = 0 vs F = 105): the random-T
   baseline, the expert-T retry (which proved the bridge does carry information —
   MI flipped from −0.004 to +0.024 nats — yet still L = 0), and the
   aggressive-prompt retry (rewriting the slow's text guidance to emphasize
   shooting also produced T = 0). Root cause: Stage A was trained on the bare
   game-state prompt; T appends a slow-model text suffix and L prepends bridge
   tokens, both of which present OOD input to the frozen action head. On
   reward-symmetric games (MsPacman, Seaquest) the resulting policy drift still
   scores; on SI the drift biases away from FIRE and gives zero. Future remedies:
   co-train Stage A with suffix-augmented prompts, or use Stage D PPO under the
   deployment distribution.

## What's queued / running

- (all overnight experiments complete — Seaquest, SpaceInvaders, latent token-count (N)
  ablation, MI on all three bridges)

## What's still future work

- **Stage D PPO** — on-policy fine-tuning of the bridge with reward signal. Not yet
  needed for the headline claim, but would push L scores higher and is the natural
  next step.
- **Latency** — vision-token caching to hit 67 ms / 15 Hz.
- **Scaling ablation** — the 30B-A3B slow model has been run; it does **not** widen
  the L − T gap, which is part of the evidence retiring the capacity-ceiling thesis.
- **More games** — additional cells to tighten the predictor correlation.
- **Demo video** — F/T/L side-by-side playthrough.

## Files of record

- `results/eval_v2_mspacman.json` — headline 12-episode F/T/L numbers
- `checkpoints/stage_a/mspacman_sb3dqn_v2.pt` — Stage A action_head (val 32.1 %)
- `checkpoints/stage_a/seaquest_sb3dqn.pt` — Stage A action_head (val 24.2 %)
- `checkpoints/stage_c/v2_mspacman.pt` — v2 trained slow projection (33.6 M)
- `results/t_trajectories_v2/*.pt` — 10 MsPacman + 10 Seaquest T-trajectories with
  raw slow residuals (4096-d × N=8 per emission)

All experiments reproducible via the scripts in this repo; entry points are
`scripts/run_text_bridge_baseline.py` (data collection), `src/training/stage_a_behavioral.py`,
`src/training/stage_c_v2.py`, and `src/eval/benchmark.py`.
