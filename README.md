# The Latent Bridge: A Continuous Slow–Fast Channel for Real-Time Game Agents

> 🎬 **Deliverable: a working demo.** The fast/slow architecture is the means; the
> deliverable is a real-time agent you can watch play and replay. Recorded MP4 + live
> web playback + interactive website are the primary outputs; the experiments in
> [`docs/06_results.md`](docs/06_results.md) ground the demo's claims (roadmap:
> [`docs/07_next_steps.md`](docs/07_next_steps.md)).
>
> **Paper:** [arXiv:2606.24470](https://arxiv.org/abs/2606.24470) · **Website:** <https://01.me/research/latent-bridge-games>

We want agents that operate a computer like a person — read the screen, issue inputs, close
the loop — and **real-time games are the hardest case**: the agent must act every few tens of
milliseconds while pursuing a goal that needs planning over seconds. No single open multimodal
LLM does both: a **reasoning** VLM (Qwen3-VL-8B-Thinking, 8B) is ~1.5 s too slow for the ~15 Hz
control loop, while a **reactive** VLM (MiniCPM-o 4.5, 9B) has no deliberation. The fast/slow
split is the fix — Thinking Machines' [Interaction Models](https://thinkingmachines.ai/blog/interaction-models/)
make it explicit via shared text/context; we test an **open** alternative.

This project investigates whether a learned continuous-valued **latent bridge** — project the
slow model's residuals into the fast model's input-embedding space, LLaVA-style — beats the
standard **text channel** (the slow model writes a prompt suffix the fast model reads) between
two **frozen** models at matched ~8–9 B scale, so the *channel*, not a capability gap, is the
load-bearing variable. The 33 M-param bridge is the only trained component.

**Headline finding:** the latent bridge helps *if and only if* slow reasoning helps the task
(**T > F**) — `L−F` tracks `T−F` at **Pearson r = 0.93**. Tuned per channel, the latent is never
significantly worse than the text bridge and significantly better on **2 of 7** games; combining
both channels *interferes*, so couple via exactly **one**. Details below.

## 🎯 Per-game scores — 8 Atari games, 12 episodes per cell (fixed-greedy view)

The best L vs T result per game (using whichever Stage A — bare or robust —
gave the higher L score):

| Game | F | T | **L** | L vs T |
|---|---|---|---|---|
| MsPacman | 256 ± 24 | 408 ± 88 | **628 ± 341** | **+54 %** |
| Seaquest | 42 ± 19 | 63 ± 11 | **80 ± 0** | **+26 %** |
| RoadRunner | 0 ± 0 | 475 ± 160 | **608 ± 29** | **+28 %** |
| River Raid (robust SA) | 1033 ± 19 | 337 ± 77 | **612 ± 297** | **+82 %** |
| SpaceInvaders (robust SA) | 107 ± 60 | 18 ± 18 | 15 ± 0 | recovered from 0 |
| Enduro (robust SA) | 0.8 ± 1.0 | 4.9 ± 5.6 | **5.8 ± 2.5** | +18 % |
| Q*bert (robust SA) | 25 ± 0 | **125 ± 0** | 50 ± 0 | T > L (greedy; tie under tuned decoders) |
| Pong | −21 ± 0 | −21 ± 0 | −21 ± 0 | reactive floor |

> ⚠️ **These are fixed-*greedy*-decoder numbers.** The latent's advantage over
> text is decoder-specific: a full decoder sweep (greedy, τ∈{0.3…1.5}) shows it
> vanishes at every fixed sampling temperature. The honest comparison tunes the
> action decoder per channel on held-out seeds (*best-achievable*): there the latent
> is **never significantly worse than text and significantly better on 2 of 7**
> (MsPacman, RoadRunner); the other 5 are ties. See the [paper](https://arxiv.org/abs/2606.24470)
> for the decoder-robust tables. The continuous-vs-categorical hypothesis these
> per-game scores once motivated is **retired** — emission statistics do not predict
> sign(L−T) (lexical-diversity r=+0.05, n.s.).

**Current claim (decoder-robust)**: the latent bridge helps *if and only if* slow
reasoning helps the task (T > F) — `L−F` tracks `T−F` at **Pearson r = 0.93** across
7 Atari games + MetaDrive (the controlled negative). Whether to couple is a property
of the task, not the channel; if you couple, use exactly one channel (text+latent
together *interferes*, −96 % on RoadRunner).

The largest *greedy* L−T gap (+82 %) is on **River Raid** after robust Stage A (a tie
under tuned decoders). The cleanest qualitative demo is **RoadRunner**: F=0 vs L=608
(reproducible; an earlier run scored 967, but the F=0 baseline makes the magnitude
run-to-run-unstable — the L>T direction is robust).

### Stage A robustness recipe (the second-order finding)

Stage A trained on bare prompts becomes out-of-distribution when T appends a
text suffix or L prepends bridge tokens. We diagnosed this through three
SpaceInvaders interventions (random-T, expert-T, aggressive-prompt all gave
T=L=0) and confirmed by fixing it: `--suffix-prob=0.5` Stage A retraining
breaks the collapse.

**Targeted, not universal**: applying robust SA to games where L > T already
worked (MsPacman, Seaquest) *hurt* — the slight bare-prompt accuracy drop
dominated the suffix-robustness gain. The recipe is:
- Use robust SA when T/L collapse to ~0 (SI, RR-bare, Q*bert, Enduro)
- Don't use it when T/L already win (MsPacman, Seaquest)
- Surprise: RoadRunner F=0 under bare SA was overfitting, not policy
  stuckness — robust SA recovered F to 958 and all three strategies tie (~925-1000)

### Detailed SpaceInvaders breakdown (diagnostic chain)
| Strategy | bare Stage A | robust Stage A |
|---|---|---|
| F | 105 ± 0   | 107 ± 60 |
| T | **0 ± 0** | **18 ± 18** |
| L | **0 ± 0** | **15 ± 0** |

The L=T=0 collapse was diagnosed across four interventions (random-T,
expert-T, aggressive-prompt, all gave 0; robust-Stage-A retry recovered both
T and L to nonzero). Bridge MI under expert-T was +0.024 nats — the bridge
*did* learn structure — but the deployed policy still collapsed because of
the action head's OOD-brittleness, not the bridge itself.

- L > T claim: under fixed greedy decoding, 4-of-7 games; under the
  decoder-robust best-achievable comparison (tune decoder per channel on
  held-out seeds), the latent significantly wins 2-of-7 (MsPacman, RoadRunner),
  ties the other 5, and never significantly loses.
- SpaceInvaders diagnosis end-to-end validated: L=T=0 under bare Stage A;
  L=15, T=18 under robust Stage A. The bridge mechanism was never broken;
  Stage A OOD-brittleness was. PPO under deployment distribution
  is the next step to close the F-L gap.

### Slow-only S baseline (MsPacman, n=3)
| Strategy | Score | Comment |
|---|---|---|
| **S (slow only, ~1 Hz)** | 113 ± 24 | Just use the big model: 4 s/decision, too slow |
| F (fast only, 15 Hz) | 256 ± 24 | Reactive, no planning |
| T (text bridge) | 408 ± 88 | Slow guides fast via text |
| **L (latent bridge)** | **628 ± 341** | Slow guides fast via latents |

On MsPacman the ordering is **S < F < T < L** — slow-only is too slow for real-time,
fast-only lacks planning, and both bridges help (the latent most, under greedy decoding).
This is *not* a universal law: whether either bridge beats fast-only is task-dependent
(the **T > F** predictor), and the latent's edge over text is decoder-specific (see the
caveat above).

The full story (v1 cross-attn → v2 LLaVA-style redesign) is in
[`docs/06_results.md`](docs/06_results.md).

## What works (v2)

The latent bridge is **LLaVA-style**: the slow model produces N=8 latent tokens in the
fast model's 4096-d input embedding space; they are **prepended to the fast model's
input sequence** so all 36 LLM layers attend over them through the standard causal
attention path. This is the same architectural pattern that LLaVA, BLIP-2, Flamingo,
and MiniCPM-o itself use for multimodal coupling.

```
slow model (Qwen3-VL-8B-Thinking)
   └─ residuals at layer 24, last N=8 positions
       └─ ThoughtProjection (4096 → 4096 → 4096 + LayerNorm, ~33M params trainable)
            └─ N=8 latent tokens in fast model's embedding space
                 └─ PREPENDED to fast model's input embedding sequence
                      └─ fast model (MiniCPM-o 4.5) — all 36 LLM layers attend
                           └─ action_head on last hidden state → action logits
```

## What didn't work (v1)

The original design tried mid-layer cross-attention into a 256-d ring buffer at LLM
depths 12 & 24, with 71M trainable fast-side params. This **converged to KL=0.004 on
training data but failed at deployment** (L=225 vs F=256, bimodal with 4/12 catastrophic
episodes). Three architectural variants (ungated / gated / gated+head-tune) all failed.

Why v1 failed:
1. **No inductive bias** for arbitrary 256-d vectors — the LLM had no pretraining for
   that format and ~5K Stage C samples wasn't enough to learn one from scratch.
2. **Information bottleneck** — only 2 of 36 layers saw the bridge; the rest had to
   propagate it via the residual stream alone.

v2 solves both by matching the LLM's input-embedding pattern (text-like inductive bias)
and using the full attention stack.

## Hypotheses

- **H1**: Latent bridge ≥ text bridge on games needing both reflex + planning.
  **✅ Decoder-robust**: tuned per channel (held-out decoder selection), the latent is
  never significantly worse than text and significantly better on 2/7 (MsPacman,
  RoadRunner). The original greedy "L > T on 4/7" over-credited the latent — the
  advantage is greedy-specific (see the decoder-sensitivity note above).
- **H2**: Latent-vs-text gap *grows* with strategic complexity. **❌ Refuted** — emission
  statistics don't predict sign(L−T) (lexical-diversity *r* = +0.05, n.s.; the
  continuous-vs-categorical hypothesis is retired). What *does* gate the bridge is the
  behavioral predictor: it pays off iff slow reasoning beats reaction on the task (**T > F**).
- **H3**: A frozen base + a small trained channel recovers most of a unified upper bound.
  **✅ Confirmed**: only the ~33 M-param slow-projection bridge trains; both base models are
  frozen. (The latent's ceiling is its text teacher — Stage C distills L toward T.)

## Repo layout

```
latent-bridge-games/
├── README.md
├── docs/
│   ├── 01_framing.md            # research thesis + scope
│   ├── 02_related_work.md       # surveyed prior art
│   ├── 03_experiment_plan.md    # experiment plan (updated for v2)
│   ├── 04_architecture.md       # v2 LLaVA-style bridge spec
│   ├── 05_status.md             # session-by-session findings log
│   ├── 06_results.md            # paper-style results summary
│   └── 07_next_steps.md         # roadmap
├── paper/                          # LaTeX source + generated figures (main.pdf)
├── web-react/                      # interactive website source (Vite/React)
├── src/
│   ├── env/atari_wrapper.py     # ALE wrapper + MsPacman/Frostbite/Seaquest RAM decoders
│   ├── models/fast_model.py     # MiniCPM-o + v2 bridge-token prepend + vision cache
│   ├── models/slow_model.py     # Qwen3-VL-8B-Thinking + trainable ThoughtProjection
│   ├── bridge/ring_buffer.py    # (v1 legacy; unused in v2)
│   ├── training/
│   │   ├── stage_a_behavioral.py     # Stage A imitation (frozen base + action_head only)
│   │   ├── stage_c_v2.py             # v2 Stage C: trainable slow ThoughtProjection only
│   │   ├── stage_c_bridge.py         # (v1 legacy: cross-attn KL training)
│   │   ├── prompts.py                # per-game Stage B text prompts
│   │   └── imitation_data.py         # global 18-way action space + per-game maps
│   └── eval/
│       ├── benchmark.py              # multi-strategy multi-game eval harness
│       └── mi_diagnostic.py          # bridge information-content diagnostic
├── configs/                          # YAML run configs
├── scripts/
│   ├── collect_trajectories.py       # SB3-expert trajectory collection (CPU)
│   ├── run_text_bridge_baseline.py   # T-trajectory collection (saves v2 raw residuals)
│   ├── aggregate_results.py          # multi-eval comparison table
│   ├── make_figures.py               # paper-quality matplotlib figures
│   ├── overnight_pipeline.sh         # autonomous overnight orchestration
│   └── ...
├── tests/                            # 45 unit + integration tests (CPU-runnable)
├── results/                          # per-condition raw eval outputs
└── checkpoints/                      # Stage A + Stage C trained checkpoints
```

## Reproducing the headline number

> Trained checkpoints and `results/` are **not shipped** (both are gitignored). Train the
> Stage A behavioral-cloning policy first (`python -m src.training.stage_a_behavioral ...`;
> the per-game `scripts/*_pipeline.sh` run the full A→C→eval chain end-to-end). The steps
> below assume `checkpoints/stage_a/` already exists.

```bash
# 1. Collect T-trajectories (45 min on GPU)
HF_HUB_OFFLINE=1 python scripts/run_text_bridge_baseline.py \
    --game MsPacman --episodes 10 --ticks 750 \
    --out-dir results/t_trajectories_v2

# 2. Train v2 Stage C bridge (12 min on GPU)
HF_HUB_OFFLINE=1 python -m src.training.stage_c_v2 \
    --trace 'results/t_trajectories_v2/MsPacman_seed*.pt' \
    --stage-a-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --out checkpoints/stage_c/v2_mspacman.pt

# 3. F/T/L head-to-head eval (~60 min for 36 episodes)
HF_HUB_OFFLINE=1 python -m src.eval.benchmark \
    --strategies F T L --games MsPacman --seeds 0 1 2 --episodes 4 \
    --fast-ckpt checkpoints/stage_a/mspacman_sb3dqn_v2.pt \
    --bridge-ckpt checkpoints/stage_c/v2_mspacman.pt \
    --out results/eval_v2_mspacman.json

# 4. Aggregate + plot
python scripts/aggregate_results.py
python scripts/make_figures.py
```

## Hardware

NVIDIA RTX Pro 6000, 96GB VRAM. Joint inference of MiniCPM-o 4.5 (bf16, 18GB) and
Qwen3-VL-8B-Thinking (bf16, 17GB) leaves ~60GB headroom for training/PPO batches.
A single scaling ablation with Qwen3-30B-A3B-Thinking (~60GB) fits at inference with
~10GB headroom; would need activation checkpointing or AWQ-4bit for joint training.

## Status

- [x] Joint inference validation (34GB VRAM, ~270ms cold tick)
- [x] Stage A behavioral cloning + the **v2 Stage C latent bridge** (the core channel)
- [x] Stage B text-bridge baseline (T = +59 % over F on MsPacman)
- [x] Cross-game sweep: 7 Atari games + MetaDrive (the controlled negative)
- [x] Stage A robustness recipe (`--suffix-prob=0.5`) — validates the OOD-brittleness diagnosis
- [x] Decoder-sensitivity sweep + held-out **best-achievable** per-channel selection (2-of-7 sig. wins)
- [x] Combined-channel (T+L) experiment — *interferes*; couple via exactly one channel
- [x] **Behavioral predictor**: L−F tracks T−F, r = 0.93 (0.96 over all 16 cells)
- [x] Bridge-replacement control (learned content tracks T > F)
- [x] 30B-A3B cross-scale ablation (more slow-model capacity does **not** widen L−T)
- [x] Latent token-count (N) ablation N=4/8/16 (deploy-only N=16 best — no capacity ceiling)
- [x] MI diagnostic; vision-token cache (latency option)
- [x] Recorded MP4 demos + interactive website (`web-react` → `web-dist`)
- [ ] Stage D PPO (online RL; will it recover SpaceInvaders?)
- [ ] Slow-only S + Oracle O baselines tabulated for all games
- [ ] Scale to the motivating target: real-time computer-use / game agents on phone & desktop

## License

MIT License — see [`LICENSE`](LICENSE). Copyright (c) 2026 Pine AI.
