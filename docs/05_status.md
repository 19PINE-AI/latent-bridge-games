# Project status

A concise record of what was built and the load-bearing empirical findings. For
the full current results and hypothesis status see `06_results.md`; for the design
rationale see `03_experiment_plan.md` and `04_architecture.md`.

## What was built

- **Models** — frozen MiniCPM-o 4.5 (fast, 9B) + Qwen3-VL-8B-Thinking (slow, 8B),
  joint inference in ~34 GB VRAM. Only the 33M-param latent bridge is trained.
- **Fast model** (`src/models/fast_model.py`) — classification action head on the
  final hidden state (18-way ALE-canonical space, per-game legal mask), zero-init
  so Stage A imitation is the only signal that shapes it. `max_slice_nums=1` and an
  optional `torch.compile` bring warm latency under the 15 Hz budget.
- **Slow model** (`src/models/slow_model.py`) — `emit()` generates a short CoT and
  projects N=8 residual-stream tokens through the trainable `ThoughtProjection`
  (LLaVA-style, 4096-d, prepended to the fast model's input embeddings).
- **Training** — Stage A behavioral cloning (SB3 expert, ε-greedy), Stage B text
  bridge (also emits Stage C supervision), Stage C `KL(student‖teacher)` bridge
  training, Stage D PPO scaffold.
- **Eval** (`src/eval/benchmark.py`) — multi-strategy F/T/L/S/O harness, bootstrap
  CIs, decoder sweep + held-out per-channel decoder selection, MI diagnostic.

## Headline result (MsPacman, 12 episodes/cell)

| Strategy | Mean ± Std | Median |
|---|---|---|
| F (fast only) | 256 ± 24 | 250 |
| T (text bridge) | 408 ± 88 | 385 |
| **L (latent bridge)** | **628 ± 341** | **550** |

The latent-as-token bridge beats both fast-only and the text bridge on MsPacman.
This is the central paper result; the cross-game picture is in `06_results.md`.

> **Decoder caveat.** The greedy L>T edge is decoder-specific. Tuned per channel on
> held-out seeds ("best-achievable"), the latent is never significantly worse than
> text and significantly better on **2 of 7** games (MsPacman, RoadRunner); the rest
> are ties. A single fixed greedy decoder over-credits it (looks like 4 of 7).

## Key findings

- **v1 → v2 architecture.** The v1 bridge (mid-layer cross-attention into a 256-d
  ring buffer at LLM depths 12 & 24, 71M params) converged to KL≈0.004 offline but
  failed on-policy — bimodal collapse from offline→online distribution shift, plus
  an action head mis-calibrated for bridge-perturbed hidden states. The v2 redesign
  gives the latent the same privileges multimodal LLMs use (tokens in the
  input-embedding space, attended by all 36 layers) and succeeds.
- **SpaceInvaders is a clean negative (F=105, T=0, L=0).** Three convergent
  interventions (aggressive prompt, expert-T re-collection, bridge architecture)
  leave it at zero. Diagnosis: Stage A trains on bare game-state prompts, so the T/L
  suffix is OOD for the frozen action head; on reward-asymmetric SI this biases away
  from FIRE. The robustness recipe (`--suffix-prob=0.5`) validates the diagnosis.
  `KL` bridge training also bounds L above by T, so a passive teacher caps the student.
- **Couple via exactly one channel.** Using text and latent together interferes
  (hurts 3 of 7, RoadRunner −96 %).
- **Latent token count (N).** Tuned on the deploy side, N=16 is best — no capacity
  ceiling. (The earlier "bandwidth Goldilocks" reading is retired; see `06_results.md`.)

## Design decisions

- **Action head** — classification head on the final hidden state (vs constrained
  decoding to action-name tokens): consistent with the bridge surgery and makes
  Stage D PPO a clean 18-way categorical with log-probs read from the head.
- **Slow emission** — continuous at 1–2 Hz (vs event-triggered): simpler curriculum,
  and the perception buffer absorbs redundancy. Event-triggered is a future ablation.
- **Stage A expert** — SB3 pretrained DQN/PPO: a clean upper-target signal
  independent of the fast model's prior.
