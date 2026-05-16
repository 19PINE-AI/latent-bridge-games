# Project status

## Completed (initial scaffold)

- [x] Repo structure (`src/`, `configs/`, `docs/`, `scripts/`, `tests/`)
- [x] Planning docs (framing, related work, experiment plan, architecture)
- [x] Code skeleton:
  - [x] `src/env/atari_wrapper.py` — ALE env wrapper with 15Hz fast-tick + 1Hz text-state hook
  - [x] `src/models/fast_model.py` — MiniCPM-o wrapper + BridgeCrossAttention layer (zero-init)
  - [x] `src/models/slow_model.py` — slow-model wrapper + ThoughtProjection head
    (primary: Qwen3-VL-8B-Thinking; scaling ablation: Qwen3-30B-A3B-Thinking)
  - [x] `src/bridge/ring_buffer.py` — ThoughtBuffer + PerceptionBuffer + age encoding
  - [x] `src/training/stage_{a,c,d}_*.py` — stage entry points (NotImplementedError stubs)
  - [x] `src/eval/benchmark.py` — eval harness stub
- [x] Configs: `bridge.yaml`, `fast_model.yaml`, `slow_model.yaml`, `training.yaml`, `eval.yaml`
- [x] Tests: `tests/test_ring_buffer.py` + `test_env_buffer_integration.py` +
  `test_ram_decoders.py` + `test_prompts.py` — 18 passing
- [x] Scripts: `scripts/validate_hardware.py`, `scripts/random_baseline.py`,
  `scripts/collect_trajectories.py`, `scripts/show_text_state.py`

## Completed (research execution, in order)

- [x] Established random-policy floor: MsPacman 290 ± 134, Frostbite 78 ± 37, etc.
  (matches published Atari-random baselines). Stored at `results/random_baseline.json`.
- [x] Researched authoritative ALE RAM layouts for MsPacman + Frostbite
  (sources: AtariARI, OCAtari, ALE source).
- [x] Implemented + tested RAM decoders in `atari_wrapper.py`. Score correctness is
  guarded by a regression test (`test_*_score_matches_reward`) that compares decoded
  score to ALE's cumulative reward.
- [x] Discovered + fixed an LSB/MSB byte-order bug: AtariARI lists score bytes as
  120/121/122 but stores them LSB-first. The agent-supplied formula had the wrong order
  *and* a spurious ×10 multiplier. Now decoded empirically.
- [x] Slow-model prompt template (`src/training/prompts.py`) for T condition, per-game.
- [x] Swapped primary slow model to Qwen3-VL-8B-Thinking (same-scale endpoints);
  Qwen3-30B-A3B-Thinking retained as a single Frostbite scaling ablation.
- [x] Stage A imitation dataset (`src/training/imitation_data.py`) with a unified
  18-way action space and per-game legal-mask. Maps every game's local action indices
  (MsPacman: 9, Pong: 6, Frostbite/Pitfall: 18) onto the global ALE-canonical space.
- [x] Trajectory collector (`scripts/collect_trajectories.py`) upgraded to accept an
  SB3 DQN/PPO expert via `--expert-policy <path.zip>` with epsilon-greedy exploration.
  Companion SB3-preprocessed env stepped in lockstep so the expert sees its training
  observation while we record raw RGB.
- [x] Verified `Qwen3-VL-8B-Thinking` config loads via `AutoConfig` (no
  trust_remote_code; native in transformers 4.57+). Text hidden size = 4096, 36 layers,
  vision encoder hidden = 1152. Slow-model loader updated to use
  `AutoModelForImageTextToText` + `AutoProcessor`.
- [x] **Hardware gate passed.** Joint inference of MiniCPM-o 4.5 + Qwen3-VL-8B-Thinking
  fits in 33.9GB VRAM (17.6GB fast + 16.3GB slow), matching the architecture-doc estimate
  (~44GB conservative). Joint forward+gen latency on slow side: 0.85s for 8 tokens.
- [x] **`SlowModel.emit()` wired end-to-end.** Loads from local cache (HF_HUB_OFFLINE
  bypasses transformers' `_patch_mistral_regex` bug that ignores `local_files_only`),
  accepts the Stage-B chat-format messages, attaches the raw RGB frame as a vision
  content block, runs `generate(output_hidden_states=True)`, extracts the residual
  stream at layer 24 for each generated token, and projects through the
  256-dim ThoughtProjection head. Per-emission validated on a real MsPacman
  RAM-decoded state: model produces semantically coherent reasoning over ghost
  positions and Δ vectors; per-token L2 norms ∈ [2.8, 3.9]; per-dim std = 0.13
  (non-collapsed). Throughput: ~41 tokens/s → ~40-token emissions fit a 1Hz budget.
- [x] Required workaround: stubbed `minicpmo` package at site-packages — MiniCPM-o-4.5's
  bundled `utils.py` has a non-relative `from minicpmo.utils import ...` that
  transformers' static import-check trips on; the actual call site is a lazy video
  helper we never invoke. PyPI `minicpmo` is broken (metadata="unknown" + only audio
  code); the stub satisfies the static check without affecting image inference.
- [x] **🎯 v2 HEADLINE RESULT — LLaVA-style latent-as-token bridge succeeds**
  (`results/eval_v2_mspacman.json`, 12 episodes per cell, MsPacman):

  | Strategy | Mean ± Std | Median | n |
  |---|---|---|---|
  | F (fast only) | 256 ± 24 | 250 | 12 |
  | T (text bridge) | 408 ± 88 | 385 | 12 |
  | **L (v2 latent bridge)** | **628 ± 341** | **550** | 12 |

  - **L v2 vs F: +145% mean lift** (256 → 628)
  - **L v2 vs T: +54% mean** (408 → 628), **+43% median** (385 → 550)
  - Best single L episode: **1740** (far above the top of any prior condition)

  This validates the user's two-part diagnosis that defeated v1:
    1. The latent bridge needed an LLM-pretraining-compatible inductive bias
       (4096-d tokens in the LLM's own input-embedding space, not arbitrary
       256-d vectors).
    2. The latent bridge needed full-stack attention (all 36 LLM layers), not
       mid-layer cross-attn at only depths 12 and 24.

  v2 architecture (the standard LLaVA / BLIP-2 / MiniCPM-o pattern):
    - Slow model: N=8 token positions from its emission, each projected through
      a trainable MLP `4096→4096→4096` with LayerNorm bookends (33.6M params).
    - Fast model: bridge tokens **prepended to the input embedding sequence**.
      All 36 LLM layers attend over them via the standard causal attention path.
    - Stage C training: only the slow's ThoughtProjection is trainable; fast
      model entirely frozen. KL(student||teacher) loss converges to mean=0.026.

  This is the central paper result. The text channel does work (+59% over F),
  but the latent channel **also works and works better than text** when given
  LLM-native architectural privileges.

- [x] **EXPANDED EVAL: three L variants vs F vs T** (each 12 ep, MsPacman):

  | Variant | Mean ± Std | Median | Failure mode |
  |---|---|---|---|
  | F | 256 ± 24 | 250 | (baseline) |
  | T | **408 ± 88** | 385 | (best — +59% over F) |
  | L ungated (baseline) | 225 ± 85 | 255 | **Bimodal**: 4/12 episodes scored 70-160 (catastrophic); when not catastrophic ≈ F |
  | L gated only | **256 ± 110** | 255 | No catastrophes; bridge mostly silent → matches F |
  | L gated + action-head tune | 210 ± 0 | 210 | **Mode collapse** — joint KL+tune has a degenerate solution; ALL F/T/L become identical deterministic policy |

  **What we learned about why L fails on-policy despite KL=0.004 on offline data:**

  1. **The bridge perturbs hidden states at layers 12/24** in a way the frozen Stage A
     action_head can't read robustly. In ~33% of episodes this causes a deterministic
     bad action sequence that ends the episode prematurely.
  2. **Gating fixes the catastrophe but not the lift**. With a learnable sigmoid gate
     starting at sigmoid(-2)≈0.12, the action_head can suppress bad bridge outputs by
     closing the gate. Catastrophic episodes disappear (no scores <100). But the gate
     stays mostly closed everywhere, so the bridge contributes near-zero on average and
     L ≈ F (256 vs 256).
  3. **Joint action_head + bridge KL training has a degenerate solution**: both can
     collapse to the same deterministic output, satisfying KL=0 perfectly without
     learning anything useful. Observed empirically: all F/T/L cells became identical
     210/375 episodes after this training. **Do not unfreeze action_head during KL
     training without entropy regularization or labeled CE supervision.**

  **Real next step**: the underlying issue is offline-vs-online distribution shift
  (the Stage C training distribution doesn't include states the L policy visits) AND
  the action_head being mis-calibrated for bridge-perturbed hiddens. Both are exactly
  what Stage D (online RL) is designed for. DAgger alone won't fix #2; PPO with
  bridge in the loop will.
- [x] **HEADLINE EVAL RESULT — MsPacman, 12 episodes per cell (3 seeds × 4 eps)**
  (`results/eval_full.json`):

  | Strategy | Mean ± Std | Median | Latency | On-Clock |
  |---|---|---|---|---|
  | F (fast-only) | 256 ± 24 | 250 | 44ms | 81% |
  | **T (text bridge)** | **408 ± 88** | 385 | 50ms | 80% |
  | L (latent bridge) | 225 ± 85 | 255 | 64ms | 71% |

  **H1 confirmed for T**: text-bridge gives +59% over fast-only (256 → 408). Strong
  effect well outside the noise envelope.

  **H1 REJECTED for L on this configuration**: latent bridge is bimodal — median
  matches F (255 vs 250), but several episodes die early producing 70-160 scores.
  The Stage C bridge converged to KL=0.004 on offline T-rollout frames, but at
  deployment L visits states T never visited. Distribution-shift / compounding
  error: the textbook offline-imitation failure mode.

  **Asymmetric vulnerability**: T appends text as a *prompt suffix* that the model
  can choose to ignore; L modifies the residual stream additively at layers 12+24
  so the model can't escape a bad bridge output.

  **Implications for the paper**: the bridge architecture works distributionally
  (KL is tiny) but fails on-policy. The plan's Stage D — online RL fine-tuning of
  the bridge in deployment — is now clearly the load-bearing next step, not a
  nice-to-have. DAgger-style on-policy bridge correction is the obvious cheaper
  alternative.

  **Latency is fine**: all three conditions are at or under the 67ms 15Hz target
  when warm (44/50/64 ms, 71-81% on-clock).
- [x] **Frostbite T-trajectories**: 10 episodes collected (`results/t_trajectories/Frostbite_seed{0..9}.pt`),
  mean score 55 ± 20 (matches published random baseline). Stage C-Frostbite and
  L-vs-T-Frostbite eval pending (Frostbite has no SB3 zoo expert; would need to
  train a Stage A teacher first).
- [x] **Full pipeline validated end-to-end on real GPU (this session, GPU rebooted clean).**
    1. T-condition (slow + fast joint): 75 ticks of MsPacman in 10.6s wall-clock, 5
       slow emissions at 1.5s each (~64 tok/s, fits 1Hz budget). Joint VRAM 36.5GB.
       Each emission produces 96-token text + 256-dim thought-vector stream. Trajectory
       persisted to `results/t_episode_mspacman_seed0.pt` (9.7MB).
    2. Stage C2 KL training on that trajectory: 61 valid steps in 6s. First run hit
       a NaN bug — `0 × -inf` when illegal-action positions intersected with the
       KL formula. Fixed by computing KL only over legal-action indices; regression
       test added (`test_kl_loss_handles_legal_mask_without_nan`).
    3. Stage C signal test (perturbed action_head + non-zero bridge): KL = 0.923,
       8/8 bridge xattn params get non-zero gradients, action_head gradients flow.
       Confirms the central novel architectural element (cross-attn injection at
       layers 12/24 of the frozen Qwen3-8B backbone) trains correctly under signal.
- [x] **Stage C COCONUT-curriculum scaffold** (`src/training/stage_c_data.py`,
  `src/training/stage_c_bridge.py`). Dataset reads T-condition trajectories and
  flat-indexes per-tick samples with the most-recent active emission attached
  (slow_text + slow_vecs + emission-age). Trainer runs *two* fast-model forwards
  per step: teacher (full text, no bridge) and student (truncated text + latent
  buffer); loss = KL(student || teacher) over the 18-way action distribution.
  Curriculum: C0 (text only), C1 (drop second-half text, fill bridge with
  second-half latents), C2 (drop all text, full latents). 7 new CPU tests cover
  the dataset indexing, emission-walkback, curriculum text-truncation, and the
  thought-buffer materialization (with padding for short emissions).
  Bridge-only optimizer (xattn Q/K/V/O at layers 12, 24). Awaiting T-condition
  trajectory data to run the actual KL training (T runtime is GPU-blocked).
- [x] **Stage A behavioral cloning loop end-to-end** (`src/training/stage_a_behavioral.py`).
  Cross-entropy on (frame, action) with legal-action masking, AdamW on
  action_head + bridge xattn weights, train/val split, checkpoint serialization.
  Smoke run: 24 train steps + 6 val on a 30-sample random-policy trajectory loaded
  cleanly through forward/backward/optimizer/eval/save. 71M trainable parameters
  (bridge xattn dominates; action head is only 74K). Val accuracy 0% on random data
  is the correct null result.
- [x] **T-condition runtime** (`scripts/run_text_bridge_baseline.py`) code-complete.
  F-condition validated end-to-end on GPU: 15 ticks MsPacman in 4.2s wall-clock.
  Action-space inverse mapping `global_to_local_action()` added — global 18-way
  argmax/sample passes through to ALE's per-game local indices at the env boundary.
  Full T (slow + fast joint, ~35GB) blocked on GPU contention with other workloads.
- [x] **Latency profile** of `FastModel.predict_action`:
    - `max_slice_nums=2` (default): **279ms** end-to-end
    - `max_slice_nums=1` (locked in): **197ms** (saves 82ms — fewer image slices for
      the SigLIP+resampler to encode; Atari 210x160 doesn't need multi-slice anyway).
    - Per-stage at slices=1: processor 14ms, host→device 42ms, vision tower 80ms,
      LLM forward 52ms, action head <1ms.
    - **torch.compile** on `llm.model`: extra **1.33× speedup**, mean 53→40ms
      (min 30ms). Defer to opt-in flag — compile interacts unpredictably with
      backward pass; safe to enable for inference (Stage D PPO rollouts, eval).
  At 10Hz (100ms target) we're within reach with these two wins; sub-67ms (15Hz)
  needs vision-tower work (lower input resolution, or per-tick caching).
- [x] **`FastModel.predict_action` end-to-end pipeline working.** Atari frame (RGB
  210×160) → MiniCPM-o `processor` (4-list signature: prompt strings, images, audios,
  audio-parts) → `input_ids[1, 90]` + `pixel_values[1][1][3, 14, 14504]` + image_bound →
  `model.get_vllm_embedding()` fuses 64 SigLIP+resampler visual tokens into the text
  embedding stream → `model.llm.model(inputs_embeds=...)` with **bridge hooks firing**
  at layers 12 and 24 → `action_head(last_hidden)` → `[1, 18]` logits → legal-mask sets
  illegal actions to -inf. Latency: **355ms cold**, well over the 67ms 15Hz target;
  optimization (KV-cache reuse, smaller `max_slice_nums`, torch.compile) is follow-up
  work. Required workaround: download `tokenization_minicpmo_fast.py` + tokenizer.json
  + vocab.json separately via `hf_hub_download` (the initial model snapshot doesn't
  include them).
- [x] **`FastModel` cross-attention injection wired and validated end-to-end.**
  Probed MiniCPM-o live structure: top-level `MiniCPMO`, LLM at `.llm.model.layers`
  (36 Qwen3DecoderLayers, hidden=4096), vision tower at `.vpm`/`.resampler`, plus
  unused audio/TTS components. `BridgeCrossAttention` modules registered at layers 12
  and 24 via `register_forward_hook` (post-block). Smoke tests:
    1. **Zero-init no-op:** with `o_proj.weight = 0`, max |h_with_bridge - h_base| =
       **0.000000** — bridge surgery doesn't perturb the frozen model at init.
       Stage A imitation can train the action head alone with the bridge present-but-silent.
    2. **Non-zero bridge changes output:** randomizing one `o_proj` produces a 0.625
       max diff in hidden state — bridge is genuinely plumbed into the residual stream.
    3. **Action head:** `[B, T, 4096] → [B, 18]` produces correct-shape logits.
  VRAM with MiniCPM-o + bridge attached: 19.0GB.

## Next concrete steps (Week 1, blocked items first)

1. **BLOCKED on GPU availability:** Run `python scripts/validate_hardware.py` to
   confirm joint inference of MiniCPM-o + Qwen3-VL-8B-Thinking fits (~44GB estimated).
   The GPU is currently at ~100GB allocated by other experiments on this machine.
2. Wire up `FastModel.forward()` — hook MiniCPM-o's backbone forward and inject
   cross-attention at layers 12 and 24. Requires reading `openbmb/MiniCPM-o-4_5`
   modeling source to identify the hookable forward signature.
3. Wire up `SlowModel.emit()` — generate text + extract residual stream at layer 24
   (Qwen3-VL-8B) + apply projection head. The prompt template lives in
   `src/training/prompts.py`.
4. Implement Stage A imitation-learning data pipeline:
   - Decide expert source (random / scripted heuristic / pretrained DQN / self-distill).
     Current placeholder: trajectory collector at `scripts/collect_trajectories.py`
     emits random policy. For Stage A we need stronger trajectories.
   - Action-token discretization: see decision below.
5. Run a small slow-model-only smoke test using the cached
   `Qwen/Qwen3-VL-8B-Thinking` (already on disk) to produce one T-condition emission
   from a decoded text-state — verifies the prompt template + emit path end-to-end.

## Design decisions

### Action head: classification head on last hidden state — DECIDED

MiniCPM-o emits text tokens. To produce an ALE action at every fast tick (15Hz) we need
a discrete output of size `n_actions` (max 18; MsPacman uses 9, Frostbite 18). Considered:

1. **Classification head on last hidden state (chosen).** A `Linear(hidden_dim, 18)` on
   top of the Qwen3-8B backbone's final hidden state, sampled at the position
   corresponding to a fixed "[ACTION]" sentinel token appended to the prompt. Trained
   with cross-entropy in Stage A.
2. **Constrained decoding to 18 action-name tokens.** Cleaner if you want the model's
   text-generation behavior unaltered, but expensive: needs ~1 token of generation per
   tick + tokenizer-side action-name handling.

**Decision:** option (1). Rationale: we're already inserting custom cross-attn layers
into the backbone for the bridge, so adding a classification head is consistent with the
existing surgery. It also makes Stage D PPO straightforward — the policy is a 18-way
categorical with log-probs read directly from the head's softmax. The full 18-way space
is used for all games (with masked sampling on the legal subset per game), so a single
head supports the entire eval sweep.

Implementation note: head is initialized to zero so the model behaves as base + uniform
random at init, and Stage A imitation supervision is the only signal that shapes it.

### Slow-model emission policy: continuous 1Hz — DECIDED

Continuous emission at 1Hz is simpler, makes the curriculum cleaner (every C0 tick has a
text emission to align with), and the perception buffer cross-attn handles redundancy
naturally. Event-triggered emission is a future-work ablation.

### Stage D: PPO with imitation-only fallback — DECIDED

PPO is the standard choice. If it diverges in week 3, fall back to extended Stage C
imitation training, which we expect to recover most of the L–T gap on Tier 1-2 (less
clear on Tier 3 where exploration matters more).

### Stage A imitation-learning expert source — OPEN

Stage A needs an expert policy to imitate. Options:
- Random policy (current): fast but caps Stage A at the random floor, providing no signal
  about whether MiniCPM-o can learn action prediction at all.
- Pretrained DQN/PPO from Stable-Baselines3 model zoo: standard, reliable, but introduces
  an external model dependency and ties scores to that policy's habits.
- MiniCPM-o zero-shot with constrained decoding: distills from the model itself; may
  produce low-quality data on novel games.
- Scripted heuristic (e.g., "always head toward nearest dot"): cheap but game-specific.

**Tentative direction:** SB3 pretrained DQN as expert for Stage A; this gives a clean
upper-target signal independent of MiniCPM-o's prior. Decision deferred until after the
hardware-validation gate.

## Risk log

(Open at project start. Update as risks materialise.)

| Date | Risk | Status |
|---|---|---|
| 2026-05-15 | OOM on joint inference (primary 8B+9B config) | Very unlikely (~44GB est); untested |
| 2026-05-15 | MiniCPM-o cross-attn injection is harder than expected | Untested |
| 2026-05-15 | F-only saturates on chosen games | Untested |
| 2026-05-15 | Same-size slow model lacks reasoning depth → small L–T gap on Tier 3 | Open; 30B-A3B scaling ablation is the disambiguation |
