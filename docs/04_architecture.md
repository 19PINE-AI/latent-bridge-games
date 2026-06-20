# Architecture

## **v2 design (current) — LLaVA-style latent-as-token bridge**

The original v1 design (mid-layer cross-attention into the frozen LLM at depths 12/24,
with a 256-dim thought buffer) was empirically shown to fail at deployment despite
converging to KL=0.004 on offline data. Diagnosis: the bridge had no learned inductive
bias (random-init 256-d vectors with ~5K training samples), and information only
entered the LLM at two specific layers — most of the 36-layer stack saw nothing.

**v2 fixes both issues** by adopting the standard multimodal pattern used by LLaVA,
BLIP-2, MiniCPM-o, Qwen-VL, etc.: the slow model's residuals are projected into the
*fast model's input embedding space* (4096-d, not 256-d) as *N latent tokens* (default
N=8), and these are **prepended as actual tokens to the fast model's input sequence**.
The LLM processes them through every one of its 36 layers with full attention, exactly
the same path text tokens take. The frozen LLM thus inherits its enormous text-pretraining
inductive bias for handling sequence tokens of arbitrary content — we are not asking it
to learn a new modality from scratch with cross-attention surgery.

```
┌─────────────────────┐         ┌────────────────────────────────────┐
│  Atari game state   │  60Hz   │   Slow Model: Qwen3-VL-8B-Thinking │
│  (RGB frame +       │ ──────▶ │   (frozen base; trainable          │
│   game-state text)  │         │    ThoughtProjection: 4096→4096)   │
└────────┬────────────┘         │   1-2 Hz emission                  │
         │                      └────────────┬───────────────────────┘
         │                                   │
         │ 15Hz visual                       │ N=8 latent tokens
         │                                   │ in fast-model embedding
         │                                   │ space (D=4096)
         ▼                                   ▼
┌─────────────────────────────────────────────────────────┐
│  Fast Model: MiniCPM-o 4.5  (Qwen3-8B backbone)         │
│                                                         │
│  inputs_embeds = [bridge_tokens(N×4096) ;               │
│                    vision_tokens(64×4096) ;             │
│                    text_tokens(...×4096)]               │
│                                                         │
│  ▶ ALL 36 LLM LAYERS attend over bridge tokens          │
│  (same path text uses — no mid-layer surgery)           │
│                                                         │
│  Last hidden state → action_head → action_logits[18]    │
└────────┬────────────────────────────────────────────────┘
         │ action token at 15Hz
         ▼
┌─────────────────────┐
│  ALE step           │
│  reward, next frame │
└─────────────────────┘
```

### What changed from v1

| Aspect | v1 (deprecated) | v2 (current) |
|---|---|---|
| Bridge entry point | Cross-attn at LLM layers 12 & 24 | **Input embedding (all 36 layers attend)** |
| Bridge dim | 256 | **4096 (LLM hidden size)** |
| Bridge tokens per emission | 16-entry ring buffer, 256-d each | **N=8 tokens, 4096-d each (latest emission only)** |
| Trainable bridge params | Q/K/V/O cross-attn (71M) | **ThoughtProjection only (~33M, slow side)** |
| LLM modification | Mid-layer cross-attn injection | **None — just concat tokens to input** |
| Inductive bias | None (random 256-d) | **Inherits LLM's full sequence-token prior** |

### Outcome on MsPacman (v1)

| Strategy | Mean ± Std | Notes |
|---|---|---|
| F (fast only) | 256 ± 24 | baseline |
| T (text bridge) | **408 ± 88** | +59% over F |
| L (v1 latent) | 225 ± 85 | bimodal: 4/12 catastrophic |

The text channel's lift is large and reliable; the v1 latent bridge fails. v2 is
designed to close this gap by giving the latents the same architectural privileges
text enjoys.

## Fast model: MiniCPM-o 4.5

- 9B params total (SigLip2 + Whisper-medium + CosyVoice2 + Qwen3-8B backbone)
- Frozen base weights
- Trainable adapters:
  - LoRA on Qwen3-8B backbone's attention layers (rank 16, alpha 32) — for action-head
    behavior adaptation
  - **No bridge modules on the fast side (v2)**: the latent bridge enters as prepended
    input tokens, so the fast model needs no added cross-attention layers. (The v1 design
    inserted dedicated cross-attention layers at depths 12 and 24; these were removed in
    v2 — see "What changed from v1" above.)
- Input modalities used: video frames at 15Hz (Atari 84x84 RGB upsampled to 224x224)
- Output: action token (one of 18 ALE actions), emitted every 67ms

**Why prepended tokens rather than mid-layer cross-attn (v2):** prepending latent tokens
to `inputs_embeds` lets all 36 LLM layers attend over the bridge through the exact code
path text uses, inheriting the frozen LLM's full sequence-token prior. The v1 cross-attn
approach exposed the bridge to only 2 of 36 layers and had no pretraining inductive bias,
and it failed at deployment.

## Slow model: Qwen3-VL-8B-Thinking

- ~8B params, dense
- Native vision encoder (SigLIP-class) — accepts 1Hz frame snapshots directly
- bf16, frozen base
- Trainable adapters:
  - LoRA on attention projection layers (rank 16) — for adapting thought emission to
    game-context patterns
  - **ThoughtProjection** (v2): MLP `4096 → 4096 → 4096` with LayerNorm on output.
    Maps slow model's residual at layer 24 to bridge tokens in the **fast model's
    input embedding space**. Takes the last `N=8` token positions from the slow's
    emission (these contain the slow model's "conclusion" closest to its answer).
    This is the **only trainable bridge parameter** in v2 (~33M params, a 2-layer MLP).
  - **Cross-attention LoRA** at layer 20: reads the perception buffer (Stage C3 only)
- Input: 1Hz frame snapshot + rolling 4-second window of (a) compact text game state
  (RAM-derived entity positions, score, lives, level) and (b) perception summaries from
  fast model
- Output: text reasoning tokens (with `<think>` block) + N=8 latent tokens at ~1-2Hz

**Why same-architecture-family on both sides:** MiniCPM-o 4.5's backbone is Qwen3-8B and
Qwen3-VL-8B-Thinking is built on the same family. **In v2 this is critical**: the slow's
hidden_dim (4096) equals the fast's hidden_dim (4096), so the projection is dimension-
preserving rather than cross-family. The fast model's input embedding space is therefore
*compatible* with the slow's residual stream — the projection just needs to learn a
selective re-encoding, not a cross-architecture mapping.

### Scaling ablation (single configuration, not the main experiment)

For one Tier-3 game (Frostbite), we additionally run T and L with Qwen3-30B-A3B-Thinking
as the slow model. This 30B-A3B MoE has 3B active params per token and a 5120-dim residual
stream. It uses a projection head whose input dim is 5120 (rather than 4096) but whose
output is still the fast model's 4096-d input embedding space. Purpose: test whether the
latent-vs-text gap changes with slow-model capability. See `docs/03_experiment_plan.md`
for protocol.

**Why a learned projection rather than the raw residual stream:** the slow model's residual
stream is not natively in the fast model's input-embedding space, so it cannot be prepended
as tokens directly. The trainable `ThoughtProjection` MLP re-encodes the selected residuals
into that space, trained via the COCONUT-style curriculum (reconstruction against
text-channel predictions).

## The bridge (v2 — LLaVA-style)

### Slow → fast (latent-as-token)
- Slow model emits a `<think>...</think>` response (up to ~96 tokens).
- The slow model's hidden states at layer 24 are captured for the **last N=8 token positions** of the emission (these contain the model's conclusion closest to its answer, which has the highest information density per token).
- A trainable `ThoughtProjection` MLP (4096 → 4096 → 4096 with LayerNorm) maps each
  of these N residuals into the **fast model's input embedding space** (4096-d).
- The result is `[N=8, 4096]` "latent tokens" that look statistically like text token
  embeddings to the fast model.
- These tokens are **prepended** to the fast model's `inputs_embeds` sequence (before
  the image and text tokens). All 36 LLM layers attend over them through full causal
  attention — exactly the same code path as text.
- Latent token count: N=8 by default. A latent token-count (N) ablation finds that
  matched N=8 is a Goldilocks point at matched train/deploy, but a deploy-only N=16
  scores best — i.e., the model uses far less capacity than the nominal token budget
  provides, so there is no capacity/bandwidth ceiling driving the latent-vs-text gap.

### Why we replaced the v1 ring buffer

The v1 design used a 16-entry × 256-dim ring buffer read by dedicated cross-attention
layers at LLM depths 12 and 24. This had two architectural problems empirically:

1. **No inductive bias**: 256-d vectors with arbitrary content; the model had no
   pretraining for this format and Stage C had to learn the mapping from scratch
   with only ~5K training samples.
2. **Information bottleneck at most layers**: Only layers 12 and 24 saw the bridge.
   The other 34 layers had no direct access; bridge information had to propagate via
   the residual stream alone, attenuated by intervening attention + MLPs.

The text channel (T condition) doesn't have either problem — text tokens enter at
layer 0 and are attended over by all 36 layers, and the LLM has billions of
pretraining examples teaching it how to use sequence tokens. **v2 gives the latent
bridge the same architectural privileges as text.**

### Perception buffer (fast → slow) — deferred to v2.1

Currently not implemented. Will be added analogously when bidirectional coupling is
needed.

## Memory budget (96GB target)

### Inference (primary 8B+9B configuration)
| Component | bf16 size |
|---|---|
| MiniCPM-o 4.5 weights | 18GB |
| Qwen3-VL-8B-Thinking weights | 17GB |
| KV cache (both models, 4K context each) | 4GB |
| Bridge layer weights | 0.5GB |
| Activations (peak) | 4GB |
| **Total** | **~44GB** |

### Training (Stage C / D, primary configuration)
| Component | size |
|---|---|
| MiniCPM-o frozen + LoRA | 21GB |
| Qwen3-VL-8B frozen + LoRA + projection | 19GB |
| Optimizer state (LoRA + projection only) | 2GB |
| Activations + gradients (peak, bs=4) | 8GB |
| Bridge buffers | 1GB |
| **Total** | **~51GB** |

Comfortable headroom (~45GB). Lets us use bs=4-8 for stable PPO at Stage D and keep
multiple LoRA checkpoints resident for fast switching during ablation runs.

### Scaling-ablation budget (one Tier-3 game, 30B-A3B slow)
| Component | bf16 size |
|---|---|
| MiniCPM-o 4.5 weights | 18GB |
| Qwen3-30B-A3B weights | 60GB |
| KV cache + bridge + activations | 9GB |
| **Total (inference)** | **~87GB** |

Stage D training in this configuration is tight; mitigations:
- Reduce bs to 1 with grad accumulation
- Activation checkpointing on Qwen3-30B layers
- AWQ-4bit on slow model (saves ~30GB) if needed for joint training

## Training objectives

### Stage A (fast-only, behavioral cloning)
- Loss: cross-entropy on next action given frame + recent context
- No bridge involvement

### Stage C0 (text-bridge baseline)
- No training — uses Stage A's fast model with text from slow model concatenated to
  fast-model context

### Stage C1, C2 (latent-bridge curriculum)
- Loss = α · CE(action) + β · MSE(latent-conditioned-logits, text-conditioned-logits)
- The MSE term is the COCONUT-style supervision: forces latent vectors to encode the same
  predictive content as the text they replace
- α=1.0, β=2.0 (latent reconstruction is the load-bearing signal at this stage)

### Stage D (joint RL)
- Loss: PPO clipped objective with reward = game score delta + small entropy bonus
- LoRA params on both models updated; base weights frozen
- Bridge projection LoRA on slow model also updated

## Configuration files

All hyperparameters live under `configs/`:
- `configs/fast_model.yaml` — LoRA ranks, learning rates
- `configs/slow_model.yaml` — LoRA on output proj, projection-head dim
- `configs/bridge.yaml` — latent token count (N), emission rates, capture layer
- `configs/training.yaml` — per-stage hyperparameters, optimizer, schedule
- `configs/eval.yaml` — game list, seed counts, metric definitions
