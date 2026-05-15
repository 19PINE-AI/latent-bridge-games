# Architecture

## Component overview

```
┌─────────────────────┐         ┌──────────────────────┐
│  Atari game state   │  60Hz   │   Slow Model         │
│  (RGB frame +       │ ──────▶ │   Qwen3-VL-8B        │
│   game-state text)  │         │   -Thinking          │
└────────┬────────────┘         │   (frozen + LoRA     │
         │                      │    projection head)  │
         │ 15Hz visual          │   1-2 Hz emission    │
         ▼                      └──────────┬───────────┘
┌─────────────────────┐                    │
│  Fast Model         │   cross-attn       │ thought
│  MiniCPM-o 4.5      │ ◀──────────────────┤ vectors
│  (frozen + LoRA on  │   over thought-    │ (D_bridge=256)
│   attn + new        │   buffer           ▼
│   cross-attn        │             ┌──────────────────┐
│   layers)           │             │  Thought Buffer  │
│                     │             │  ring, K=16      │
│  Output: action     │  perception │  (256-dim each   │
│  token at 15Hz      │  summary    │   + age stamp)   │
└────────┬────────────┘  ─────────▶ └──────────────────┘
         │                            │
         │ action                     │
         ▼                            │
┌─────────────────────┐               │
│  ALE step           │               │
│  reward, next frame │               │
└─────────────────────┘               │
                                      ▼
                              ┌──────────────────┐
                              │ Perception Buffer│
                              │ ring, K=8        │
                              │ (256-dim each)   │
                              │ → read by slow   │
                              │   model's LoRA   │
                              │   cross-attn     │
                              └──────────────────┘
```

## Fast model: MiniCPM-o 4.5

- 9B params total (SigLip2 + Whisper-medium + CosyVoice2 + Qwen3-8B backbone)
- Frozen base weights
- Trainable adapters:
  - LoRA on Qwen3-8B backbone's attention layers (rank 16, alpha 32) — for action-head
    behavior adaptation
  - **New cross-attention layers** inserted at depths 12 and 24 of the Qwen3-8B backbone,
    attending over the thought-vector ring buffer (these are not LoRA — they're added
    layers initialized to identity-via-zero-init output projection so the model behaves
    identically to the base at init)
- Input modalities used: video frames at 15Hz (Atari 84x84 RGB upsampled to 224x224)
- Output: action token (one of 18 ALE actions), emitted every 67ms

**Why cross-attn rather than concat into context:** the thought buffer has K=16 entries.
Concatenating into the context would push the buffer into the same attention space as
recent input tokens, contaminating the attention budget. Dedicated cross-attn layers
keep the bridge readable without budget pressure on the main attention.

## Slow model: Qwen3-VL-8B-Thinking

- ~8B params, dense
- Native vision encoder (SigLIP-class) — accepts 1Hz frame snapshots directly
- bf16, frozen base
- Trainable adapters:
  - LoRA on attention projection layers (rank 16) — for adapting thought emission to
    game-context patterns
  - **Projection head**: linear from residual stream (4096-dim at layer 24) to thought
    vector (256-dim)
  - **Cross-attention LoRA** at layer 20: reads the perception buffer (Stage C3 only)
- Input: 1Hz frame snapshot + rolling 4-second window of (a) compact text game state
  (RAM-derived entity positions, score, lives, level) and (b) perception summaries from
  fast model
- Output: text reasoning tokens (with `<think>` block) + projected thought vectors at ~1-2Hz

**Why same-architecture-family on both sides:** MiniCPM-o 4.5's backbone is Qwen3-8B and
Qwen3-VL-8B-Thinking is built on the same family. This means the hidden dims (4096) are
identical on both ends of the bridge, so the projection head is a straightforward 4096→256
rather than a cross-family adapter. It also makes the perception-buffer cross-attention
on the slow side architecturally symmetric with the thought-buffer cross-attention on the
fast side.

### Scaling ablation (single configuration, not the main experiment)

For one Tier-3 game (Frostbite), we additionally run T and L with Qwen3-30B-A3B-Thinking
as the slow model. This 30B-A3B MoE has 3B active params per token and a 5120-dim residual
stream. It uses a slightly different projection head (5120→256 with a 512-dim hidden
layer). Purpose: test whether the latent-vs-text gap grows with slow-model capability,
evidencing the bandwidth claim. See `docs/03_experiment_plan.md` for protocol.

**Why projection rather than direct residual stream:** the residual stream is too
high-dimensional and unstructured for the fast model to attend over efficiently. A
learned projection compresses to 256-dim while preserving task-relevant information,
trained via the COCONUT-style curriculum (reconstruction against text-channel
predictions).

## The bridge

### Thought buffer (slow → fast)
- Ring of K=16 entries
- Each entry: 256-dim vector + age stamp (in fast ticks)
- Slow model writes one entry per emission (1-2 Hz)
- Fast model's new cross-attn layers attend over all 16 on every tick (15Hz)
- Bandwidth: 2 entries/s × 256 dim × bf16 = ~8 Kbps continuous

### Perception buffer (fast → slow)
- Ring of K=8 entries
- Each entry: 256-dim vector summarizing recent game state from fast model's perspective
- Fast model writes one entry per ~500ms (every ~7 fast ticks)
- Slow model's LoRA cross-attn reads on each forward pass

### Async coupling
- Both models run on independent threads / CUDA streams
- Buffers are append-only with positional encoding for age
- Fast model emits actions at fixed tick rate regardless of buffer freshness
- If slow model's emission rate drops, fast model attends over staler buffer entries
  with appropriate downweighting via age encoding

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
| MiniCPM-o frozen + LoRA + cross-attn | 21GB |
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
- `configs/fast_model.yaml` — LoRA ranks, cross-attn insertion depths, learning rates
- `configs/slow_model.yaml` — LoRA on output proj, projection-head dim
- `configs/bridge.yaml` — buffer sizes, emission rates, age-encoding scheme
- `configs/training.yaml` — per-stage hyperparameters, optimizer, schedule
- `configs/eval.yaml` — game list, seed counts, metric definitions
