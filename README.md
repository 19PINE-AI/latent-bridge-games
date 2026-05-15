# Latent Bridge: Fast-Slow Model Coupling for Real-Time Agents

A research project investigating whether continuous-valued latent bridges between a frozen
real-time multimodal model (MiniCPM-o 4.5, 9B) and a frozen reasoning model
(Qwen3-30B-A3B-Thinking) can outperform text-channel splits for tasks requiring both
sub-200ms reactive output and long-horizon deliberation.

**Centerpiece scenario:** Atari-class video games requiring fast reflexes AND strategic
planning (Ms. Pac-Man, Frostbite). Secondary: live video-stream narration.

## The architectural claim

Current real-time AI systems either (a) use a single small model that lacks reasoning depth,
or (b) split into a fast interaction model + slow reasoning model that communicate via text
prompts (Thinking Machines Lab pattern). We argue text channels are bandwidth-limited and
demonstrate that a learned continuous-valued bridge — trained via COCONUT-style staged
curriculum, with both base models frozen — recovers most of the slow-model-offline capability
at real-time tick rates.

## Hypotheses

- **H1:** MiniCPM-o 4.5 + Qwen3-30B-A3B-Thinking via latent bridge produces game scores
  that strictly dominate fast-only and text-bridge architectures on games requiring both
  reactive and strategic components.
- **H2:** A phase-transition exists in game complexity — text bridges suffice for simple
  games but fail at higher strategic load; latent bridges hold up.
- **H3:** COCONUT-style curriculum training with the slow model frozen closes most of the
  gap to a unified jointly-trained upper bound.

## Repo layout

```
latent-bridge-games/
├── README.md
├── docs/
│   ├── 01_framing.md             # research thesis + scope
│   ├── 02_related_work.md         # surveyed prior art
│   ├── 03_experiment_plan.md      # 4-week timeline
│   └── 04_architecture.md         # bridge spec + memory budget
├── src/
│   ├── env/                       # game environment wrappers
│   ├── models/                    # fast/slow model adapters
│   ├── bridge/                    # ring-buffer + cross-attention
│   ├── training/                  # stage curricula
│   ├── eval/                      # benchmark harness
│   └── utils/
├── configs/                       # YAML run configs
├── scripts/                       # one-shot validation + run scripts
├── tests/                         # unit tests for components
├── results/                       # per-condition raw outputs
└── checkpoints/                   # LoRA + bridge weights
```

## Status

- [x] Repo scaffold
- [x] Planning docs
- [ ] Week 1: Joint inference validation
- [ ] Week 2: Bridge curriculum (Stages A/B)
- [ ] Week 3: Joint LoRA + ablations
- [ ] Week 4: Eval + paper

## Hardware

NVIDIA RTX Pro 6000, 96GB VRAM. Joint inference of MiniCPM-o 4.5 (bf16, 18GB) and
Qwen3-30B-A3B-Thinking (bf16, 60GB) fits with ~18GB headroom for training overhead.
