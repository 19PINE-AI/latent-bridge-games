# Related Work

Surveyed across dual-system architectures (robotics + non-physical), latent reasoning
within single models, real-time interaction architectures, and game-playing agents.
This is the working bibliography — sources are documented inline rather than in a
separate `.bib` until paper-writing time.

## Dual-system architectures in robotics

These projects converged on continuous-vector latent bridges between fast motor policies
and slow reasoning models. Our work transfers the architectural pattern to symbolic /
visual (game) domains where the slow model is a frozen text-reasoning LLM.

- **Helix (Figure AI, 2025-2026):** S2 onboard VLM at 7-9 Hz emits latent semantic
  representations that S1 visuomotor policy consumes at 200 Hz to produce joint targets.
  Helix-02 adds S0 at 1 kHz for whole-body control. Three timescales with continuous
  latent interfaces. Reference architecture for the timescale hierarchy.
  [Figure AI blog](https://www.figure.ai/news/helix), [Helix-02](https://www.figure.ai/news/helix-02)

- **π0 / π0.5 (Physical Intelligence, 2024-2025):** Slow VLM emits FAST action tokens
  (VQ-tokenizer learned) consumed by S1 flow-matching policy. Step toward latent
  communication; tokens still discrete but action-relevant.
  [π0 paper (2410.24164)](https://arxiv.org/abs/2410.24164),
  [π0 blog](https://www.pi.website/blog/pi0)

- **GR00T N1 (NVIDIA, 2025):** Open-source humanoid foundation model with the same
  dual-system pattern. Slow VLM-based System 2 produces conditioning for fast
  diffusion-based System 1.
  [NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks)

- **Fast-in-Slow / FiS-VLA (Wu et al., 2025):** The architectural insight most relevant to
  us. S1 is **embedded within S2** by repurposing the final transformer blocks of the
  same VLM backbone. Co-trained with dual loss (next-token prediction for S2, diffusion
  for S1). Shared representations beat separate models. We cannot use this approach
  directly because our slow model is frozen, but it sets the upper bound we measure
  against.
  [FiS-VLA (2506.01953)](https://arxiv.org/html/2506.01953v1)

## Latent reasoning within single models

- **COCONUT (Hao et al., Meta, 2024-2025):** Trains a single LLM to feed its last hidden
  state back as the next input embedding instead of decoding to text. Uses a multi-stage
  curriculum: at stage k, replace k language reasoning steps with c·k latent steps.
  Continuous thoughts can encode multiple alternative reasoning paths (BFS-like). 97% vs
  77.5% on ProsQA against text CoT. **Our training curriculum is directly inspired by
  COCONUT** but applied cross-model rather than within a single model.
  [COCONUT (2412.06769)](https://arxiv.org/abs/2412.06769),
  [GitHub](https://github.com/facebookresearch/coconut)

- **Token Assorted (Su et al., 2025):** Mixes VQ-VAE-encoded latent tokens with text
  tokens in a single reasoning trace. Compromise between full-latent and full-text. We
  evaluate this as one of the bridge variants.
  [Token Assorted (2502.03275)](https://arxiv.org/html/2502.03275)

- **Asynchronous Reasoning (Tian et al., 2025):** Training-free method that rearranges
  the KV cache to make multiple async reasoning streams appear as a single sequence.
  Important precedent for our async coupling design.
  [Async Reasoning (2512.10931)](https://arxiv.org/html/2512.10931v1)

## Real-time interaction architectures

- **Thinking Machines Lab Interaction Models (Mira Murati et al., May 2026):** The
  direct comparison target. Trained-from-scratch full-duplex multimodal model with 200ms
  micro-turns. Pairs with a separate asynchronous background reasoning model that
  communicates via text. 0.40s turn-taking latency vs 1.18s for GPT-Realtime-2.0 and
  0.57s for Gemini Live. Our claim: the text-channel handoff between their interaction
  model and background reasoner leaves headroom that a latent bridge captures.
  [TML blog](https://thinkingmachines.ai/blog/interaction-models/),
  [Sean Goedecke analysis](https://www.seangoedecke.com/interaction-models/)

- **AsyncVoice Agent (2025):** Real-time explanation streaming for LLM planning. Closest
  prior art for "slow model thoughts surfaced to user in real time" but uses text channel
  and targets user-facing explanation rather than agent control.
  [AsyncVoice (2510.16156)](https://arxiv.org/html/2510.16156v1)

- **LTS-VoiceAgent (2026):** Listen-Think-Speak framework with semantic triggering and
  incremental reasoning. Designed for voice but architecturally similar pattern. We
  reuse the conceptual frame.
  [LTS-VoiceAgent (2601.19952)](https://arxiv.org/html/2601.19952)

- **MiniCPM-o 4.5 (OpenBMB, 2025):** Our fast model. 9B total params built on SigLip2 +
  Whisper-medium + CosyVoice2 + Qwen3-8B. Full-duplex multimodal streaming with TDM
  scheme. Sub-12GB inference cost. Already handles video at 10fps.
  [HF](https://huggingface.co/openbmb/MiniCPM-o-4_5),
  [GitHub](https://github.com/OpenBMB/MiniCPM-o),
  [arxiv](https://arxiv.org/html/2604.27393)

## Game-playing agents (baselines and comparison context)

- **Atari DQN family (DeepMind, 2013-2017):** The foundational benchmark. Original DQN at
  ~human level on most Atari games via pure pixel-to-action policy learning. Our fast-only
  baseline targets the regime DQN-class agents operate in.

- **MuZero (DeepMind, 2019):** Model-based RL with learned dynamics; very strong on Atari
  with significant compute. Sets the upper bound for fast-only-with-planning agents.

- **Agent57 (Badia et al., 2020):** First to exceed human baseline on all 57 Atari games.

- **Decision Transformer / Trajectory Transformer (Chen, Janner et al., 2021):** Offline
  RL via sequence modeling. Architecturally similar to what our fast model would learn if
  it were the only model.

- **GATO (DeepMind, 2022) and MultiGame DT (Lee et al., 2022):** Generalist game-playing
  via single transformer. Closer in spirit to our fast model, but no slow-thinking
  augmentation.

- **VPT / Voyager:** Web-scale pretraining for game playing (VPT on Minecraft) and
  open-ended agents (Voyager). Not direct comparisons but motivate the "fast model
  pretrained at scale" assumption.

## Where this work fits

Our positioning, by what each prior work does NOT do:

- **Helix / π0 / GR00T:** dual-system for physical robotics, not symbolic / game domains.
- **FiS-VLA:** dual-system in shared backbone, requires retraining the slow model — we
  cannot retrain a state-of-the-art reasoning model.
- **COCONUT / Token Assorted:** latent reasoning within a single model; we apply
  cross-model.
- **TML interaction models:** dual-model split for voice/video, communication via text
  channel; we propose latent bridge as the channel.
- **Atari DQN family:** fast-only agents trained end-to-end; no augmentation with
  external reasoning model.
- **Agentic LLM game playing (Cradle, GameAgent):** uses the slow LLM directly for
  action selection — fails on real-time games because slow LLM latency exceeds tick rate.
  We use the slow LLM as a deliberative augmentation, not a primary controller.

## The gap

No prior work demonstrates cross-model latent bridges (with both base models frozen) for
real-time game playing or other symbolic/visual non-robotic tasks. This is the gap we
target empirically.
