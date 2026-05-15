# Project status

## Completed (initial scaffold)

- [x] Repo structure (`src/`, `configs/`, `docs/`, `scripts/`, `tests/`)
- [x] Planning docs (framing, related work, experiment plan, architecture)
- [x] Code skeleton:
  - [x] `src/env/atari_wrapper.py` — ALE env wrapper with 15Hz fast-tick + 1Hz text-state hook
  - [x] `src/models/fast_model.py` — MiniCPM-o wrapper + BridgeCrossAttention layer (zero-init)
  - [x] `src/models/slow_model.py` — Qwen3-30B-A3B wrapper + ThoughtProjection head
  - [x] `src/bridge/ring_buffer.py` — ThoughtBuffer + PerceptionBuffer + age encoding
  - [x] `src/training/stage_{a,c,d}_*.py` — stage entry points (NotImplementedError stubs)
  - [x] `src/eval/benchmark.py` — eval harness stub
- [x] Configs: `bridge.yaml`, `fast_model.yaml`, `slow_model.yaml`, `training.yaml`, `eval.yaml`
- [x] Tests: `tests/test_ring_buffer.py` (CUDA-required smoke tests)
- [x] Scripts: `scripts/validate_hardware.py` (Week-1 joint-fit validation)

## Next concrete steps (Week 1)

1. Run `python scripts/validate_hardware.py` to confirm both models fit in 96GB.
2. Wire up `FastModel.forward()` — needs to hook MiniCPM-o's backbone forward
   and inject cross-attention at layers 12 and 24.
3. Wire up `SlowModel.emit()` — generate text + extract residual stream at the
   projection layer + apply projection head.
4. Wire up `AtariEnv._text_state()` — read RAM via `env.unwrapped.ale.getRAM()`
   for Ms. Pac-Man (sprite positions are at well-known RAM addresses).
5. Implement Stage A behavioral cloning loop.

## Open design questions

- **Slow-model emission policy:** continuous emission at 1Hz, or event-triggered
  (only when game state changes significantly)? Continuous is simpler; event-triggered
  may be more compute-efficient. Default: continuous for the paper, note event-triggered
  as future work.
- **Action discretisation:** MiniCPM-o emits text tokens; we need to map to 18 ALE
  actions. Two approaches:
  - Add a small classification head on top of the last hidden state (cleaner)
  - Constrain decoding to the 18 action-name tokens (preserves MiniCPM-o behavior)
  Default: classification head, since we're already adding cross-attn layers.
- **PPO vs imitation-only for Stage D:** PPO is the standard choice but is unstable
  and may be hard to make work in 1 week with a 9B model. Fallback: skip RL, rely on
  imitation + Stage C supervision only. Slightly weaker paper but tractable.

## Risk log

(Open at project start. Update as risks materialise.)

| Date | Risk | Status |
|---|---|---|
| 2026-05-15 | OOM on joint inference | Untested |
| 2026-05-15 | MiniCPM-o cross-attn injection is harder than expected | Untested |
| 2026-05-15 | F-only saturates on chosen games | Untested |
