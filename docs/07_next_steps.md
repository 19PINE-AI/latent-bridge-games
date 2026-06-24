# Plan: Demo Deliverable + Remaining Experiments

> Last updated: 2026-05-17. This document defines the work to convert the system from
> "we have positive headline numbers" to "anyone can watch this work, replay it, and
> interact with it via a website."

## Framing correction

Earlier sessions treated this as a paper-ablation project. The actual deliverable is a
demo — recorded videos AND live playback — that *shows* the fast/slow agent thinking
together, with the supporting experiments grounding the demo's claims. The README and
status docs now reflect this.

## Priority order

The order below balances paper value, user-stated priorities, and dependency structure
(items earlier in the list unblock items later).

1. **Demo infrastructure (highest priority).** Add per-tick frame + slow-emission
   logging to `src/eval/benchmark.py`. Build an MP4 renderer that overlays slow-model
   text as subtitles on the gameplay video. Build a live web playback server (Flask +
   WebSocket) for the website. Without this, none of the other experiments are
   visible.

2. **Stage A robustness ablation.** Co-train Stage A on a mix of bare and
   suffix-augmented prompts. Direct validation of the SpaceInvaders diagnosis: if T/L
   recover on SI without any other change, the Stage A OOD-brittleness story is
   confirmed.

3. **Slow-only baseline (S) and Oracle (O).** Two cells from the original four-strategy
   plan that never ran. S = slow model emits actions at its own rate (1-2 Hz) — shows
   "just use the big model" is too slow. O = pre-computed slow analyses injected as
   text — bounds the maximum any fast/slow coupling can buy.

4. **Tier-1 game (Pong or Breakout).** The original H2 prediction was L ≈ T on Tier 1.
   We have data on three Tier-2/3 games but zero on Tier 1, so we can't even check the
   *direction* of the strategic-complexity claim. Need a new RAM decoder + prompt
   template.

5. **Slow-model capacity ablation with `Qwen/Qwen3-VL-30B-A3B-Thinking-FP8`.** Tests
   whether more slow-model capacity widens L − T. The behavioral predictor says the
   bridge helps iff slow reasoning helps (T > F), with L − F tracking T − F at r = 0.93;
   a larger slow model that lifts T − F should, on this account, also lift L − T.
   Confirming that link strengthens the predictor story. (This is a capacity question,
   not a test of the retired "bandwidth"/latent-channel-width thesis.)

6. **Stage D PPO.** System-completion item AND highest-information-gain experiment for
   SpaceInvaders (will it recover from L = 0 under PPO?). Biggest effort, biggest
   upside. Best done last because the demo + S/O baselines + scaling ablation should
   be locked in first.

## Execution plan (parallelized)

### Phase 1 — Setup (sequential, ~1.5h)
- **1A.** Frame + slow-emission logging in benchmark.py (`--save-trace-dir` flag).
- **1B.** MP4 render script: frames + matplotlib/PIL overlay of slow text as caption.
- **1C.** Live playback Flask + WebSocket server (`scripts/live_demo_server.py`).
- **1D.** Start `Qwen/Qwen3-VL-30B-A3B-Thinking-FP8` download in background (~30 GB).

### Phase 2 — Generate demos + run low-cost experiments while download runs (~3h)
- **2A.** Generate MP4 demos for all 3 games × {F, L} = 6 videos using existing
  checkpoints. Side-by-side comparison HTML stub.
- **2B.** Slow-only S baseline on MsPacman (~30 min).
- **2C.** Oracle O baseline on MsPacman (~30 min).
- **2D.** Stage A mixed-prompt re-training on SpaceInvaders (~1h). Then re-eval T/L.

### Phase 3 — Tier 1 + capacity ablation (~4h, after Phase 2 + download done)
- **3A.** Pong RAM decoder + prompt template. F + T eval (no Stage A needed if F
  scores baseline immediately; if F = 0 we add Stage A).
- **3B.** Slow-model capacity ablation: T-collection with 30B slow, Stage C v2
  retraining, F/T/L eval on MsPacman. Check whether the larger slow model widens both
  T − F and L − T.

### Phase 4 — Stage D PPO (~12-24h, after Phase 3)
- **4A.** Implement PPO driver on top of existing `stage_d_rl.py` scaffold.
- **4B.** Smoke test (2 episodes, verify gradients).
- **4C.** SI PPO run (~6h) — the diagnostic test.
- **4D.** MsPacman PPO run (~6h) — push the headline number.
- **4E.** Seaquest PPO run (~6h) — break the deterministic exploit.

### Phase 5 — Closing (sequential)
- Aggregate, refresh `06_results.md`, README, build a website bundle with the demos,
  result tables, and interactive viewer. Tag a release.

## Time budget

| Phase | Wall-clock |
|---|---|
| 1 | 1.5 h |
| 2 | 3 h (parallel with download) |
| 3 | 4 h |
| 4 | 12-24 h |
| 5 | 2 h |
| **Total** | **~24-36 h** wall, mostly GPU-bound |

## Risk register

| Risk | Mitigation |
|---|---|
| Live playback complexity (Flask + frames + slow streaming) | Start with MP4 only; live mode is a stretch if Phase 4 takes longer than budget |
| 30B-FP8 hidden_size differs from 8B — need projection retrain | Make ThoughtProjection input-dim-aware; one-line config change |
| 30B-FP8 MoE residual extraction returns unexpected shape | Smoke test before launching full pipeline |
| PPO instability collapses the policy | KL anchor against Stage C, fall-back checkpoint |
| Pong has no SB3-zoo expert | F-mode with bare prompt may saturate; if not, scratch-train a DQN (~hours) |
| Stage A robustness retrain hurts MsPacman/Seaquest scores | Train a separate `_robust.pt` checkpoint; keep originals |

## Files of record (to be created)

- `scripts/render_demo_mp4.py` — frames + overlay → MP4
- `scripts/live_demo_server.py` — Flask + WebSocket live playback
- `scripts/stage_a_mixed_prompt.sh` — robustness ablation pipeline
- `scripts/scaling_ablation_30b.sh` — 30B slow-model capacity ablation (does more
  slow-model capacity widen L − T?)
- `scripts/ppo_pipeline.sh` — Stage D PPO orchestration
- `src/env/atari_wrapper.py` — add Pong/Breakout decoders
- `src/training/prompts.py` — add Pong/Breakout templates
- `src/training/stage_d_rl.py` — extend scaffold with rollout + update loop
- `web-react/` (Phase 5) — interactive demo site (built to `web-dist/`)
