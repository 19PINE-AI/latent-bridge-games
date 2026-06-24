# `scripts/` guide

This directory holds the reproducible pipelines, reusable tools, and a number of
one-off orchestration scripts kept for provenance. Start with the **canonical
entry points**; everything else is supporting or historical.

Run scripts from the repo root. Most expect the two frozen base models to be
reachable via `LB_FAST_MODEL_PATH` / `LB_SLOW_MODEL_PATH` or the HF cache (see
the top-level `README.md` → Setup).

## Canonical entry points (start here)

| Script | Purpose |
|---|---|
| `validate_hardware.py` | Check both models fit on the GPU and measure tick latency. **Run first.** |
| `<game>_pipeline.sh` | Full per-game A→B→C→eval chain. Games: `mspacman` (see below), `spaceinvaders`, `pong`, `qbert`, `riverraid`, `roadrunner`, `enduro`. |
| `aggregate_results.py` | Roll up all `results/eval_*.json` into one comparison table. |
| `make_figures.py` | Paper-quality matplotlib figures from the eval JSONs. |
| `record_ftl_demos.sh` | Record F/T/L traces for every game and render the demo videos. |
| `build_combined_demo.py` | Build the single narrated multi-game demo MP4 (ElevenLabs, gTTS fallback). |
| `live_demo_server.py` | Flask + SSE server that replays traces and serves the built React site. |

> The MsPacman chain is documented step-by-step in the top-level `README.md`
> ("Reproducing the headline number"). The other `*_pipeline.sh` scripts run the
> same A→B→C→eval chain end-to-end for their game.

## Core tools

**Trajectory collection** — `collect_trajectories.py` (Atari, random or SB3
expert), `collect_metadrive_expert.py`, `collect_minigrid_expert.py` (A* expert),
`collect_highway_expert.py`.

**Conditions / baselines** — `run_text_bridge_baseline.py` (T; also emits Stage C
supervision), `run_slow_only_baseline.py` (S), `run_oracle_baseline.py` /
`run_true_oracle.py` (O), `random_baseline.py` (floor).

**Bridge training / analysis** — `dagger_bridge.py` (on-policy correction),
`nearest_neighbor_decode.py` / `nearest_neighbor_diagnostics.py` (latent-token
interpretability), `stage_a_ood_kl_probe.py` (OOD-brittleness probe).

**Decoder selection / tables** — `decoder_select_cv.py` (held-out per-channel
decoder selection), `make_best_achievable_table.py`.

**Website / demos** — `extract_website_data.py` (→ `web-react/src/data/research.json`),
`render_demo_mp4.py`, `build_combined_srt.py`, `show_text_state.py`,
`stage_model_local.py` (stage an HF snapshot into a standalone dir).

## Cross-domain extensions

- **MetaDrive** (driving; the controlled negative): `run_metadrive_topdown.sh` is
  the working path (top-down renderer, no GL/Xvfb). `run_metadrive_{egl,xvfb,pipeline}.sh`
  are alternate render backends; `run_metadrive_{planning,v2expert,fixteacher,diagnostics}.sh`
  and `record_metadrive_demo.sh` cover the planning map, teacher fixes, and controls.
  `train_metadrive_ppo.py`, `eval_metadrive_straight.py` are the expert + baseline.
- **MiniGrid**: `run_minigrid_autonomous.sh`, `eval_minigrid_shaped.py`, `run_shaped_eval.sh`.
- **Highway**: `run_highway_pipeline.sh`.

## Ablations / robustness studies

`run_bridge_replace_sweep.sh` (trained vs zero vs random-norm latent — the
bridge-replacement control), `true_bandwidth_ablation.sh` (N=4/8/16 latent
tokens), `scaling_ablation_30b.sh` (8B → 30B-A3B slow model), `latency_sweep_v2.sh`
(vision-refresh trade-off), `stage_a_robustness_si.sh` / `robust_stage_a_*.sh`
(suffix-prob robust Stage A), `rev_*.sh` (greedy-vs-sampling decoder sweeps behind
the "greedy-artifact" finding), `ppo_si_*.sh` (Stage D PPO retries).

## One-off / historical (provenance only — not general entry points)

These were session-specific overnight chains that poll for `/tmp/*.out`
completion markers from a particular run; they will **not** run on a clean
checkout and are kept only to document how the experiments were sequenced:
`overnight_pipeline.sh`, `phase2_and_scaling.sh`, `master_top3_chain.sh`,
`gap_closers_chain.sh`, all `follow_on_*.sh`, `qbert_resume.sh`,
`continue_minigrid.sh` / `mg_finish_clean.sh` (MiniGrid resumes), and
`demo_pipeline.sh` / `demos_new_games.sh` (superseded by `record_ftl_demos.sh`).
