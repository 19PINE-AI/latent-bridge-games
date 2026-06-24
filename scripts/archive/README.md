# `scripts/archive/` — provenance-only orchestration

These scripts are kept to document how the experiments were actually sequenced.
They are **not** reproducible entry points: most poll for `/tmp/*.out` completion
markers from a particular overnight run and will block forever on a clean
checkout. For reproduction, use the canonical entry points in [`../README.md`](../README.md).

Contents:
- **Overnight chains** — `overnight_pipeline.sh`, `phase2_and_scaling.sh`,
  `master_top3_chain.sh`, `gap_closers_chain.sh`
- **Wait-then-launch follow-ons** — `follow_on_*.sh` (pong+ppo, phase8, oracle
  retry, enduro+qbert, rr+roadrunner, and the 30B-scaling 30b/awq/awq_retry2/bf16
  variants)
- **Resume scripts** — `qbert_resume.sh`, `continue_minigrid.sh`, `mg_finish_clean.sh`
- **Superseded demo builders** — `demo_pipeline.sh`, `demos_new_games.sh`
  (replaced by [`../record_ftl_demos.sh`](../record_ftl_demos.sh))
