export interface StrategyRow {
  id: "S" | "F" | "T" | "L";
  name: string;
  score: number;
  latencyDesc: string;
  comment: string;
}

// MsPacman four-strategy comparison
export const STRATEGIES: StrategyRow[] = [
  {
    id: "S",
    name: "S — Slow only (~1 Hz)",
    score: 113,
    latencyDesc: "~4 s per decision (≈ 0.25 Hz effective)",
    comment: "Just use the big model: too slow. Confirms the project premise.",
  },
  {
    id: "F",
    name: "F — Fast only (15 Hz)",
    score: 256,
    latencyDesc: "33 ms / decision (warm path)",
    comment: "Reactive policy, no strategic context.",
  },
  {
    id: "T",
    name: "T — Text bridge",
    score: 408,
    latencyDesc: "~40 ms fast tick + 1.5 s slow at 1 Hz",
    comment: "Slow guides fast via a ~300-char text suffix (median 302).",
  },
  {
    id: "L",
    name: "L — Latent bridge",
    score: 628,
    latencyDesc: "~38 ms fast tick + 1.5 s slow at 1 Hz",
    comment: "Slow guides fast via 8 latent tokens (33 M trainable params); +5 ms over F on the warm path.",
  },
];

// Latent token-count (N) ablation (MsPacman, 12 episodes, matched train=deploy)
export interface BandwidthRow { N: number; L: number; comment: string; }
export const BANDWIDTH: BandwidthRow[] = [
  { N: 4, L: 296, comment: "Above F=256 but below T=408" },
  { N: 8, L: 628, comment: "Best at matched N" },
  { N: 16, L: 259, comment: "Below F at matched N (deploy-only N=16 instead gives 720 — no capacity ceiling)" },
];

// Vision-cache latency sweep (F MsPacman)
export interface LatencyRow { vrf: number; latency: number; speedup: string; score: number; }
export const LATENCY: LatencyRow[] = [
  { vrf: 1, latency: 33, speedup: "—",   score: 180 },
  { vrf: 4, latency: 20, speedup: "−39 %", score: 110 },
  { vrf: 15, latency: 17, speedup: "−48 %", score: 140 },
];
