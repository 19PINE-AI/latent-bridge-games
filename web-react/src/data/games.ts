// Per-game results with 95% bootstrap CIs (n=10000 resamples) and
// Welch's t / Mann-Whitney U p-values, generated from the raw per-episode
// JSON dumps. Numbers match paper Table 1.
export interface GameResult {
  id: string;
  name: string;
  shortName?: string;
  tier: 1 | 2 | 3;
  variant?: "bare" | "robust";
  F: number; F_std: number; F_ci?: [number, number];
  T: number; T_std: number; T_ci?: [number, number];
  L: number; L_std: number; L_ci?: [number, number];
  L_vs_T_pct: number | null;
  /** Welch's t two-sided p-value; falls back to MWU when one cell has zero variance. */
  pvalue?: number;
  /** True when zero-variance forced fallback to MWU. */
  pvalueIsMWU?: boolean;
  /** Cohen's d for L − T (positive: L > T). */
  cohensD?: number;
  notes: string;
  category: "win-L" | "win-T" | "collapse" | "floor" | "partial";
  /** True if abs(score) is small relative to the F floor — treat result with caution. */
  smallScores?: boolean;
  bridgeMI?: { action: number; reward: number };
  videoF?: string;
  videoT?: string;
  videoL?: string;
  videoSideBySide?: string;
  videoFTL?: string;
  isHeadline?: boolean;
}

export const GAMES: GameResult[] = [
  {
    id: "roadrunner-bare",
    name: "Road Runner",
    tier: 3,
    variant: "bare",
    F: 0, F_std: 0, F_ci: [0, 0],
    T: 608, T_std: 250, T_ci: [467, 733],
    L: 967, L_std: 49, L_ci: [942, 992],
    L_vs_T_pct: 59,
    pvalue: 0.0004,
    cohensD: 1.99,
    notes: "Bare F sits at NOOP and cannot escape the Coyote. Slow's directional commitment unlocks the rightward escape; L preserves the joint (Coyote, pellet, obstacles) where T can only enumerate.",
    category: "win-L",
    isHeadline: true,
    videoF: "demos/roadrunner_F.mp4",
    videoT: "demos/roadrunner_T.mp4",
    videoL: "demos/roadrunner_L.mp4",
    videoFTL: "demos/roadrunner_F_T_L.mp4",
    videoSideBySide: "demos/roadrunner_F_vs_L.mp4",
  },
  {
    id: "mspacman",
    name: "Ms. Pac-Man",
    shortName: "MsPacman",
    tier: 2,
    variant: "bare",
    F: 256, F_std: 25, F_ci: [242, 269],
    T: 408, T_std: 92, T_ci: [359, 458],
    L: 628, L_std: 356, L_ci: [501, 842],
    L_vs_T_pct: 54,
    pvalue: 0.06,
    cohensD: 0.85,
    notes: "Headline result. L distribution is bimodal: 8/12 episodes in [390, 690], 4/12 above 1000 (best 1740). The tail is the slow's spatial guidance chaining 3+ pellet-cluster transitions; even the body beats T's entire 95% CI.",
    category: "win-L",
    isHeadline: true,
    bridgeMI: { action: 0.002, reward: 0.010 },
    videoF: "demos/mspacman_F.mp4",
    videoT: "demos/mspacman_T.mp4",
    videoL: "demos/mspacman_L.mp4",
    videoFTL: "demos/mspacman_F_T_L.mp4",
    videoSideBySide: "demos/mspacman_F_vs_L.mp4",
  },
  {
    id: "riverraid-robust",
    name: "River Raid (robust SA)",
    shortName: "RiverRaid",
    tier: 3,
    variant: "robust",
    F: 1033, F_std: 20, F_ci: [1023, 1045],
    T: 337, T_std: 81, T_ci: [297, 383],
    L: 612, L_std: 310, L_ci: [450, 781],
    L_vs_T_pct: 82,
    pvalue: 0.011,
    cohensD: 1.21,
    notes: "Largest L−T gap in the sweep. Bare Stage A collapsed both bridges; mixed-prompt training shifted T slightly down (383→337) and L dramatically up (360→612). Same fix recipe as Q*bert; opposite direction.",
    category: "win-L",
    isHeadline: true,
    videoF: "demos/riverraid_F.mp4",
    videoT: "demos/riverraid_T.mp4",
    videoL: "demos/riverraid_L.mp4",
    videoFTL: "demos/riverraid_F_T_L.mp4",
    videoSideBySide: "demos/riverraid_F_vs_L.mp4",
  },
  {
    id: "seaquest",
    name: "Seaquest",
    tier: 3,
    variant: "bare",
    F: 42, F_std: 20, F_ci: [32, 53],
    T: 63, T_std: 12, T_ci: [57, 70],
    L: 80, L_std: 0, L_ci: [80, 80],
    L_vs_T_pct: 26,
    pvalue: 0.0004,
    cohensD: 2.04,
    notes: "Deterministic 8-kill exploit (L std=0) — L locks into a stable surfacing+kill pattern. Effect size large despite small absolute scores.",
    category: "win-L",
    videoF: "demos/seaquest_F.mp4",
    videoT: "demos/seaquest_T.mp4",
    videoL: "demos/seaquest_L.mp4",
    videoFTL: "demos/seaquest_F_T_L.mp4",
    videoSideBySide: "demos/seaquest_F_vs_L.mp4",
  },
  {
    id: "enduro-robust",
    name: "Enduro (robust SA)",
    tier: 2,
    variant: "robust",
    F: 0.8, F_std: 1.1, F_ci: [0.2, 1.3],
    T: 4.9, T_std: 5.9, T_ci: [2.0, 8.4],
    L: 5.8, L_std: 2.6, L_ci: [4.3, 7.3],
    L_vs_T_pct: 19,
    pvalue: 0.63,
    cohensD: 0.20,
    smallScores: true,
    notes: "L numerically ahead but absolute scores too small to support a confident claim (p=0.63, Cohen's d=0.20). Reported for completeness; we do not count it as a positive case.",
    category: "partial",
    videoF: "demos/enduro_F.mp4",
    videoT: "demos/enduro_T.mp4",
    videoL: "demos/enduro_L.mp4",
    videoFTL: "demos/enduro_F_T_L.mp4",
    videoSideBySide: "demos/enduro_F_vs_L.mp4",
  },
  {
    id: "qbert-robust",
    name: "Q*bert (robust SA)",
    shortName: "Qbert",
    tier: 2,
    variant: "robust",
    F: 25, F_std: 0, F_ci: [25, 25],
    T: 125, T_std: 0, T_ci: [125, 125],
    L: 50, L_std: 0, L_ci: [50, 50],
    L_vs_T_pct: -60,
    pvalue: 1e-7, // MWU on fully separated distributions
    pvalueIsMWU: true,
    notes: "Greedy: text beats latent 2.5× (T=125, L=50) — but every cell is zero-variance, the deterministic regime. Under multinomial sampling (τ=1.0) this INVERTS to a 2.9× latent win (L=250, T=88, p≈0.006): the greedy 'text win' was a decoder artifact, not a property of the (categorical) emission. The one inversion does not survive a decoder ablation.",
    category: "win-T",
    videoF: "demos/qbert_F.mp4",
    videoT: "demos/qbert_T.mp4",
    videoL: "demos/qbert_L.mp4",
    videoFTL: "demos/qbert_F_T_L.mp4",
    videoSideBySide: "demos/qbert_F_vs_L.mp4",
  },
  {
    id: "spaceinvaders-robust",
    name: "Space Invaders (robust SA)",
    shortName: "SpaceInvaders",
    tier: 2,
    variant: "robust",
    F: 107, F_std: 62, F_ci: [83, 145],
    T: 18, T_std: 19, T_ci: [10, 30],
    L: 15, L_std: 0, L_ci: [15, 15],
    L_vs_T_pct: -17,
    pvalue: 0.27,
    pvalueIsMWU: true,
    notes: "Robust SA broke the zero floor, but F=107 still dominates either bridge. The fix recovers L/T from collapse, not to a useful policy. Honest read: SI remains a negative result for fast/slow coupling.",
    category: "partial",
    videoF: "demos/spaceinvaders_F.mp4",
    videoT: "demos/spaceinvaders_T.mp4",
    videoL: "demos/spaceinvaders_L.mp4",
    videoFTL: "demos/spaceinvaders_F_T_L.mp4",
    videoSideBySide: "demos/spaceinvaders_F_vs_L.mp4",
  },
  {
    id: "spaceinvaders-bare",
    name: "Space Invaders (bare SA)",
    tier: 2,
    variant: "bare",
    F: 105, F_std: 0, F_ci: [105, 105],
    T: 0, T_std: 0, T_ci: [0, 0],
    L: 0, L_std: 0, L_ci: [0, 0],
    L_vs_T_pct: null,
    notes: "Three knob-tuning attempts (random-T, expert-T, aggressive-prompt) all gave T=L=0 despite KL convergence (~0.005). Bridge had +0.024 nats action MI under expert-T — structure was learned, just not deployable. Diagnosis trigger: Stage A OOD-brittleness.",
    category: "collapse",
    bridgeMI: { action: 0.024, reward: 0.012 },
    videoF: "demos/spaceinvaders_F.mp4",
    videoT: "demos/spaceinvaders_T.mp4",
    videoL: "demos/spaceinvaders_L.mp4",
    videoFTL: "demos/spaceinvaders_F_T_L.mp4",
  },
  {
    id: "riverraid-bare",
    name: "River Raid (bare SA)",
    tier: 3,
    variant: "bare",
    F: 1067, F_std: 88, F_ci: [1020, 1116],
    T: 383, T_std: 60, T_ci: [350, 413],
    L: 360, L_std: 0, L_ci: [360, 360],
    L_vs_T_pct: -6,
    pvalue: 0.014,
    pvalueIsMWU: true,
    notes: "Both bridges collapsed below F=1067. Robust Stage A retraining fixed L (→612, +82% over T); see row above.",
    category: "collapse",
  },
  {
    id: "pong",
    name: "Pong",
    tier: 1,
    variant: "bare",
    F: -21, F_std: 0, F_ci: [-21, -21],
    T: -21, T_std: 0, T_ci: [-21, -21],
    L: -21, L_std: 0, L_ci: [-21, -21],
    L_vs_T_pct: 0,
    notes: "All three at the −21 loss floor. Stage A val-accuracy is 25.1% (random=16.7% on 6 actions, only 1.5× random vs MsPacman's 2.9×). The action head can barely move the paddle; no slow-model guidance can rescue a reactive policy that does not exist.",
    category: "floor",
  },
  {
    id: "metadrive",
    name: "MetaDrive",
    tier: 3,
    variant: "robust",
    F: 87.8, F_std: 26.3, F_ci: [70, 105],
    T: 85.1, T_std: 25.4, T_ci: [68, 102],
    L: 85.1, L_std: 25.4, L_ci: [68, 102],
    L_vs_T_pct: 0,
    notes: "Non-Atari driving domain (planning-heavy map). The controlled NEGATIVE: slow reasoning never beats fast-only (T ≤ F) even when the task requires route planning, so the latent is inert (bridge-replace: L ≈ L_zero ≈ L_random). This is the boundary case that defines the T>F predictor.",
    category: "collapse",
    videoF: "demos/metadrive_F.mp4",
    videoT: "demos/metadrive_T.mp4",
    videoL: "demos/metadrive_L.mp4",
    videoFTL: "demos/metadrive_F_T_L.mp4",
    videoSideBySide: "demos/metadrive_F_vs_L.mp4",
  },
];

export const HEADLINE = GAMES.filter(g => g.isHeadline);

// Bridge-replacement control across all 7 Atari games (canonical decoder per game).
// L with the trained latent vs zeroed vs random-at-matched-norm. "Learned" = trained
// meaningfully exceeds both controls. Numbers read from results/br_*.json.
export interface BridgeReplacePoint {
  game: string;
  trained: number; zero: number; random: number;
  tmf: number;       // T − F for this game (predictor x-axis)
  verdict: "learned" | "inert" | "harmful";
}
export const BRIDGE_REPLACE: BridgeReplacePoint[] = [
  { game: "RoadRunner",    trained: 967, zero: 0,    random: 8,    tmf: 608,  verdict: "learned" },
  { game: "MsPacman",      trained: 666, zero: 408,  random: 410,  tmf: 152,  verdict: "learned" },
  { game: "Seaquest",      trained: 100, zero: 28,   random: 5,    tmf: 22,   verdict: "learned" },
  { game: "Qbert",         trained: 123, zero: 63,   random: 117,  tmf: 100,  verdict: "inert" },
  { game: "Enduro",        trained: 8,   zero: 1,    random: 5,    tmf: -3,   verdict: "inert" },
  { game: "SpaceInvaders", trained: 0,   zero: 148,  random: 90,   tmf: -105, verdict: "harmful" },
  { game: "Riverraid",     trained: 360, zero: 1013, random: 1003, tmf: -683, verdict: "harmful" },
];

// Aggregate stats for landing-page numbers.
export const SUMMARY = {
  totalGames: 8,         // (Frostbite excluded — Stage A at random)
  evaluable: 7,          // (Pong reported as floor)
  L_wins_greedy: 4,      // greedy: MsPacman, Seaquest, RoadRunner, RiverRaid-robust
  L_wins_robust: 3,      // decoder-robust (win under greedy AND sampling): MsPacman, RoadRunner, RiverRaid
  draws_or_partial: 2,   // Enduro-robust (small), SI-robust (partial), Pong (floor)
  largestGapPct: 82,
  largestGapGame: "River Raid (robust SA)",
} as const;

// Decoder-fragility experiment (2026-06-14): the two zero-variance greedy cells were
// re-run under multinomial sampling (tau=1.0, n=12). They flip in OPPOSITE directions,
// so a decoder-consistent L>T tally stays at 4 of 7 (Seaquest out, Q*bert in) — NOT 5.
// The three high-variance greedy winners (MsPacman/RoadRunner/RiverRaid) win under both.
export interface DecoderCell {
  game: string;
  greedy: { F: number; T: number; L: number };
  sample: { F: number; T: number; L: number };
  flip: string;        // what the sampling decoder changes
  note: string;
}
export const DECODER_FRAGILITY: DecoderCell[] = [
  {
    game: "Q*bert (robust)",
    greedy: { F: 25, T: 125, L: 50 },
    sample: { F: 66.7, T: 87.5, L: 250.0 },
    flip: "T > L greedy → noisy under sampling",
    note: "one τ=1.0 run gave L=250 (2.9×), but it didn't replicate; on the full grid Q*bert is a high-variance tie, not a latent win.",
  },
  {
    game: "Seaquest (bare)",
    greedy: { F: 41.7, T: 63.3, L: 80.0 },
    sample: { F: 63.3, T: 143.3, L: 135.0 },
    flip: "L > T (+26%)  →  L ≈ T (tie, p = 0.60)",
    note: "the headline +26% win was a greedy artifact; both bridges still crush F (slow still helps).",
  },
];

// Best-achievable per-channel (decoder = tuned deployment hyperparameter, selected on held-out
// seeds via leave-one-seed-out). The fixed-greedy "L beats T 26-82%" advantage is decoder-specific;
// tuned per channel the latent is never significantly worse and significantly better on 2/7.
export interface BestAchievable {
  game: string;
  T: number; Tdec: string;
  L: number; Ldec: string;
  B: number;            // combined channel (text + latent together)
  ltVerdict: "L" | "T" | "tie";   // best-L vs best-T
  combineEffect: "interferes" | "complements" | "neutral";  // best-B vs best single
  combinePct: number;   // (B - best_single)/best_single, %
}
export const BEST_ACHIEVABLE: BestAchievable[] = [
  { game: "MsPacman",      T: 401, Tdec: "τ0.5", L: 628, Ldec: "greedy", B: 319, ltVerdict: "L",   combineEffect: "interferes", combinePct: -49 },
  { game: "RoadRunner",    T: 475, Tdec: "greedy", L: 608, Ldec: "greedy", B: 25, ltVerdict: "L",   combineEffect: "interferes", combinePct: -96 },
  { game: "River Raid",    T: 639, Tdec: "τ0.5", L: 566, Ldec: "τ0.7",   B: 452, ltVerdict: "tie", combineEffect: "interferes", combinePct: -29 },
  { game: "Seaquest",      T: 120, Tdec: "τ0.5", L: 130, Ldec: "τ0.7",   B: 138, ltVerdict: "tie", combineEffect: "neutral",    combinePct:  +6 },
  { game: "Q*bert",        T: 185, Tdec: "τ1.5", L: 146, Ldec: "τ1.0",   B: 125, ltVerdict: "tie", combineEffect: "neutral",    combinePct: -33 },
  { game: "Enduro",        T:   3, Tdec: "τ0.5", L:   2, Ldec: "greedy", B:   4, ltVerdict: "tie", combineEffect: "neutral",    combinePct: +25 },
  { game: "SpaceInvaders", T: 162, Tdec: "τ1.0", L: 142, Ldec: "τ1.0",   B: 142, ltVerdict: "tie", combineEffect: "neutral",    combinePct: -13 },
];
// Headline of the best-achievable analysis: 2 significant latent wins, 5 ties, 0 losses;
// combining both channels significantly INTERFERES on 3/7 and never helps → couple via one channel.
export const BEST_ACHIEVABLE_SUMMARY = {
  latentSigWins: 2, latentTies: 5, latentSigLosses: 0,
  combineInterferes: 3, combineComplements: 0, combineNeutral: 4,
} as const;

// The behavioural predictor: latent benefit (L−F) vs text benefit (T−F) across
// 7 Atari games + a non-Atari driving domain (MetaDrive). Numbers read from the
// eval JSONs; canonical = best-L Stage-A variant per game (a tuned hyperparameter).
// MetaDrive uses the planning-heavy regime.
// Pearson r(T−F, L−F) = 0.92 (n=8 best-variant); 0.94 over all 16 game/variant cells.
export interface PredictorPoint {
  game: string;
  domain: "atari" | "driving";
  variant: "bare" | "robust" | "driving";
  F: number; T: number; L: number;
  TmF: number;  // T − F : does slow reasoning beat fast-only?
  LmF: number;  // L − F : does the latent bridge beat fast-only?
}

export const PREDICTOR: PredictorPoint[] = [
  { game: "RoadRunner",    domain: "atari",   variant: "bare",   F: 0.0,    T: 608.3, L: 966.7, TmF: 608.3, LmF: 966.7 },
  { game: "MsPacman",      domain: "atari",   variant: "bare",   F: 255.8,  T: 407.5, L: 628.3, TmF: 151.7, LmF: 372.5 },
  { game: "Qbert",         domain: "atari",   variant: "robust", F: 25.0,   T: 125.0, L: 50.0,  TmF: 100.0, LmF: 25.0  },
  { game: "Seaquest",      domain: "atari",   variant: "bare",   F: 41.7,   T: 63.3,  L: 80.0,  TmF: 21.7,  LmF: 38.3  },
  { game: "Enduro",        domain: "atari",   variant: "bare",   F: 3.2,    T: 0.0,   L: 7.8,   TmF: -3.2,  LmF: 4.5   },
  { game: "MetaDrive",     domain: "driving", variant: "driving",F: 87.8,   T: 85.1,  L: 85.1,  TmF: -2.7,  LmF: -2.7  },
  { game: "SpaceInvaders", domain: "atari",   variant: "robust", F: 107.1,  T: 18.3,  L: 15.0,  TmF: -88.8, LmF: -92.1 },
  { game: "Riverraid",     domain: "atari",   variant: "robust", F: 1032.5, T: 336.7, L: 611.7, TmF: -695.8, LmF: -420.8 },
];

export const PREDICTOR_R = 0.92;
export const PREDICTOR_R_ALL = 0.94;  // all 16 game/variant cells, no selection

// MetaDrive (non-Atari driving) — the controlled negative. Numbers from the
// eval JSONs (n=8, robust head). The bridge-replacement control shows the latent
// is inert: zeroing/randomising it does not lower the score.
export const METADRIVE = {
  reactive: { F: 71.2, T: 69.5, L: 69.5 },        // greedy, default lane-keeping map (matched teacher)
  planningGreedy: { F: 87.8, T: 85.1, L: 85.1 },  // SXSXSX map, greedy
  planningSample: { F: 123.2, T: 43.5, L: 36.7 }, // SXSXSX map, sample τ=1
  control: { L: 85.1, Lzero: 89.5, Lrandom: 97.2 }, // planning greedy bridge-replace
  expertReward: 211, randomReward: 8,
} as const;

// Per-game slow-emission statistics (seed-0 T-trajectory) — the
// quantitative axis for the continuous-vs-categorical claim.
export interface EmissionStats {
  game: string;
  n_emissions: number;
  char_median: number;
  unique_per_emission: number;
  gzip_ratio: number;
  numbers_per_emission: number;
  delta_LT_pct: number | null;
  variant: "bare" | "robust";
}

export const EMISSION_STATS: EmissionStats[] = [
  { game: "RoadRunner",    n_emissions: 8,  char_median: 302, unique_per_emission: 12.75, gzip_ratio: 0.240, numbers_per_emission: 9.4,  delta_LT_pct:  59, variant: "bare"   },
  { game: "SpaceInvaders", n_emissions: 19, char_median: 303, unique_per_emission: 11.68, gzip_ratio: 0.205, numbers_per_emission: 11.4, delta_LT_pct: -17, variant: "robust" },
  { game: "Seaquest",      n_emissions: 32, char_median: 335, unique_per_emission: 11.34, gzip_ratio: 0.213, numbers_per_emission: 8.9,  delta_LT_pct:  26, variant: "bare"   },
  { game: "MsPacman",      n_emissions: 35, char_median: 293, unique_per_emission: 10.14, gzip_ratio: 0.203, numbers_per_emission: 8.6,  delta_LT_pct:  54, variant: "bare"   },
  { game: "Qbert",         n_emissions: 24, char_median: 312, unique_per_emission:  8.12, gzip_ratio: 0.162, numbers_per_emission: 5.9,  delta_LT_pct: -60, variant: "robust" },
  { game: "Riverraid",     n_emissions: 40, char_median: 320, unique_per_emission:  6.95, gzip_ratio: 0.129, numbers_per_emission: 7.2,  delta_LT_pct:  82, variant: "robust" },
  { game: "Enduro",        n_emissions: 50, char_median: 291, unique_per_emission:  4.42, gzip_ratio: 0.078, numbers_per_emission: 9.5,  delta_LT_pct:  19, variant: "robust" },
];

// Raw slow-model emission samples (post-thinking output) used by T verbatim
// and by L after projection into the fast LLM's embedding space.
export const EMISSION_SAMPLES = [
  {
    game: "RoadRunner",
    text: "Got it, let's break this down. The current state: Road Runner is at (3,0), Coyote is at (0,0) with a Δ_x of +3, so Coyote is closing in. The nearest birdseed is at x=159, which is ahead. There are obstacles at (170,16) and (170,0), so trucks/landmines are in the way. First, the …",
    payload: "joint state across (Coyote x-delta, pellet position, obstacle layout) + commitment",
    classification: "continuous-rich",
    diversity: 12.75,
  },
  {
    game: "Q*bert",
    text: "Got it, let's analyze the Q*bert state. The player is at (0,0), Coily is at (65,0) which is on the right edge. The purple and green balls are at y=148, so they're falling. The goal is to jump tiles to change colors, avoid Coily and balls. First, threat: Coily is on the right edge, so if Q*bert moves …",
    payload: "(jump_direction, target_colour, threat_actor)",
    classification: "categorical",
    diversity: 8.12,
  },
] as const;
