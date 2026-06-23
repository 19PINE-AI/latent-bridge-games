// Shared chart theme so every recharts panel uses one consistent visual language:
// a stable channel color map (blue = text, orange = latent, slate = fast) and one
// tooltip / axis style instead of copy-pasted inline objects.

/** Canonical per-channel colors. Used everywhere F/T/L appear in a chart. */
export const CHANNEL = { F: "#586173", T: "#7bb5ff", L: "#ffb84d" } as const;

export const SERIES_LABEL = {
  F: "F (fast only)",
  T: "T (text bridge)",
  L: "L (latent bridge)",
} as const;

/** Verdict / quadrant accents (predictor, bridge-replacement). */
export const SIGNAL = { good: "#5fd991", bad: "#ff8d50", driving: "#1f6feb" } as const;

export const AXIS_STROKE = "#9094a4";
export const AXIS_TICK = { fill: "#bfc3d1" } as const;
export const AXIS_FONT = 12;

export const tooltipStyle = {
  background: "#13151c",
  border: "1px solid #262a36",
  borderRadius: 8,
  color: "#e6e7eb",
  fontSize: 12,
} as const;

export const tooltipCursor = { fill: "rgba(255,184,77,0.06)" } as const;
