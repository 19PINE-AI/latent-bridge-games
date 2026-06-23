import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip,
         Legend, LabelList } from "recharts";
import { GAMES } from "../data/games";
import { CHANNEL, SERIES_LABEL, AXIS_STROKE, AXIS_TICK, tooltipStyle, tooltipCursor } from "../chartTheme";

// Show the reported variant per game (matches the predictor/headline cells)
const PRIMARY_GAMES = [
  "roadrunner-bare",
  "mspacman",
  "riverraid-robust",
  "seaquest",
  "enduro-robust",
  "qbert-robust",
  "spaceinvaders-robust",
];

export default function ScoresChart() {
  const data = PRIMARY_GAMES.map(id => {
    const g = GAMES.find(x => x.id === id)!;
    return {
      name: g.name.replace(" (robust SA)", "*").replace(" (bare SA)", "†"),
      F: g.F, T: g.T, L: g.L,
      L_T: g.L_vs_T_pct,
      category: g.category,
    };
  });

  return (
    <div className="bg-panel rounded-2xl border border-border p-5">
      <div className="flex items-center justify-between mb-4 px-1">
        <h3 className="font-semibold text-ink">Scores per game (12 episodes each)</h3>
        <div className="text-xs text-muted">* robust Stage A · † bare Stage A</div>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 24, right: 16, bottom: 0, left: 0 }}>
            <XAxis dataKey="name" stroke={AXIS_STROKE} fontSize={12}
                   tick={AXIS_TICK} interval={0} />
            <YAxis stroke={AXIS_STROKE} fontSize={12} tick={AXIS_TICK} />
            <Tooltip contentStyle={tooltipStyle} cursor={tooltipCursor} />
            <Legend wrapperStyle={{ fontSize: 12, color: "#bfc3d1" }} />
            <Bar dataKey="F" name={SERIES_LABEL.F} fill={CHANNEL.F} />
            <Bar dataKey="T" name={SERIES_LABEL.T} fill={CHANNEL.T} />
            <Bar dataKey="L" name={SERIES_LABEL.L} fill={CHANNEL.L}>
              <LabelList dataKey="L_T" position="top" fill={CHANNEL.L} fontSize={11}
                         formatter={((v: any) =>
                           v == null ? "" : (v as number) > 0 ? `+${v}%` : `${v}%`) as any} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
