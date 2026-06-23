import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip,
         Legend, LabelList } from "recharts";
import { GAMES } from "../data/games";

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
            <XAxis dataKey="name" stroke="#9094a4" fontSize={12}
                   tick={{ fill: "#bfc3d1" }} interval={0} />
            <YAxis stroke="#9094a4" fontSize={12} tick={{ fill: "#bfc3d1" }} />
            <Tooltip
              contentStyle={{
                background: "#13151c",
                border: "1px solid #262a36",
                borderRadius: 8,
                color: "#e6e7eb",
                fontSize: 12,
              }}
              cursor={{ fill: "rgba(255,184,77,0.06)" }}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: "#bfc3d1" }} />
            <Bar dataKey="F" name="F (fast only)" fill="#586173" />
            <Bar dataKey="T" name="T (text bridge)" fill="#5894ff" />
            <Bar dataKey="L" name="L (latent bridge)">
              <LabelList dataKey="L_T" position="top" fill="#ffb84d" fontSize={11}
                         formatter={((v: any) =>
                           v == null ? "" : (v as number) > 0 ? `+${v}%` : `${v}%`) as any} />
              {data.map((d, i) => (
                <Cell key={i} fill={d.category === "win-L" ? "#5fd991" :
                                    d.category === "win-T" ? "#ff8d50" :
                                    "#a48fff"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
