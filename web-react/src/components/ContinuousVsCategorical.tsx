import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, ResponsiveContainer, Tooltip,
         ReferenceLine, Label, Cell } from "recharts";
import { EMISSION_STATS } from "../data/games";

/** Quantitative axis we TESTED AND REJECTED: lexical diversity (unique whitespace tokens
 * per slow emission) vs (L − T)/T. We hypothesised emission statistics would predict the
 * sign of L − T; they do not (Pearson r ≈ +0.05, n.s.). Retained as a negative result. */
export default function ContinuousVsCategorical() {
  const data = EMISSION_STATS
    .filter(d => d.delta_LT_pct !== null)
    .map(d => ({
      game: d.game,
      x: d.unique_per_emission,
      y: (d.delta_LT_pct ?? 0),
      gzip: d.gzip_ratio,
      n: d.n_emissions,
      variant: d.variant,
    }));

  return (
    <div className="grid lg:grid-cols-[1.4fr_1fr] gap-6">
      <div className="bg-panel rounded-2xl border border-border p-5">
        <h3 className="font-semibold text-ink mb-1">Lexical diversity does <em>not</em> predict L vs T</h3>
        <p className="text-xs text-muted mb-3">
          Each point is one game. <strong>x</strong> = unique whitespace tokens per slow emission
          (seed-0 trajectory). <strong>y</strong> = (L − T) / T. We hypothesised more
          continuous-rich emissions (higher x) would favour the latent — but the cloud is flat
          (Pearson r ≈ +0.05). There is no threshold; emission statistics do not forecast the sign.
        </p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 16, right: 24, bottom: 28, left: 12 }}>
              <XAxis type="number" dataKey="x" stroke="#9094a4" fontSize={12}
                     domain={[4, 14]} tick={{ fill: "#bfc3d1" }}>
                <Label value="Unique whitespace tokens per emission" position="bottom"
                       offset={10} fill="#9094a4" fontSize={11} />
              </XAxis>
              <YAxis type="number" dataKey="y" stroke="#9094a4" fontSize={12}
                     tick={{ fill: "#bfc3d1" }}
                     tickFormatter={(v: any) => `${v > 0 ? "+" : ""}${v}%`}>
                <Label value="(L − T) / T" angle={-90} position="insideLeft"
                       fill="#9094a4" fontSize={11} dy={36} />
              </YAxis>
              <ZAxis range={[140, 140]} />
              <ReferenceLine y={0} stroke="#586173" strokeDasharray="3 3" />
              <Tooltip
                cursor={{ strokeDasharray: "3 3", stroke: "#586173" }}
                contentStyle={{
                  background: "#13151c",
                  border: "1px solid #262a36",
                  borderRadius: 8,
                  color: "#e6e7eb",
                  fontSize: 12,
                }}
                content={({ active, payload }) => {
                  if (!active || !payload?.[0]) return null;
                  const p: any = payload[0].payload;
                  return (
                    <div style={{
                      background: "#13151c", border: "1px solid #262a36",
                      borderRadius: 8, padding: 8, fontSize: 12, color: "#e6e7eb",
                    }}>
                      <div className="font-semibold">{p.game} <span className="text-muted text-[10px]">({p.variant})</span></div>
                      <div className="text-xs mt-1">unique/em: {p.x.toFixed(2)}</div>
                      <div className="text-xs">gzip ratio: {p.gzip.toFixed(3)}</div>
                      <div className="text-xs">ΔL−T: {p.y > 0 ? "+" : ""}{p.y}%</div>
                      <div className="text-xs text-muted">{p.n} emissions</div>
                    </div>
                  );
                }}
              />
              <Scatter data={data}>
                {data.map((d, i) => (
                  <Cell key={i} fill={d.y > 0 ? "#5fd991" : "#ff8d50"} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-muted mt-2 leading-snug">
          n = 7 points (one per evaluable game variant); Pearson r ≈ 0.05 (slope ≈ +0.008 per
          diversity unit, not significant). We report this as a <em>negative</em> result: lexical
          diversity does <strong>not</strong> predict the sign of L − T a priori. The predictor
          that <em>does</em> hold is behavioural (next section): L helps iff T &gt; F.
        </p>
      </div>

      <div className="bg-panel rounded-2xl border border-border p-5">
        <h3 className="font-semibold text-ink mb-3">Emission stats per game</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-muted">
              <tr>
                <th className="text-left pb-2">Game</th>
                <th className="text-right pb-2">uniq/em</th>
                <th className="text-right pb-2">gzip</th>
                <th className="text-right pb-2">#/em</th>
                <th className="text-right pb-2">ΔL−T</th>
              </tr>
            </thead>
            <tbody>
              {EMISSION_STATS.map(s => (
                <tr key={s.game} className="border-t border-border/60">
                  <td className="py-2 text-ink">{s.game}</td>
                  <td className="py-2 text-right font-mono tabular-nums">{s.unique_per_emission.toFixed(2)}</td>
                  <td className="py-2 text-right font-mono tabular-nums">{s.gzip_ratio.toFixed(3)}</td>
                  <td className="py-2 text-right font-mono tabular-nums">{s.numbers_per_emission.toFixed(1)}</td>
                  <td className={`py-2 text-right font-mono tabular-nums ${
                    s.delta_LT_pct == null ? "text-muted" :
                    s.delta_LT_pct > 0 ? "text-good" : "text-bad"
                  }`}>
                    {s.delta_LT_pct == null ? "—" :
                      `${s.delta_LT_pct > 0 ? "+" : ""}${s.delta_LT_pct}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-muted mt-3 leading-snug">
          <strong className="text-ink">uniq/em</strong> = unique whitespace tokens per emission.
          <strong className="text-ink"> gzip</strong> = compression ratio; lower means more
          redundant/categorical. <strong className="text-ink"> #/em</strong> = numeric tokens
          (digits) per emission — proxy for coordinate-heavy state.
        </p>
      </div>
    </div>
  );
}
