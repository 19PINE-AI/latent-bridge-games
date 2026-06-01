import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, ResponsiveContainer, Tooltip,
         ReferenceLine, ReferenceArea, Label, Cell } from "recharts";
import { PREDICTOR, PREDICTOR_R, METADRIVE } from "../data/games";

/** Signed-sqrt transform so RoadRunner (+967) doesn't crush the cluster near 0,
 * while keeping sign and ordering honest. Ticks are relabelled to true deltas. */
const sst = (v: number) => Math.sign(v) * Math.sqrt(Math.abs(v));
const TRUE_TICKS = [-700, -100, -10, 0, 10, 100, 700];

/** The behavioural predictor: the latent bridge helps iff slow reasoning helps the
 * task (T > F). MetaDrive (driving) is the controlled negative at the origin. */
export default function PredictorSection() {
  const data = PREDICTOR.map(p => ({
    game: p.game, domain: p.domain,
    x: sst(p.TmF), y: sst(p.LmF),
    TmF: p.TmF, LmF: p.LmF, F: p.F, T: p.T, L: p.L,
  }));
  const lim = Math.max(...data.flatMap(d => [Math.abs(d.x), Math.abs(d.y)])) * 1.15;

  return (
    <div className="grid lg:grid-cols-[1.5fr_1fr] gap-6">
      {/* left: predictor scatter */}
      <div className="bg-panel rounded-2xl border border-border p-5">
        <h3 className="font-semibold text-ink mb-1">
          Latent helps iff slow reasoning helps&nbsp;
          <span className="text-muted font-normal">(Pearson r = {PREDICTOR_R}, n = 8)</span>
        </h3>
        <p className="text-xs text-muted mb-3">
          Each point is one game. <strong>x</strong> = text-bridge benefit (T − F);
          <strong> y</strong> = latent-bridge benefit (L − F). Signed-√ axis (true reward
          deltas on the ticks). Points hug the diagonal: the bridge pays off only when slow
          reasoning beats fast reaction (upper-right). MetaDrive (driving, ◆) sits at the origin.
        </p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 16, right: 24, bottom: 28, left: 16 }}>
              <ReferenceArea x1={0} x2={lim} y1={0} y2={lim} fill="#5fd991" fillOpacity={0.06} />
              <XAxis type="number" dataKey="x" stroke="#9094a4" fontSize={12}
                     domain={[-lim, lim]} ticks={TRUE_TICKS.map(sst)}
                     tickFormatter={(v: number) => {
                       const t = TRUE_TICKS.find(tt => Math.abs(sst(tt) - v) < 1e-6);
                       return t === undefined ? "" : (t === 0 ? "0" : (t > 0 ? `+${t}` : `${t}`));
                     }}
                     tick={{ fill: "#bfc3d1" }}>
                <Label value="Text bridge benefit  T − F" position="bottom"
                       offset={10} fill="#9094a4" fontSize={11} />
              </XAxis>
              <YAxis type="number" dataKey="y" stroke="#9094a4" fontSize={12}
                     domain={[-lim, lim]} ticks={TRUE_TICKS.map(sst)}
                     tickFormatter={(v: number) => {
                       const t = TRUE_TICKS.find(tt => Math.abs(sst(tt) - v) < 1e-6);
                       return t === undefined ? "" : (t === 0 ? "0" : (t > 0 ? `+${t}` : `${t}`));
                     }}
                     tick={{ fill: "#bfc3d1" }}>
                <Label value="Latent bridge benefit  L − F" angle={-90} position="insideLeft"
                       fill="#9094a4" fontSize={11} dy={60} />
              </YAxis>
              <ZAxis range={[150, 150]} />
              <ReferenceLine y={0} stroke="#586173" />
              <ReferenceLine x={0} stroke="#586173" />
              <ReferenceLine segment={[{ x: -lim, y: -lim }, { x: lim, y: lim }]}
                             stroke="#586173" strokeDasharray="3 3" />
              <Tooltip
                cursor={{ strokeDasharray: "3 3", stroke: "#586173" }}
                content={({ active, payload }) => {
                  if (!active || !payload?.[0]) return null;
                  const p: any = payload[0].payload;
                  return (
                    <div style={{
                      background: "#13151c", border: "1px solid #262a36",
                      borderRadius: 8, padding: 8, fontSize: 12, color: "#e6e7eb",
                    }}>
                      <div className="font-semibold">{p.game}
                        <span className="text-muted text-[10px]"> ({p.domain})</span></div>
                      <div className="text-xs mt-1">F={p.F} · T={p.T} · L={p.L}</div>
                      <div className="text-xs">T−F: {p.TmF > 0 ? "+" : ""}{p.TmF}</div>
                      <div className="text-xs">L−F: {p.LmF > 0 ? "+" : ""}{p.LmF}</div>
                    </div>
                  );
                }}
              />
              <Scatter data={data}>
                {data.map((d, i) => (
                  <Cell key={i}
                        fill={d.domain === "driving" ? "#1f6feb"
                              : (d.LmF > 0 && d.TmF > 0) ? "#5fd991" : "#ff8d50"} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-muted mt-2 leading-snug">
          Upper-right (green) = the bridge helps (T &gt; F and L &gt; F): RoadRunner, MsPacman,
          Seaquest, Q*bert. Lower-left = it hurts: River Raid, Space Invaders. The latent
          tracks the text teacher because Stage C distils it (KL(πL‖πT)) — so the bridge is a
          property of the <em>task</em>, not the channel.
        </p>
      </div>

      {/* right: MetaDrive controlled negative */}
      <div className="bg-panel rounded-2xl border border-border p-5">
        <h3 className="font-semibold text-ink mb-1">MetaDrive: the controlled negative</h3>
        <p className="text-xs text-muted mb-3">
          A real-time driving domain (10 Hz). Even when the task is rebuilt to <em>require</em>
          route planning, slow reasoning never beats fast-only — so the latent has nothing to
          carry.
        </p>
        <table className="w-full text-xs mb-3">
          <thead className="text-muted">
            <tr>
              <th className="text-left pb-2">Regime</th>
              <th className="text-right pb-2">F</th>
              <th className="text-right pb-2">T</th>
              <th className="text-right pb-2">L</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            <tr className="border-t border-border/60">
              <td className="py-2 text-ink font-sans">Reactive · greedy</td>
              <td className="py-2 text-right">{METADRIVE.reactive.F}</td>
              <td className="py-2 text-right">{METADRIVE.reactive.T}</td>
              <td className="py-2 text-right">{METADRIVE.reactive.L}</td>
            </tr>
            <tr className="border-t border-border/60">
              <td className="py-2 text-ink font-sans">Planning · greedy</td>
              <td className="py-2 text-right">{METADRIVE.planningGreedy.F}</td>
              <td className="py-2 text-right">{METADRIVE.planningGreedy.T}</td>
              <td className="py-2 text-right">{METADRIVE.planningGreedy.L}</td>
            </tr>
            <tr className="border-t border-border/60">
              <td className="py-2 text-ink font-sans">Planning · sample</td>
              <td className="py-2 text-right">{METADRIVE.planningSample.F}</td>
              <td className="py-2 text-right text-bad">{METADRIVE.planningSample.T}</td>
              <td className="py-2 text-right text-bad">{METADRIVE.planningSample.L}</td>
            </tr>
          </tbody>
        </table>
        <div className="rounded-lg border border-border/60 p-3 text-[11px] text-muted leading-snug">
          <div className="text-ink font-semibold mb-1">Bridge-replacement control (planning, greedy)</div>
          L<sub>trained</sub> = {METADRIVE.control.L} ·
          L<sub>zero</sub> = {METADRIVE.control.Lzero} ·
          L<sub>random</sub> = {METADRIVE.control.Lrandom}.
          Replacing the trained latent with zeros or random vectors does <strong>not</strong>
          lower the score — the latent is <strong>inert</strong>. Contrast MsPacman, where
          L<sub>trained</sub>=628 ≫ L<sub>random</sub>=387. This is the diagnostic that
          separates a real latent channel from a hollow one.
        </div>
      </div>
    </div>
  );
}
