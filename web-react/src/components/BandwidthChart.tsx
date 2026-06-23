import { ComposedChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell,
         ReferenceLine, LabelList } from "recharts";
import { BANDWIDTH } from "../data/strategies";
import { CHANNEL, tooltipStyle, tooltipCursor } from "../chartTheme";

export default function BandwidthChart() {
  const data = BANDWIDTH.map(d => ({
    N: `N = ${d.N}`,
    L: d.L,
    comment: d.comment,
    color: d.N === 8 ? CHANNEL.L : "#9a7b46",
  }));
  return (
    <div className="bg-panel rounded-2xl border border-border p-5">
      <h3 className="font-semibold text-ink mb-1">Latent token-count (N) ablation (MsPacman)</h3>
      <p className="text-xs text-muted mb-3">
        <strong>Matched sweep</strong> (train and deploy both at the same N).
        Goldilocks shape with N = 8 best at matched N. Three points fit any U-shape; we
        present this as consistent with — not a sharp identification of — a peak. This is
        <em> not</em> a bandwidth/capacity story: training at N = 8 and deploying at N = 16
        scores best of all (720), so used capacity sits far below nominal.
      </p>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 24, right: 8, bottom: 0, left: 0 }}>
            <XAxis dataKey="N" stroke="#9094a4" fontSize={12} tick={{ fill: "#bfc3d1" }} />
            <YAxis stroke="#9094a4" fontSize={12} tick={{ fill: "#bfc3d1" }}
                   label={{ value: "L score", angle: -90, position: "insideLeft",
                            fill: "#9094a4", fontSize: 12 }} />
            <Tooltip contentStyle={tooltipStyle} cursor={tooltipCursor} />
            <ReferenceLine y={256} stroke={CHANNEL.F} strokeDasharray="4 4"
                           label={{ value: "F = 256", fill: "#9094a4", fontSize: 10,
                                    position: "left" }} />
            <ReferenceLine y={408} stroke={CHANNEL.T} strokeDasharray="4 4"
                           label={{ value: "T = 408", fill: CHANNEL.T, fontSize: 10,
                                    position: "left" }} />
            <Bar dataKey="L">
              <LabelList dataKey="L" position="top" fill="#ffb84d" fontSize={12}
                         formatter={((v: any) => String(v ?? "")) as any} />
              {data.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
