import { ComposedChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell,
         ReferenceLine, LabelList } from "recharts";
import { BANDWIDTH } from "../data/strategies";

export default function BandwidthChart() {
  const data = BANDWIDTH.map(d => ({
    N: `N = ${d.N}`,
    L: d.L,
    comment: d.comment,
    color: d.N === 8 ? "#5fd991" : "#a48fff",
  }));
  return (
    <div className="bg-panel rounded-2xl border border-border p-5">
      <h3 className="font-semibold text-ink mb-1">Bridge bandwidth ablation (MsPacman)</h3>
      <p className="text-xs text-muted mb-3">
        Latent score is <strong>non-monotonic</strong> in N. Sweet spot at N = 8.
      </p>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 24, right: 8, bottom: 0, left: 0 }}>
            <XAxis dataKey="N" stroke="#9094a4" fontSize={12} tick={{ fill: "#bfc3d1" }} />
            <YAxis stroke="#9094a4" fontSize={12} tick={{ fill: "#bfc3d1" }}
                   label={{ value: "L score", angle: -90, position: "insideLeft",
                            fill: "#9094a4", fontSize: 12 }} />
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
            <ReferenceLine y={256} stroke="#586173" strokeDasharray="4 4"
                           label={{ value: "F = 256", fill: "#9094a4", fontSize: 10,
                                    position: "left" }} />
            <ReferenceLine y={408} stroke="#5894ff" strokeDasharray="4 4"
                           label={{ value: "T = 408", fill: "#5894ff", fontSize: 10,
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
