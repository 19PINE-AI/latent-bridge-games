import { STRATEGIES, LATENCY } from "../data/strategies";
import { Zap } from "lucide-react";

export default function StrategiesTable() {
  return (
    <div className="grid lg:grid-cols-[1.3fr_1fr] gap-6">
      <div className="bg-panel rounded-2xl border border-border overflow-hidden">
        <div className="px-5 py-3 border-b border-border bg-panel-2">
          <h3 className="font-semibold text-ink">Four-strategy comparison (MsPacman)</h3>
          <p className="text-xs text-muted mt-1">
            Ordering <strong>S &lt; F &lt; T &lt; L</strong> — slow alone is too slow;
            fast alone lacks context; latent bridge wins.
          </p>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-panel-2 text-muted">
            <tr>
              <th className="px-4 py-2 text-left">Strategy</th>
              <th className="px-4 py-2 text-right">Score</th>
              <th className="px-4 py-2 text-left">Per-decision latency</th>
            </tr>
          </thead>
          <tbody>
            {STRATEGIES.map(s => {
              const isL = s.id === "L";
              return (
                <tr key={s.id} className={`border-t border-border/60 ${isL ? "bg-good/5" : ""}`}>
                  <td className="px-4 py-3">
                    <div className={`font-semibold ${isL ? "text-good" : "text-ink"}`}>
                      {s.name}
                    </div>
                    <div className="text-xs text-muted mt-0.5">{s.comment}</div>
                  </td>
                  <td className={`px-4 py-3 text-right font-mono tabular-nums text-base ${
                    isL ? "text-good font-bold" : "text-ink"
                  }`}>{s.score}</td>
                  <td className="px-4 py-3 text-sm text-muted">{s.latencyDesc}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="bg-panel rounded-2xl border border-border p-5">
        <div className="flex items-center gap-2 mb-1">
          <Zap size={16} className="text-accent" />
          <h3 className="font-semibold text-ink">Vision cache sweep</h3>
        </div>
        <p className="text-xs text-muted mb-3">
          Per-tick latency drops 48 % when the SigLIP+resampler tower is cached across N
          consecutive fast ticks. Configurable via
          <code className="mx-1 px-1.5 py-0.5 rounded bg-panel-2 text-[11px]">--vision-refresh-every</code>.
        </p>
        <table className="w-full text-sm">
          <thead className="text-muted text-xs uppercase tracking-wider">
            <tr><th className="text-left pb-2">vrf</th>
                <th className="text-right pb-2">Latency</th>
                <th className="text-right pb-2">Speedup</th></tr>
          </thead>
          <tbody>
            {LATENCY.map(l => (
              <tr key={l.vrf} className="border-t border-border/60">
                <td className="py-2.5 font-mono">{l.vrf}{l.vrf === 1 ? " (baseline)" : ""}</td>
                <td className="py-2.5 text-right font-mono">{l.latency} ms</td>
                <td className={`py-2.5 text-right font-semibold ${
                  l.speedup === "—" ? "text-muted" : "text-good"
                }`}>{l.speedup}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-muted mt-3">
          Combined with <code className="px-1 py-0.5 rounded bg-panel-2 text-[11px]">torch.compile</code> on the LLM forward, warm path is well under the 67 ms (15 Hz) target.
        </p>
      </div>
    </div>
  );
}
