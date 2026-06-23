import { CheckCircle2, Trophy } from "lucide-react";
import { BEST_ACHIEVABLE, BEST_ACHIEVABLE_SUMMARY as S } from "../data/games";

/** The headline result: with the decoder tuned per channel on held-out seeds, the latent
 * bridge significantly beats text on 2/7, ties the rest, never loses — and combining both
 * channels interferes. This is the fair comparison the paper leads with. */
export default function BestAchievableSection() {
  return (
    <div className="space-y-6">
      <div className="grid sm:grid-cols-3 gap-4">
        <Stat value={`${S.latentSigWins} / 7`} label="Latent significantly beats text"
              sub="MsPacman +57%, RoadRunner +28%" tone="good" />
        <Stat value={`${S.latentTies} / 7`} label="Statistical ties"
              sub="latent never significantly worse" tone="muted" />
        <Stat value={`${S.combineInterferes} / 7`} label="Combining both channels hurts"
              sub="RoadRunner −96%; never helps" tone="bad" />
      </div>

      <div className="bg-panel rounded-2xl border border-border p-5 sm:p-6">
        <div className="flex items-center gap-2 mb-1">
          <Trophy size={16} className="text-good" />
          <h3 className="font-semibold text-ink">Best-achievable F/T/L, decoder tuned per channel (held-out)</h3>
        </div>
        <p className="text-sm text-muted mb-4 max-w-3xl">
          The action decoder is a deployment hyperparameter, so each channel gets its own best
          decoder, selected on held-out seeds (leave-one-seed-out). <strong className="text-ink">B</strong> is the
          combined channel — text suffix <em>and</em> latent tokens in one forward pass.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wider text-muted border-b border-border">
                <th className="text-left py-2 font-medium">Game</th>
                <th className="text-right py-2 font-medium">best T</th>
                <th className="text-right py-2 font-medium">best L</th>
                <th className="text-right py-2 font-medium">best B (both)</th>
                <th className="text-left py-2 pl-3 font-medium">L vs T</th>
                <th className="text-left py-2 pl-3 font-medium">combine</th>
              </tr>
            </thead>
            <tbody>
              {BEST_ACHIEVABLE.map((d) => (
                <tr key={d.game} className="border-b border-border/50">
                  <td className="py-2 text-ink font-medium">{d.game}</td>
                  <td className="py-2 text-right font-mono text-xs text-ink/80">{d.T} <span className="text-muted">({d.Tdec})</span></td>
                  <td className={`py-2 text-right font-mono text-xs ${d.ltVerdict === "L" ? "text-good font-semibold" : "text-ink/80"}`}>{d.L} <span className="text-muted">({d.Ldec})</span></td>
                  <td className="py-2 text-right font-mono text-xs text-ink/80">{d.B}</td>
                  <td className={`py-2 pl-3 text-xs ${d.ltVerdict === "L" ? "text-good font-semibold" : "text-muted"}`}>
                    {d.ltVerdict === "L" ? "L wins" : d.ltVerdict === "T" ? "T wins" : "tie"}
                  </td>
                  <td className={`py-2 pl-3 text-xs ${d.combineEffect === "interferes" ? "text-bad font-semibold" : "text-muted"}`}>
                    {d.combineEffect}{d.combineEffect === "interferes" ? ` ${d.combinePct}%` : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs text-muted leading-relaxed flex items-start gap-2">
          <CheckCircle2 size={14} className="text-good shrink-0 mt-0.5" />
          <span>
            <strong className="text-ink">Design rule:</strong> the predictor (T &gt; F, next section)
            says <em>whether</em> to couple; if so, couple via <strong className="text-ink">exactly one
            channel</strong>, with the latent the safe-or-better default. Don't feed both — on
            RoadRunner that is a −96% wipeout. The per-game greedy scores (a rosier but
            decoder-specific picture) are in the <a href="#results" className="text-link hover:underline">full
            results table</a>.
          </span>
        </p>
      </div>
    </div>
  );
}

function Stat({ value, label, sub, tone }: {
  value: string; label: string; sub: string; tone: "good" | "bad" | "muted";
}) {
  const valueCls = tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : "text-ink";
  return (
    <div className="bg-panel rounded-2xl border border-border p-5">
      <div className={`text-3xl font-bold leading-tight ${valueCls}`}>{value}</div>
      <div className="text-sm text-ink/90 mt-1 font-medium">{label}</div>
      <div className="text-xs text-muted mt-0.5 leading-snug">{sub}</div>
    </div>
  );
}
