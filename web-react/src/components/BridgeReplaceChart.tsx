import { BRIDGE_REPLACE } from "../data/games";

/** Bridge-replacement control across all 7 Atari games: is the latent's effect real
 * learned content, or just the architecture of having prepended tokens? We replace the
 * trained latent with zeros / random-at-matched-norm. Trained >> controls = learned. */
export default function BridgeReplaceChart() {
  // order by T−F so the link to the predictor is visible (helps where slow helps)
  const rows = [...BRIDGE_REPLACE].sort((a, b) => b.tmf - a.tmf);
  const max = Math.max(...rows.flatMap(r => [r.trained, r.zero, r.random]));

  const VC: Record<string, string> = {
    learned: "text-good", inert: "text-muted", harmful: "text-bad",
  };
  const VLABEL: Record<string, string> = {
    learned: "learned content", inert: "inert (≈ controls)", harmful: "harmful (controls win)",
  };

  return (
    <div className="grid lg:grid-cols-[1.6fr_1fr] gap-6">
      <div className="bg-panel rounded-2xl border border-border p-5">
        <h3 className="font-semibold text-ink mb-1">Is the latent real learned content?</h3>
        <p className="text-xs text-muted mb-4">
          For each game, the latent (L) score with the <strong>trained</strong> bridge vs the
          bridge <strong>zeroed</strong> vs <strong>random</strong> vectors at matched norm.
          If trained ≫ both controls, the latent carries genuine learned signal; if the controls
          match or beat it, the effect is architectural or harmful. Games ordered by T − F.
        </p>
        <div className="space-y-3">
          {rows.map(r => (
            <div key={r.game}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-ink font-medium">{r.game}
                  <span className={`ml-2 ${VC[r.verdict]}`}>· {VLABEL[r.verdict]}</span>
                </span>
                <span className="font-mono text-muted">T−F {r.tmf > 0 ? "+" : ""}{r.tmf}</span>
              </div>
              <div className="space-y-1">
                <Bar label="trained" v={r.trained} max={max} cls="bg-accent" highlight />
                <Bar label="zero"    v={r.zero}    max={max} cls="bg-muted/50" />
                <Bar label="random"  v={r.random}  max={max} cls="bg-link/50" />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-panel rounded-2xl border border-border p-5 text-sm leading-relaxed">
        <h3 className="font-semibold text-ink mb-2">What it shows</h3>
        <p className="text-muted">
          The trained latent beats both controls on exactly the three games where slow reasoning
          helps the task (<span className="text-good">RoadRunner, Seaquest, MsPacman</span> — all
          T &gt; F). On RoadRunner the latent is <em>almost entirely</em> learned: zeroing it drops
          the score from 967 to 0.
        </p>
        <p className="text-muted mt-3">
          Where slow reasoning does <em>not</em> help (<span className="text-bad">River Raid,
          Space Invaders</span>, T ≤ F), the trained latent is <strong>harmful</strong> — zeroing
          or randomising it scores better. The same inert/harmful pattern holds for MetaDrive.
        </p>
        <p className="text-muted mt-3">
          So <span className="text-ink">how much of L is learned content is itself predicted by
          T &gt; F</span> — the mechanistic complement to the r = 0.92 predictor. This is the control
          that separates a genuinely informative latent channel from a hollow one.
        </p>
      </div>
    </div>
  );
}

function Bar({ label, v, max, cls, highlight }: {
  label: string; v: number; max: number; cls: string; highlight?: boolean;
}) {
  const pct = max > 0 ? Math.max(2, (v / max) * 100) : 2;
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 text-[10px] text-muted text-right shrink-0">{label}</span>
      <div className="flex-1 bg-panel-2 rounded h-5 relative overflow-hidden border border-border">
        <div className={`h-full ${cls} ${highlight ? "" : "opacity-70"}`} style={{ width: `${pct}%` }} />
        <span className="absolute right-1.5 top-0 h-5 flex items-center text-[10px] font-mono text-ink">
          {Math.round(v)}
        </span>
      </div>
    </div>
  );
}
