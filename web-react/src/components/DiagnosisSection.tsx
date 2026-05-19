import { ChevronRight, AlertTriangle, CheckCircle2 } from "lucide-react";

export default function DiagnosisSection() {
  return (
    <div className="grid lg:grid-cols-[1.2fr_1fr] gap-6">
      <div className="bg-panel rounded-2xl border border-border p-6">
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle size={16} className="text-bad" />
          <h3 className="font-semibold text-ink">The OOD-brittleness diagnosis</h3>
        </div>
        <p className="text-sm leading-relaxed text-ink/90">
          Initial pattern: L &gt; T cleanly on MsPacman / Seaquest / RoadRunner. But L = T = 0
          on <strong>Space Invaders</strong> despite three knob-tuning attempts
          (random-T, expert-T, aggressive-prompt). MI was even <em>positive</em> for the
          expert-T bridge (+0.024 nats action info) — yet deployed L still scored 0.
        </p>
        <div className="my-5 grid sm:grid-cols-2 gap-3">
          <Step n={1} title="Symptom"
                body="Reward-asymmetric games (SI: only FIRE scores) collapse to L=T=0 under bare Stage A. Reward-symmetric games still work." />
          <Step n={2} title="Hypothesis"
                body="Stage A behavioral cloning trained on bare prompts. T appends a text suffix; L prepends bridge tokens. Both are OOD to a frozen head." />
          <Step n={3} title="Test"
                body="Retrain Stage A with --suffix-prob=0.5 (mixed bare/suffix prompts). Validated on 2 of 2 collapsed games." />
          <Step n={4} title="Result"
                body="SI: 0/0 → 15/18 (broke zero floor). River Raid: collapse → +82% L>T (largest gap in the sweep)." />
        </div>
        <div className="flex items-center gap-2 text-sm text-good">
          <CheckCircle2 size={16} />
          <strong>Diagnosis confirmed end-to-end on a 2nd game type.</strong>
        </div>
      </div>

      <div className="bg-panel rounded-2xl border border-border p-6">
        <h3 className="font-semibold text-ink mb-3">Refined headline</h3>
        <div className="space-y-4 text-sm">
          <div>
            <div className="text-good font-semibold mb-1">When latent wins (6 games)</div>
            <p className="text-ink/85 leading-relaxed">
              Slow content is <strong>continuous-rich</strong> — positions, distances, fuel
              levels, multi-entity spatial state. Text serializes lossy; latent preserves
              the structure. Mean gap +44 %, from +18 % (Enduro) to +82 % (RR-robust).
            </p>
          </div>
          <div>
            <div className="text-bad font-semibold mb-1">When text wins (Q*bert)</div>
            <p className="text-ink/85 leading-relaxed">
              Slow content is <strong>categorical</strong> — "jump UP-RIGHT to tile 3,2."
              Discrete decisions compress losslessly into text; latent's continuous
              compression introduces noise. T = 125, L = 50.
            </p>
          </div>
          <div className="bg-bg/60 rounded-lg p-3 border border-border text-xs text-muted">
            <strong className="text-ink">Takeaway:</strong> the bandwidth claim is
            refined, not refuted. Latent &gt; text when slow content is
            continuous-rich; text ≥ latent when content is purely categorical.
          </div>
        </div>
      </div>
    </div>
  );
}

function Step({ n, title, body }: { n: number; title: string; body: string }) {
  return (
    <div className="bg-panel-2 rounded-lg p-3 border border-border">
      <div className="flex items-center gap-2 text-xs text-muted mb-1">
        <span className="w-5 h-5 rounded-full bg-accent/15 text-accent grid place-items-center
                         font-semibold">{n}</span>
        <span className="uppercase tracking-wider">{title}</span>
        <ChevronRight size={12} className="text-muted/60" />
      </div>
      <div className="text-sm text-ink/90 leading-snug">{body}</div>
    </div>
  );
}
