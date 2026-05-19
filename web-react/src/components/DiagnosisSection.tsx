import { ChevronRight, AlertTriangle, CheckCircle2, AlertCircle } from "lucide-react";

export default function DiagnosisSection() {
  return (
    <div className="space-y-6">
      <div className="grid lg:grid-cols-[1.2fr_1fr] gap-6">
        <div className="bg-panel rounded-2xl border border-border p-6">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle size={16} className="text-bad" />
            <h3 className="font-semibold text-ink">The OOD-brittleness diagnosis</h3>
          </div>
          <p className="text-sm leading-relaxed text-ink/90">
            Initial pattern: L &gt; T cleanly on MsPacman / Seaquest / RoadRunner. But L = T = 0
            on <strong>Space Invaders</strong> and <strong>River Raid</strong> despite three knob-tuning attempts
            on SI (random-T, expert-T, aggressive-prompt). MI was even <em>positive</em> for the
            expert-T bridge (+0.024 nats action info) — the bridge had learned structure; it just
            couldn't translate it into behaviour at deployment.
          </p>
          <div className="my-5 grid sm:grid-cols-2 gap-3">
            <Step n={1} title="Symptom"
                  body="Reward-asymmetric games (SI: only FIRE scores; RR: precise dodging required) collapse to L=T=0 under bare Stage A. Reward-symmetric games still score." />
            <Step n={2} title="Hypothesis"
                  body="Stage A trained on bare prompts. T appends a text suffix; L prepends bridge tokens. Both are OOD to a frozen action head." />
            <Step n={3} title="Test"
                  body="Retrain Stage A at suffix-probability 0.5 — half of training batches receive a synthetic slow-style suffix prepended to the prompt." />
            <Step n={4} title="Result"
                  body="River Raid: bridges collapsed → L=612, T=337, +82 % L−T (largest gap in the sweep). SI: T=L=0 → T=18, L=15 (zero floor broken but F=107 still dominates)." />
          </div>
          <div className="flex items-center gap-2 text-sm text-good">
            <CheckCircle2 size={16} />
            <strong>Diagnosis confirmed:</strong>
            <span className="text-ink/80 font-normal">collapse cause is the frozen action head, not the bridge.</span>
          </div>
        </div>

        <div className="bg-panel rounded-2xl border border-border p-6">
          <h3 className="font-semibold text-ink mb-3">Refined headline</h3>
          <div className="space-y-4 text-sm">
            <div>
              <div className="text-good font-semibold mb-1">When L &gt; T (4 games)</div>
              <p className="text-ink/85 leading-relaxed">
                Slow content is <strong>continuous-rich</strong> — positions, distances, fuel
                levels, multi-entity spatial state. Text serialization is lossy; latent
                preserves the joint structure. Gap range: +26 % (Seaquest) to +82 % (River Raid robust).
              </p>
            </div>
            <div>
              <div className="text-bad font-semibold mb-1">When T &gt; L (Q*bert)</div>
              <p className="text-ink/85 leading-relaxed">
                Slow content is <strong>categorical</strong> — &ldquo;jump UP-RIGHT to tile 3, 2.&rdquo;
                Discrete decisions compress losslessly into text; latent's continuous
                compression introduces noise. T = 125, L = 50.
              </p>
            </div>
            <div className="bg-bg/60 rounded-lg p-3 border border-border text-xs text-muted">
              <strong className="text-ink">Falsifiable a priori:</strong> compute slow-emission lexical
              diversity (unique whitespace tokens per emission) on a cheap Stage B trajectory and
              predict the sign of L−T. Threshold around ~9 tokens/emission. See the
              continuous-vs-categorical scatter below.
            </div>
          </div>
        </div>
      </div>

      <div className="bg-panel rounded-2xl border border-border p-6">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle size={16} className="text-accent" />
          <h3 className="font-semibold text-ink">The fix is asymmetric — not a universal remedy</h3>
        </div>
        <p className="text-sm text-ink/90 leading-relaxed mb-4">
          Mixed-prompt Stage A retraining helps the collapsed games but
          <strong> destroys</strong> games where the bare bridge already worked. The directionality
          of the fix also differs by game.
        </p>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <FixCard game="River Raid" bare="L=360 (collapsed)" robust="L=612 (+82 %)" verdict="rescues L" verdictCls="text-good" />
          <FixCard game="Q*bert"     bare="L=0 (collapsed)"  robust="L=50, T=125"     verdict="rescues both; T leads" verdictCls="text-good" />
          <FixCard game="Space Invaders" bare="T=L=0"        robust="L=15, T=18"      verdict="breaks zero floor; F=107 still dominates" verdictCls="text-accent" />
          <FixCard game="MsPacman"   bare="L=628 ✓"          robust="L=60 ✗"          verdict="destroys L" verdictCls="text-bad" />
        </div>
        <p className="mt-4 text-xs text-muted leading-relaxed">
          <strong className="text-ink">Recipe:</strong> apply robust Stage A only when bare T or L
          collapses to ~0. On games where bare deployment is non-degenerate, the small bare-prompt
          val-accuracy drop dominates any suffix-robustness gain.
        </p>
      </div>

      <div className="bg-panel rounded-2xl border border-border p-6">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle size={16} className="text-muted" />
          <h3 className="font-semibold text-ink">Pong: the floor case</h3>
        </div>
        <p className="text-sm text-ink/90 leading-relaxed">
          Pong sits at the −21 loss floor under all three policies. Stage A val-accuracy is only
          <strong> 25.1 %</strong> (random = 16.7 % on 6 actions), just 1.5× random — versus
          MsPacman's 2.9× and RoadRunner's 5.3×. The action head has not learned the basic
          paddle-tracking reflex. The slow's strategic guidance (&ldquo;move paddle to match ball y&rdquo;)
          cannot rescue a reactive policy that does not yet exist. Fast/slow coupling presupposes
          a working fast.
        </p>
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

function FixCard({ game, bare, robust, verdict, verdictCls }: {
  game: string; bare: string; robust: string; verdict: string; verdictCls: string;
}) {
  return (
    <div className="bg-panel-2 rounded-lg p-3 border border-border">
      <div className="font-medium text-ink mb-2 text-sm">{game}</div>
      <div className="text-[11px] uppercase tracking-wider text-muted">Bare SA</div>
      <div className="font-mono text-xs text-ink/85 mb-2">{bare}</div>
      <div className="text-[11px] uppercase tracking-wider text-muted">Robust SA</div>
      <div className="font-mono text-xs text-ink/85 mb-2">{robust}</div>
      <div className={`text-[11px] font-semibold ${verdictCls}`}>{verdict}</div>
    </div>
  );
}
