import { ChevronRight, AlertTriangle, CheckCircle2, AlertCircle, Shuffle } from "lucide-react";
import { DECODER_FRAGILITY, BEST_ACHIEVABLE } from "../data/games";

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
              <div className="text-good font-semibold mb-1">Tuned per channel, latent is safe-or-better</div>
              <p className="text-ink/85 leading-relaxed">
                The fixed-greedy "L beats T 26–82 %" advantage is <strong>decoder-specific</strong> —
                it vanishes under any sampling temperature (the latent's thin per-emission signal only
                sharpens the greedy argmax). The decoder is a deployment hyperparameter, so the fair
                test gives each channel <em>its own</em> best decoder (selected on held-out seeds):
                there the latent <strong>significantly beats text on 2 of 7</strong> (MsPacman +57 %,
                RoadRunner +28 %), <strong>ties on 5</strong>, and <strong>never significantly loses</strong>.
                The channels prefer <em>different</em> decoders (latent greedy/high-τ, text ≈ 0.5).
              </p>
            </div>
            <div>
              <div className="text-bad font-semibold mb-1">Using both channels at once interferes</div>
              <p className="text-ink/85 leading-relaxed">
                Naively, "more context is better" — so feed the head the text suffix <em>and</em> the
                latent tokens together. It <strong>backfires</strong>: combining never beats the better
                single channel (0 of 7) and significantly <strong>hurts 3 of 7</strong> — MsPacman
                −49 %, River Raid −29 %, and RoadRunner a catastrophic <strong>−96 %</strong>. The
                frozen head, trained on one conditioning signal at a time, gets a worse policy from two.
                <strong> Couple via exactly one channel</strong>, with latent the safe default.
              </p>
            </div>
            <div className="bg-bg/60 rounded-lg p-3 border border-border text-xs text-muted">
              <strong className="text-ink">A rejected predictor:</strong> we tested whether
              slow-emission lexical diversity predicts the sign of L−T — it does not (Pearson
              r = −0.08 over 7 games, n.s.; six other emission features all |r| ≤ 0.25). The axis
              that <em>does</em> predict is behavioural: does slow reasoning beat fast reaction
              (T &gt; F)? See the predictor section.
            </div>
          </div>
        </div>
      </div>

      <div className="bg-panel rounded-2xl border border-border p-6">
        <div className="flex items-center gap-2 mb-3">
          <Shuffle size={16} className="text-accent" />
          <h3 className="font-semibold text-ink">Decoder ablation: greedy flips low-variance cells — both ways</h3>
        </div>
        <p className="text-sm text-ink/90 leading-relaxed mb-4">
          Two games are decided on a <strong>zero-variance greedy cell</strong> (every episode
          identical), where the bridge's small per-emission variance can't move the argmax.
          Re-running both under multinomial sampling (τ = 1.0, n = 12) flips them in
          <strong> opposite</strong> directions — so correcting the decoder swaps Seaquest out and
          Q*bert in, keeping the count at 4 of 7, not 5.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wider text-muted border-b border-border">
                <th className="text-left py-2 font-medium">Game</th>
                <th className="text-right py-2 font-medium">Greedy F / T / L</th>
                <th className="text-right py-2 font-medium">Sample F / T / L</th>
                <th className="text-left py-2 pl-4 font-medium">Flip</th>
              </tr>
            </thead>
            <tbody>
              {DECODER_FRAGILITY.map((d) => (
                <tr key={d.game} className="border-b border-border/50 align-top">
                  <td className="py-2 text-ink font-medium">{d.game}</td>
                  <td className="py-2 text-right font-mono text-xs text-ink/80">
                    {d.greedy.F} / {d.greedy.T} / {d.greedy.L}
                  </td>
                  <td className="py-2 text-right font-mono text-xs text-ink/80">
                    {d.sample.F} / {d.sample.T} / {d.sample.L}
                  </td>
                  <td className="py-2 pl-4">
                    <div className="text-ink/90 font-medium">{d.flip}</div>
                    <div className="text-xs text-muted leading-snug">{d.note}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs text-muted leading-relaxed">
          <strong className="text-ink">Takeaway:</strong> greedy determinism is unreliable for the
          L-vs-T sign — the fixed-greedy advantage doesn't survive sampling on <em>any</em> game. The
          fix is to tune the decoder per channel and report the best-achievable comparison (below).
          The behavioural predictor (T &gt; F) is unaffected: it is about L−F vs T−F, not L vs T.
        </p>
      </div>

      <div className="bg-panel rounded-2xl border border-border p-6">
        <div className="flex items-center gap-2 mb-3">
          <CheckCircle2 size={16} className="text-good" />
          <h3 className="font-semibold text-ink">Best-achievable, held-out — and why "use both" backfires</h3>
        </div>
        <p className="text-sm text-ink/90 leading-relaxed mb-4">
          Decoder = deployment hyperparameter. Each channel at <em>its own</em> best decoder (selected
          on held-out seeds): latent <strong>significantly beats text on 2 of 7</strong>, ties 5, never
          loses. Combining both channels (<strong>B</strong> = text + latent in one pass)
          <strong> never helps and significantly interferes on 3 of 7</strong>.
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
                  <td className="py-2 text-right font-mono text-xs text-ink/80">{d.L} <span className="text-muted">({d.Ldec})</span></td>
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
        <p className="mt-4 text-xs text-muted leading-relaxed">
          <strong className="text-ink">Design rule:</strong> the predictor (T &gt; F) says <em>whether</em>
          to couple; if so, couple via <strong>exactly one channel</strong>, with the latent the
          safe-or-better default. Don't feed both — on RoadRunner that's a −96 % wipeout.
        </p>
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
          <FixCard game="Q*bert"     bare="L=0 (collapsed)"  robust="L=50, T=125"     verdict="rescues both; T leads greedy, L wins 2.9× sampling" verdictCls="text-good" />
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
          MsPacman's 2.9× and RoadRunner's 10.5×. The action head has not learned the basic
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
