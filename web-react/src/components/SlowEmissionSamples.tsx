import { MessageSquareText, ArrowRight } from "lucide-react";
import { EMISSION_SAMPLES } from "../data/games";

/** Show the raw slow-emission text for two contrasting games — the qualitative
 * evidence behind the continuous-vs-categorical claim. */
export default function SlowEmissionSamples() {
  return (
    <div className="grid lg:grid-cols-2 gap-5">
      {EMISSION_SAMPLES.map((s) => {
        const isContinuous = s.classification === "continuous-rich";
        return (
          <div key={s.game}
               className="bg-panel rounded-2xl border border-border overflow-hidden">
            <div className="px-5 py-3 bg-panel-2 border-b border-border flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <MessageSquareText size={16} className="text-accent" />
                <h3 className="font-semibold text-ink">{s.game}</h3>
              </div>
              <span className={`text-[11px] font-semibold uppercase tracking-wider px-2 py-1 rounded border ${
                isContinuous
                  ? "bg-good/15 text-good border-good/40"
                  : "bg-bad/15 text-bad border-bad/40"
              }`}>
                {s.classification}
              </span>
            </div>
            <div className="p-5">
              <div className="text-[11px] uppercase tracking-wider text-muted mb-2">
                slow-model emission (post-thinking, ~300 chars, sent verbatim under T)
              </div>
              <blockquote className="bg-bg/60 rounded-lg p-4 border border-border italic
                                     text-sm text-ink/90 leading-relaxed">
                &ldquo;{s.text}&rdquo;
              </blockquote>
              <div className="mt-4 space-y-2 text-xs">
                <Row label="lexical diversity" value={`${s.diversity.toFixed(2)} unique/em`}
                     hint={isContinuous ? "above the ~9 token threshold → L > T predicted"
                                        : "below the ~9 token threshold → T ≥ L predicted"} />
                <Row label="payload" value={s.payload} mono />
                <Row label="latent channel must compress this into"
                     value="8 tokens × 4096-d = 32,768 floats" mono />
              </div>
            </div>
          </div>
        );
      })}
      <div className="lg:col-span-2 bg-panel rounded-2xl border border-border p-5">
        <h3 className="font-semibold text-ink mb-2">Why the two cases pull in opposite directions</h3>
        <div className="grid sm:grid-cols-[1fr_auto_1fr] gap-3 items-center text-sm text-ink/85">
          <div className="bg-panel-2 rounded-lg p-3 border border-border">
            <div className="text-good font-semibold text-xs uppercase tracking-wider mb-1">RoadRunner</div>
            Joint state across multiple entities — Coyote x-delta, pellet x, two obstacle
            (x, y) positions, plus a directional commitment. Text serialization can only
            enumerate one element at a time; the continuous latent channel preserves the
            joint structure.
          </div>
          <ArrowRight className="text-muted hidden sm:block" />
          <div className="bg-panel-2 rounded-lg p-3 border border-border">
            <div className="text-bad font-semibold text-xs uppercase tracking-wider mb-1">Q*bert</div>
            A short categorical triple — pick one of 4 jump directions, one target tile
            (small finite grid), one threat actor. Fits losslessly in 200 characters.
            Compressing 96 slow-residual positions into 8 latent tokens injects more noise
            than the channel saves.
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, hint, mono }: {
  label: string; value: string; hint?: string; mono?: boolean;
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-2">
      <span className="text-muted text-[11px] uppercase tracking-wider">{label}</span>
      <span className={`${mono ? "font-mono" : ""} text-ink/90`}>{value}</span>
      {hint && <span className="text-[11px] text-muted italic">({hint})</span>}
    </div>
  );
}
