import { ArrowDownCircle } from "lucide-react";
import ArchDiagram from "./ArchDiagram";

export default function Hero() {
  return (
    <section id="hero" className="border-b border-border">
      <div className="max-w-6xl mx-auto px-6 pt-16 pb-12">
        <div className="grid lg:grid-cols-[1.1fr_1fr] gap-10 items-center">
          <div className="reveal">
            <div className="inline-flex items-center gap-2 px-3 py-1 mb-5
                            rounded-full bg-panel border border-border text-xs uppercase
                            tracking-wider text-muted">
              <span className="w-1.5 h-1.5 rounded-full bg-good animate-pulse" />
              7 Atari games + a driving domain · 12 episodes per cell · 95 % bootstrap CIs
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight leading-tight">
              Latent Bridge
            </h1>
            <h2 className="text-xl md:text-2xl text-muted font-medium mt-3 leading-snug">
              A continuous-valued bridge between a frozen 9 B reactive model and a frozen 8 B
              reasoning model — and a sharp answer to <em>when</em> it helps.
            </h2>
            <p className="mt-6 text-ink/90 leading-relaxed">
              We want agents that operate a computer like a person — read the screen, issue
              inputs, close the loop — and real-time games are the hardest case: act every few
              tens of milliseconds while planning over seconds. No single open model does both: a
              reasoning VLM is ~1.5 s too slow for the <strong>~15 Hz</strong> control loop, while
              a reactive VLM has no deliberation. The fast/slow split is the fix — Thinking
              Machines' <em>Interaction Models</em> make it explicit via shared text/context. We
              test an open alternative — a trainable <strong>33 M-param latent bridge</strong>{" "}
              prepended LLaVA-style to the fast model's input embedding, carrying the slow
              model's residuals instead of its words.
            </p>
            <p className="mt-3 text-ink/90 leading-relaxed">
              Across 7 Atari games and a driving simulator (MetaDrive), the headline is not
              &ldquo;latent beats text&rdquo; — it is a <strong>predictor</strong>: the latent
              bridge helps <em>if and only if</em> slow reasoning beats fast reaction on the task
              (T &gt; F), at Pearson <strong>r = 0.93</strong>. A bridge-replacement control on
              every game confirms the latent carries real learned content exactly where it helps,
              and is inert or harmful where it does not — MetaDrive being the controlled negative.
            </p>

            <div className="mt-8 grid grid-cols-3 gap-4">
              <Stat label="Predictor" value="r = 0.93" sub="L−F tracks T−F across 8 tasks" />
              <Stat label="Latent, tuned" value="2/7 wins" sub="sig. beats text; ties 5, never loses" />
              <Stat label="Total trainable" value="33 M" sub="of ~17 B params, both frozen" />
            </div>

            <a href="#replay"
               className="inline-flex items-center gap-2 mt-9 px-5 py-2.5 rounded-lg
                          bg-accent text-bg font-semibold hover:bg-accent/90 transition">
              <ArrowDownCircle size={18} />
              Watch fast vs latent
            </a>
          </div>

          {/* architecture at a glance */}
          <div className="reveal">
            <div className="bg-panel rounded-2xl p-5 border border-border shadow-soft">
              <div className="px-1 pb-3 text-xs text-muted uppercase tracking-wider">
                Architecture at a glance
              </div>
              <ArchDiagram />
              <p className="mt-3 px-1 text-xs text-muted leading-relaxed">
                Both models stay frozen; only the 33 M-param projection trains. The experiment
                holds everything else fixed and swaps the channel: the slow model's{" "}
                <span className="text-link">words (T)</span> vs its projected{" "}
                <span className="text-accent">latent (L)</span>. Details in{" "}
                <a href="#architecture" className="text-link hover:underline">Architecture</a>.
              </p>
            </div>
          </div>
        </div>

        {/* The narrated video is ~3.85:1, so it gets its own full-width row
            rather than sharing a two-column grid where it renders too small. */}
        <div className="reveal mt-10">
          <div className="bg-panel rounded-2xl p-4 border border-border shadow-soft">
            <div className="px-2 pb-2 flex items-center justify-between text-xs">
              <span className="text-muted uppercase tracking-wider">
                Narrated demo · 3 min 47 s
              </span>
              <a href="demos/combined_narrated.srt" download
                 className="text-link hover:underline">subtitles (SRT)</a>
            </div>
            <video src="demos/combined_narrated.mp4" controls playsInline
                   preload="metadata"
                   className="w-full rounded-lg bg-black aspect-[3376/876]" />
            <p className="mt-3 px-2 text-xs text-muted leading-relaxed">
              Full narrated tour with 3-way F/T/L clips for all 7 Atari games and MetaDrive,
              then the headline findings: Road Runner → Ms. Pac-Man → River Raid → Seaquest →
              Q*bert (decoder-dependent) → Space Invaders → Enduro → MetaDrive (the controlled
              negative) → the T &gt; F predictor (r = 0.93) → the bridge-replacement control.
              Interactive versions in <a href="#replay" className="text-link hover:underline">Replay</a> and
              {" "}<a href="#predictor" className="text-link hover:underline">Predictor</a> below.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="bg-panel-2 rounded-lg p-3 border border-border">
      <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
      <div className="text-2xl font-bold text-ink leading-tight">{value}</div>
      <div className="text-xs text-muted mt-1 leading-tight">{sub}</div>
    </div>
  );
}
