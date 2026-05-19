import { ArrowDownCircle } from "lucide-react";

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
              9-game empirical sweep · 60+ commits
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight leading-tight">
              Latent Bridge
            </h1>
            <h2 className="text-xl md:text-2xl text-muted font-medium mt-3 leading-snug">
              A continuous-valued bridge between a frozen 9 B reactive model and a frozen 8 B
              reasoning model — beating text-channel coupling on 6 of 7 Atari games.
            </h2>
            <p className="mt-6 text-ink/90 leading-relaxed">
              Real-time interactive AI is structurally torn: reasoning models tick too slowly
              for 60 Hz environments; streaming models lack reasoning depth. Production splits
              the workload via <em>text prompts</em> — which we argue is bandwidth-limited.
              Text carries hundreds of bits per call where a continuous latent channel could
              carry hundreds of thousands.
            </p>
            <p className="mt-3 text-ink/90 leading-relaxed">
              We test this on Atari: MiniCPM-o 4.5 at 15 Hz, Qwen3-VL-8B-Thinking at 1 Hz,
              coupled via a trainable <strong>33 M-param latent bridge</strong> prepended
              LLaVA-style to the fast model's input embedding.
            </p>

            <div className="mt-8 grid grid-cols-3 gap-4">
              <Stat label="L > T on" value="6 / 7" sub="games (continuous-content)" />
              <Stat label="Largest gap" value="+82 %" sub="River Raid (robust SA)" />
              <Stat label="Total trainable" value="33 M" sub="of ~17 B params" />
            </div>

            <a href="#results"
               className="inline-flex items-center gap-2 mt-9 px-5 py-2.5 rounded-lg
                          bg-accent text-bg font-semibold hover:bg-accent/90 transition">
              <ArrowDownCircle size={18} />
              See the data
            </a>
          </div>

          <div className="reveal">
            <div className="bg-panel rounded-2xl p-4 border border-border shadow-soft">
              <div className="px-2 pb-2 flex items-center justify-between text-xs">
                <span className="text-muted uppercase tracking-wider">
                  Narrated demo · 4 min 14 s
                </span>
                <a href="demos/combined_narrated.srt" download
                   className="text-link hover:underline">subtitles (SRT)</a>
              </div>
              <video src="demos/combined_narrated.mp4" controls playsInline
                     preload="metadata"
                     className="w-full rounded-lg bg-black aspect-[16/6.24]" />
              <p className="mt-3 px-2 text-xs text-muted leading-relaxed">
                All 7 games with title cards, headline numbers, and gTTS narration.
                Tells the story in order: Road Runner → Ms. Pac-Man → River Raid (+82 % win)
                → Seaquest → Q*bert (the categorical exception) → Space Invaders (the
                diagnosis) → Enduro.
              </p>
            </div>
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
