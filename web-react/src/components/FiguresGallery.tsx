import { useState } from "react";
import { X } from "lucide-react";

const FIGURES = [
  { src: "paper/figures/fig_headline.png",
    title: "Headline F / T / L comparison",
    desc: "Per-game bar chart of the four strategies across all positive-result games." },
  { src: "paper/figures/fig_bandwidth.png",
    title: "Bandwidth ablation",
    desc: "Non-monotonic L score vs. N bridge tokens. N=8 is the Goldilocks sweet spot." },
  { src: "paper/figures/fig_continuous_vs_categorical.png",
    title: "Continuous vs categorical games",
    desc: "Refined claim: L > T when slow content has continuous structure; T > L on Q*bert's categorical strategy." },
  { src: "paper/figures/fig_architecture.png",
    title: "Architecture (v2 LLaVA-style bridge)",
    desc: "Slow model's residuals → ThoughtProjection → bridge tokens prepended to fast LLM input." },
  { src: "paper/figures/fig_latency.png",
    title: "Latency breakdown",
    desc: "Per-tick latency by component; vision-token cache halves the warm tick." },
];

export default function FiguresGallery() {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div>
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
        {FIGURES.map((f, i) => (
          <button key={i} onClick={() => setOpen(i)}
                  className="text-left bg-panel rounded-xl border border-border overflow-hidden
                             hover:border-accent/50 transition group">
            <div className="bg-white p-2">
              <img src={f.src} alt={f.title}
                   className="w-full h-44 object-contain
                              group-hover:scale-[1.01] transition" />
            </div>
            <div className="p-4">
              <h3 className="font-semibold text-ink mb-1">{f.title}</h3>
              <p className="text-xs text-muted leading-snug">{f.desc}</p>
            </div>
          </button>
        ))}
      </div>
      {open !== null && (
        <div className="fixed inset-0 z-50 bg-black/85 grid place-items-center p-4"
             onClick={() => setOpen(null)}>
          <button className="absolute top-4 right-4 text-ink p-2 hover:bg-panel rounded"
                  onClick={() => setOpen(null)} aria-label="close">
            <X size={20} />
          </button>
          <div className="max-w-5xl max-h-[90vh] bg-white rounded-lg p-2 overflow-auto"
               onClick={e => e.stopPropagation()}>
            <img src={FIGURES[open].src} alt={FIGURES[open].title}
                 className="max-w-full max-h-[85vh] mx-auto block" />
          </div>
        </div>
      )}
    </div>
  );
}
