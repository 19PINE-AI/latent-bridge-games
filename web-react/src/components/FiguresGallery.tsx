import { useState } from "react";
import { X } from "lucide-react";

const FIGURES = [
  { src: "paper/figures/fig_headline.png",
    title: "Headline F / T / L comparison",
    desc: "Per-game bars with 95 % bootstrap CIs and Welch's-t significance stars. Asterisks on game labels mark robust-Stage-A variants." },
  { src: "paper/figures/fig_continuous_vs_categorical.png",
    title: "Continuous vs categorical (quantitative axis)",
    desc: "Lexical diversity of slow emissions (unique tokens/em) on x; ΔL−T on y. Q*bert is the only T-win and sits at the low-diversity end." },
  { src: "paper/figures/fig_stage_a_ood.png",
    title: "Stage A OOD-brittleness recovery",
    desc: "Bare vs robust Stage A on SpaceInvaders and River Raid. RR recovers to +82 % L−T; SI breaks the zero floor but F still dominates." },
  { src: "paper/figures/fig_bandwidth.png",
    title: "Bandwidth ablation (true sweep)",
    desc: "Train and deploy both at N. Goldilocks shape with N=8 sweet spot. n=3 sweep points." },
  { src: "paper/figures/fig_roadrunner.png",
    title: "RoadRunner close-up",
    desc: "F cannot score under bare Stage A; the slow's directional commitment unlocks L=967 vs T=608." },
  { src: "paper/figures/fig_architecture.png",
    title: "Architecture (v2 LLaVA-style bridge)",
    desc: "Slow model's residuals → 33 M-param projection → 8 bridge tokens prepended to the fast LLM. All 36 layers attend." },
  { src: "paper/figures/fig_latency.png",
    title: "Vision-cache latency × score",
    desc: "Per-tick latency falls 51 % at vrf=15; score is non-monotonic in cache window (perception × policy interaction)." },
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
