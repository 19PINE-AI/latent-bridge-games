// Whole-system diagram first (mirrors the paper's Fig. 2), then the v1-vs-v2
// bridge comparison, drawn rather than described. Shared geometry for v1/v2:
// slow model on the left, the fast LLM as a 36-layer stack on the right; the
// two figures differ in how the bridge enters the stack.
import SystemDiagram from "./SystemDiagram";

const C = {
  ink: "#e6e7eb",
  muted: "#9094a4",
  accent: "#ffb84d",
  good: "#5fd991",
  bad: "#ff6b6b",
  link: "#7bb5ff",
  border: "#262a36",
  box: "#1a1d27",
};

const STRIPES = Array.from({ length: 12 }, (_, i) => 24 + i * 13.7);

function SlowBox() {
  return (
    <>
      <rect x="12" y="72" width="120" height="64" rx="10" fill={C.box} stroke={C.border} />
      <text x="72" y="94" textAnchor="middle" fill={C.ink} fontSize="12.5" fontWeight="600">Slow model</text>
      <text x="72" y="110" textAnchor="middle" fill={C.muted} fontSize="10">Qwen3-VL-8B</text>
      <rect x="48" y="116" width="48" height="14" rx="7" fill="none" stroke={C.link} opacity="0.7" />
      <text x="72" y="126.5" textAnchor="middle" fill={C.link} fontSize="8.5">frozen</text>
    </>
  );
}

function FastStack({ stripeFill, redIdx = [] }: { stripeFill: string; redIdx?: number[] }) {
  return (
    <>
      <rect x="412" y="16" width="96" height="176" rx="10" fill={C.box} stroke={C.border} />
      {STRIPES.map((y, i) => (
        <rect key={i} x="420" y={y} width="80" height="10" rx="2"
              fill={redIdx.includes(i) ? C.bad : stripeFill}
              opacity={redIdx.includes(i) ? 0.85 : 0.3} />
      ))}
      <text x="460" y="208" textAnchor="middle" fill={C.ink} fontSize="10.5" fontWeight="600">Fast LLM · 36 layers</text>
      <text x="460" y="222" textAnchor="middle" fill={C.muted} fontSize="9.5">MiniCPM-o 4.5 · 9 B · frozen</text>
    </>
  );
}

function Arrow({ x1, y1, x2, y2, color, dash }: {
  x1: number; y1: number; x2: number; y2: number; color: string; dash?: boolean;
}) {
  return (
    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth="1.5"
          strokeDasharray={dash ? "5 4" : undefined}
          markerEnd={`url(#arch-ah-${color.slice(1)})`} />
  );
}

function Defs() {
  return (
    <defs>
      {[C.muted, C.accent, C.bad].map(c => (
        <marker key={c} id={`arch-ah-${c.slice(1)}`} viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={c} />
        </marker>
      ))}
    </defs>
  );
}

function V2Diagram() {
  return (
    <div className="overflow-x-auto">
    <svg viewBox="0 0 520 232" className="w-full h-auto min-w-[420px]" role="img"
         aria-label="v2: slow-model layer-24 residuals pass through a trained 33M MLP into 8 latent tokens prepended to the fast LLM's input; all 36 layers attend to them via standard causal attention.">
      <Defs />
      <SlowBox />
      <Arrow x1={132} y1={104} x2={160} y2={104} color={C.muted} />
      {/* projection MLP */}
      <rect x="164" y="72" width="120" height="64" rx="10" fill={C.box} stroke={C.accent} />
      <text x="224" y="93" textAnchor="middle" fill={C.ink} fontSize="12.5" fontWeight="600">MLP</text>
      <text x="224" y="108" textAnchor="middle" fill={C.muted} fontSize="10">4096 → 4096</text>
      <rect x="193" y="114" width="62" height="14" rx="7" fill="none" stroke={C.accent} />
      <text x="224" y="124.5" textAnchor="middle" fill={C.accent} fontSize="8.5">33 M · trained</text>
      <text x="148" y="152" textAnchor="middle" fill={C.muted} fontSize="9">layer-24 residuals · last 8 positions</text>
      {/* 8 latent tokens */}
      <Arrow x1={284} y1={104} x2={306} y2={104} color={C.accent} />
      {Array.from({ length: 8 }, (_, i) => (
        <rect key={i} x={312 + i * 9} y="88" width="6" height="32" rx="1.5" fill={C.accent} opacity="0.85" />
      ))}
      <text x="346" y="80" textAnchor="middle" fill={C.accent} fontSize="9.5" fontWeight="600">N = 8 latent tokens</text>
      <text x="346" y="134" textAnchor="middle" fill={C.accent} fontSize="9">prepended to the input</text>
      <Arrow x1={384} y1={104} x2={408} y2={104} color={C.accent} />
      <FastStack stripeFill={C.good} />
      <text x="12" y="226" fill={C.good} fontSize="10" fontStyle="italic">
        all 36 layers attend over the tokens — standard causal attention
      </text>
    </svg>
    </div>
  );
}

function V1Diagram() {
  return (
    <div className="overflow-x-auto">
    <svg viewBox="0 0 520 232" className="w-full h-auto min-w-[420px]" role="img"
         aria-label="v1: slow-model residuals go into a 256-dimensional ring buffer injected by cross-attention at only layers 12 and 24 of the fast LLM's 36 layers.">
      <Defs />
      <SlowBox />
      <Arrow x1={132} y1={104} x2={160} y2={104} color={C.muted} />
      {/* ring buffer */}
      <rect x="164" y="72" width="120" height="64" rx="10" fill={C.box} stroke={C.bad} opacity="0.95" />
      <text x="224" y="93" textAnchor="middle" fill={C.ink} fontSize="12.5" fontWeight="600">Ring buffer</text>
      <text x="224" y="108" textAnchor="middle" fill={C.muted} fontSize="10">256-d</text>
      <rect x="196" y="114" width="56" height="14" rx="7" fill="none" stroke={C.bad} opacity="0.8" />
      <text x="224" y="124.5" textAnchor="middle" fill={C.bad} fontSize="8.5">8 M · trained</text>
      <text x="148" y="152" textAnchor="middle" fill={C.muted} fontSize="9">layer-24 residuals · last 8 positions</text>
      {/* cross-attn taps into 2 of 36 layers */}
      <Arrow x1={284} y1={96} x2={414} y2={70} color={C.bad} />
      <Arrow x1={284} y1={112} x2={414} y2={125} color={C.bad} />
      <text x="348" y="62" textAnchor="middle" fill={C.bad} fontSize="9.5" fontWeight="600">cross-attn @ layer 24</text>
      <text x="348" y="140" textAnchor="middle" fill={C.bad} fontSize="9.5" fontWeight="600">cross-attn @ layer 12</text>
      <FastStack stripeFill={C.muted} redIdx={[3, 7]} />
      <text x="12" y="226" fill={C.bad} fontSize="10" fontStyle="italic">
        only 2 of 36 layers see the bridge · 256-d is not the LLM's native space
      </text>
    </svg>
    </div>
  );
}

export default function ArchitectureSection() {
  return (
    <div className="space-y-6">
      <div className="bg-panel rounded-2xl border border-border p-6">
        <h3 className="font-semibold text-ink mb-3">
          The whole system{" "}
          <span className="text-muted text-sm font-normal">(paper Fig. 2)</span>
        </h3>
        <SystemDiagram />
        <p className="mt-4 text-sm text-ink/85 leading-relaxed">
          One synchronous loop, one asynchronous one. The fast model picks an
          action every ~67 ms from its input token strip; the slow model reasons
          over structured state about once a second, and the fast loop never
          blocks on it. The three strategies are just three versions of that
          strip: <strong>F</strong> ignores the slow model, <strong>T</strong>{" "}
          appends its text emission verbatim, and <strong>L</strong> projects its
          layer-24 residuals through the 33 M bridge MLP — the only trained
          component — into 8 latent tokens prepended to the input.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
      <div className="bg-panel rounded-2xl border border-border p-6">
        <h3 className="font-semibold text-ink mb-3">
          v2 — LLaVA-style prepend <span className="text-good text-sm font-normal">(works)</span>
        </h3>
        <V2Diagram />
        <p className="mt-4 text-sm text-ink/85 leading-relaxed">
          Only the projection trains (~33 M params); both backbones stay frozen.
          Loss: KL(π<sub>L</sub> ‖ π<sub>T</sub>) — the latent runtime is distilled
          toward the text-bridge teacher. Bridge tokens get exactly the privileges
          text tokens have: the LLM's native input-embedding space, attended by
          every layer.
        </p>
      </div>
      <div className="bg-panel rounded-2xl border border-border p-6">
        <h3 className="font-semibold text-ink mb-3">
          v1 — cross-attention <span className="text-bad text-sm font-normal">(failed)</span>
        </h3>
        <V1Diagram />
        <p className="mt-4 text-sm text-ink/85 leading-relaxed">
          Converged offline (KL = 0.004) yet failed deployed:{" "}
          <strong>L = 225 &lt; F = 256</strong>, bimodal with 4/12 catastrophic
          episodes. Offline KL convergence is necessary but not sufficient —
          adapter-style coupling that reports only offline KL should also report
          deployment.
        </p>
      </div>
      </div>
    </div>
  );
}
