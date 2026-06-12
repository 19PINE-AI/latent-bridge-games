// Compact architecture diagram for the hero: two frozen models, two channels
// (text vs latent), one trained projection. Colors match tailwind.config.js.
const C = {
  ink: "#e6e7eb",
  muted: "#9094a4",
  accent: "#ffb84d",
  link: "#7bb5ff",
  border: "#262a36",
  box: "#1a1d27",
};

export default function ArchDiagram() {
  return (
    <svg viewBox="0 0 540 430" role="img"
         aria-label="Architecture: game frames feed a frozen fast model every tick and a frozen slow model about once a second; the slow model's text can be appended (text bridge) or its residuals projected by a trained 33M MLP into 8 latent tokens prepended to the fast model's input (latent bridge); the fast model emits an action every ~67 ms."
         className="w-full h-auto">
      <defs>
        <marker id="ah-muted" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={C.muted} />
        </marker>
        <marker id="ah-accent" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={C.accent} />
        </marker>
        <marker id="ah-link" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={C.link} />
        </marker>
      </defs>

      {/* game frames */}
      <rect x="20" y="24" width="140" height="58" rx="10" fill={C.box} stroke={C.border} />
      <text x="90" y="49" textAnchor="middle" fill={C.ink} fontSize="13" fontWeight="600">Game frames</text>
      <text x="90" y="68" textAnchor="middle" fill={C.muted} fontSize="11">15 Hz ticks</text>

      {/* fast model */}
      <rect x="330" y="20" width="190" height="86" rx="10" fill={C.box} stroke={C.border} />
      <text x="425" y="44" textAnchor="middle" fill={C.ink} fontSize="13" fontWeight="600">Fast model — reacts</text>
      <text x="425" y="62" textAnchor="middle" fill={C.muted} fontSize="11">MiniCPM-o 4.5 · 9 B</text>
      <rect x="396" y="72" width="58" height="18" rx="9" fill="none" stroke={C.link} opacity="0.7" />
      <text x="425" y="85" textAnchor="middle" fill={C.link} fontSize="10.5">frozen</text>

      {/* frames -> fast */}
      <line x1="160" y1="52" x2="326" y2="52" stroke={C.muted} strokeWidth="1.5" markerEnd="url(#ah-muted)" />
      <text x="243" y="44" textAnchor="middle" fill={C.muted} fontSize="10.5">every tick</text>

      {/* frames -> slow (left rail) */}
      <line x1="90" y1="82" x2="90" y2="306" stroke={C.muted} strokeWidth="1.5" markerEnd="url(#ah-muted)" />
      <text x="98" y="196" fill={C.muted} fontSize="10.5">~1 Hz</text>

      {/* slow model */}
      <rect x="20" y="310" width="190" height="86" rx="10" fill={C.box} stroke={C.border} />
      <text x="115" y="334" textAnchor="middle" fill={C.ink} fontSize="13" fontWeight="600">Slow model — reasons</text>
      <text x="115" y="352" textAnchor="middle" fill={C.muted} fontSize="11">Qwen3-VL-8B-Thinking</text>
      <rect x="86" y="362" width="58" height="18" rx="9" fill="none" stroke={C.link} opacity="0.7" />
      <text x="115" y="375" textAnchor="middle" fill={C.link} fontSize="10.5">frozen</text>

      {/* projection MLP (the only trained part) */}
      <rect x="330" y="312" width="190" height="68" rx="10" fill={C.box} stroke={C.accent} />
      <text x="425" y="336" textAnchor="middle" fill={C.ink} fontSize="13" fontWeight="600">Projection MLP</text>
      <text x="425" y="353" textAnchor="middle" fill={C.muted} fontSize="11">4096 → 4096 · 33 M</text>
      <rect x="394" y="359" width="62" height="16" rx="8" fill="none" stroke={C.accent} />
      <text x="425" y="371" textAnchor="middle" fill={C.accent} fontSize="10">trained</text>

      {/* slow -> projection: residuals */}
      <line x1="210" y1="346" x2="326" y2="346" stroke={C.accent} strokeWidth="1.5" markerEnd="url(#ah-accent)" />
      <text x="268" y="338" textAnchor="middle" fill={C.accent} fontSize="10.5">residuals</text>

      {/* projection -> fast: latent tokens (L) */}
      <line x1="400" y1="312" x2="400" y2="110" stroke={C.accent} strokeWidth="1.8" markerEnd="url(#ah-accent)" />
      <text x="408" y="206" fill={C.accent} fontSize="11" fontWeight="600">L · 8 latent tokens</text>
      <text x="408" y="221" fill={C.accent} fontSize="10.5">prepended to input</text>

      {/* slow -> fast: text channel (T), the baseline */}
      <line x1="150" y1="310" x2="356" y2="110" stroke={C.link} strokeWidth="1.5"
            strokeDasharray="5 4" markerEnd="url(#ah-link)" />
      <text x="232" y="262" fill={C.link} fontSize="11" fontWeight="600">T · text emission</text>
      <text x="232" y="277" fill={C.link} fontSize="10.5">appended verbatim</text>

      {/* fast -> action */}
      <line x1="480" y1="106" x2="480" y2="148" stroke={C.muted} strokeWidth="1.5" markerEnd="url(#ah-muted)" />
      <text x="480" y="164" textAnchor="middle" fill={C.ink} fontSize="11" fontWeight="600">action</text>
      <text x="480" y="178" textAnchor="middle" fill={C.muted} fontSize="10.5">every ~67 ms</text>
    </svg>
  );
}
