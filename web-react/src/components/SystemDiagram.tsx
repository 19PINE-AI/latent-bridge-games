// Whole-system diagram mirroring the paper's Fig. 2 (fig_system): the
// synchronous fast loop with its input token strip (F/T/L are just three
// versions of that strip), the asynchronous slow path below the divider,
// and the Stage A/B/C training pipeline. Site conventions: orange = latent
// channel / trained component, blue = text channel.
const C = {
  ink: "#e6e7eb",
  muted: "#9094a4",
  accent: "#ffb84d",
  link: "#7bb5ff",
  border: "#262a36",
  box: "#1a1d27",
  panel: "#171a24",
};

function Box({ x, y, w, h, stroke = C.border, lines, fill = C.box }: {
  x: number; y: number; w: number; h: number; stroke?: string; fill?: string;
  lines: { t: string; dy: number; color?: string; size?: number; bold?: boolean }[];
}) {
  return (
    <>
      <rect x={x} y={y} width={w} height={h} rx="10" fill={fill} stroke={stroke} />
      {lines.map((l, i) => (
        <text key={i} x={x + w / 2} y={y + l.dy} textAnchor="middle"
              fill={l.color ?? C.ink} fontSize={l.size ?? 11}
              fontWeight={l.bold ? 600 : 400}>{l.t}</text>
      ))}
    </>
  );
}

function Arrow({ d, color, dash, width = 1.5 }: {
  d: string; color: string; dash?: boolean; width?: number;
}) {
  return (
    <path d={d} fill="none" stroke={color} strokeWidth={width}
          strokeDasharray={dash ? "5 4" : undefined}
          markerEnd={`url(#sys-ah-${color.slice(1)})`} />
  );
}

export default function SystemDiagram() {
  return (
    <svg viewBox="0 0 940 628" className="w-full h-auto" role="img"
         aria-label="System architecture: the environment feeds frames to a frozen fast model whose input token strip is [8 latent tokens | vision tokens | game-state prompt | slow text suffix]; a 36-layer LLM and Stage-A action head emit one action per ~67 ms tick. Asynchronously, a frozen slow model reads structured state about once a second; its text emission is appended as the T channel, and its layer-24 residuals pass through the only trained component, a 33M bridge MLP, into 8 latent tokens prepended as the L channel. Below: the Stage A/B/C training pipeline.">
      <defs>
        {[C.muted, C.accent, C.link].map(c => (
          <marker key={c} id={`sys-ah-${c.slice(1)}`} viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={c} />
          </marker>
        ))}
      </defs>

      {/* ---- action return route (around the top) ---- */}
      <path d="M 861 150 L 861 14 L 80 14 L 80 78" fill="none" stroke={C.muted}
            strokeWidth="1.5" markerEnd="url(#sys-ah-9094a4)" />
      <text x="470" y="32" textAnchor="middle" fill={C.muted} fontSize="10.5">
        action · greedy argmax over game actions · every tick
      </text>

      {/* ---- environment ---- */}
      <Box x={16} y={86} w={130} h={86} lines={[
        { t: "Environment", dy: 28, bold: true, size: 12 },
        { t: "Atari @ 15 Hz", dy: 48, color: C.muted, size: 10 },
        { t: "MetaDrive @ 10 Hz", dy: 64, color: C.muted, size: 10 },
      ]} />

      {/* ---- fast loop panel ---- */}
      <rect x="180" y="40" width="744" height="240" rx="12" fill={C.panel} stroke={C.border} />
      <text x="196" y="64" fill={C.link} fontSize="12" fontWeight="600">
        Fast reactive loop — MiniCPM-o 4.5 · 9 B · frozen — one action per ~67 ms tick
      </text>

      {/* vision tower */}
      <Box x={300} y={84} w={120} h={40} lines={[
        { t: "vision tower · frozen", dy: 24, color: C.muted, size: 10 },
      ]} />
      <Arrow d="M 146 112 L 294 100" color={C.muted} />
      <text x="221" y="96" textAnchor="middle" fill={C.muted} fontSize="9.5">frame</text>
      <Arrow d="M 350 124 L 350 166" color={C.muted} />

      {/* input token strip: [L prefix | vision | state | T suffix] */}
      <text x="238" y="162" textAnchor="middle" fill={C.accent} fontSize="10" fontWeight="600">
        L · 8 latent tokens · prepended
      </text>
      {Array.from({ length: 8 }, (_, i) => (
        <rect key={i} x={200 + i * 10} y="170" width="7" height="45" rx="1.5"
              fill={C.accent} opacity="0.85" />
      ))}
      <Box x={292} y={170} w={110} h={45} lines={[
        { t: "vision tokens", dy: 27, color: C.muted, size: 10 },
      ]} />
      <Box x={410} y={170} w={120} h={45} lines={[
        { t: "game-state prompt", dy: 27, color: C.muted, size: 10 },
      ]} />
      <Box x={538} y={170} w={140} h={45} stroke={C.link} lines={[
        { t: "T · slow text suffix", dy: 20, color: C.link, size: 10, bold: true },
        { t: "appended", dy: 35, color: C.link, size: 9 },
      ]} />
      <Arrow d="M 682 192 L 700 192" color={C.muted} />

      {/* LLM + action head */}
      <Box x={704} y={150} w={96} h={84} lines={[
        { t: "36-layer", dy: 32, bold: true, size: 11.5 },
        { t: "LLM", dy: 48, bold: true, size: 11.5 },
        { t: "frozen", dy: 66, color: C.muted, size: 9.5 },
      ]} />
      <Arrow d="M 800 192 L 810 192" color={C.muted} />
      <Box x={814} y={150} w={94} h={84} lines={[
        { t: "action head", dy: 36, bold: true, size: 11.5 },
        { t: "Stage A", dy: 54, color: C.muted, size: 9.5 },
      ]} />

      {/* F/T/L definition */}
      <text x="908" y="264" textAnchor="end" fill={C.muted} fontSize="10" fontStyle="italic">
        F = no colored segment · T = + blue suffix · L = + orange prefix
      </text>

      {/* ---- sync/async divider ---- */}
      <line x1="16" y1="300" x2="924" y2="300" stroke={C.border} strokeWidth="1.5"
            strokeDasharray="6 5" />
      <text x="100" y="292" fill={C.muted} fontSize="10" fontStyle="italic">synchronous · ~15 Hz</text>
      <text x="100" y="318" fill={C.muted} fontSize="10" fontStyle="italic">asynchronous · ~1 Hz</text>

      {/* ---- slow path ---- */}
      <Arrow d="M 80 172 L 80 332" color={C.muted} />
      <Box x={16} y={336} w={150} h={78} lines={[
        { t: "structured state", dy: 26, bold: true, size: 11.5 },
        { t: "RAM objects /", dy: 46, color: C.muted, size: 10 },
        { t: "driving state", dy: 61, color: C.muted, size: 10 },
      ]} />
      <Arrow d="M 166 375 L 196 375" color={C.muted} />
      <Box x={200} y={336} w={200} h={92} lines={[
        { t: "Slow model — reasons", dy: 28, bold: true, size: 12 },
        { t: "Qwen3-VL-8B-Thinking · frozen", dy: 50, color: C.muted, size: 10 },
        { t: "~1.5 s per emission", dy: 68, color: C.muted, size: 10 },
      ]} />

      {/* T channel: text emission, appended */}
      <Arrow d="M 400 360 L 466 357" color={C.link} />
      <Box x={470} y={330} w={150} h={54} stroke={C.link} lines={[
        { t: "text emission", dy: 22, color: C.link, size: 11, bold: true },
        { t: "~300 chars", dy: 40, color: C.muted, size: 10 },
      ]} />
      <Arrow d="M 600 328 L 600 221" color={C.link} width={1.8} />

      {/* L channel: residuals -> trained MLP -> prepended tokens */}
      <Arrow d="M 400 400 L 466 433" color={C.muted} />
      <Box x={470} y={412} w={150} h={54} lines={[
        { t: "layer-24 residuals", dy: 22, size: 11 },
        { t: "last 8 positions", dy: 40, color: C.muted, size: 10 },
      ]} />
      <Arrow d="M 620 439 L 656 439" color={C.muted} />
      <Box x={660} y={408} w={200} h={68} stroke={C.accent} lines={[
        { t: "bridge MLP · 4096 → 4096", dy: 26, bold: true, size: 11.5 },
        { t: "33 M — the only trained part", dy: 48, color: C.accent, size: 10 },
      ]} />
      <Arrow d="M 760 406 Q 640 250 245 219" color={C.accent} width={1.8} />

      {/* async note */}
      <text x="16" y="448" fill={C.muted} fontSize="9.5" fontStyle="italic">The fast loop never blocks on the slow model;</text>
      <text x="16" y="462" fill={C.muted} fontSize="9.5" fontStyle="italic">the latest emission is reused (~15 ticks)</text>
      <text x="16" y="476" fill={C.muted} fontSize="9.5" fontStyle="italic">until the next one replaces it.</text>

      {/* ---- training pipeline ---- */}
      <text x="16" y="512" fill={C.ink} fontSize="11" fontWeight="600">
        Training pipeline · per game · both base models stay frozen
      </text>
      <Box x={16} y={528} w={280} h={84} lines={[
        { t: "Stage A — action head", dy: 24, bold: true, size: 11.5 },
        { t: "behavioral cloning from SB3 expert", dy: 46, color: C.muted, size: 10 },
        { t: "bare, or robust: suffix-prob 0.5", dy: 64, color: C.muted, size: 10 },
      ]} />
      <Arrow d="M 296 570 L 336 570" color={C.muted} />
      <Box x={340} y={528} w={280} h={84} lines={[
        { t: "Stage B — data", dy: 24, bold: true, size: 11.5 },
        { t: "roll out T; cache frame, slow text,", dy: 46, color: C.muted, size: 10 },
        { t: "layer-24 residuals", dy: 64, color: C.muted, size: 10 },
      ]} />
      <Arrow d="M 620 570 L 656 570" color={C.muted} />
      <Box x={660} y={528} w={264} h={84} stroke={C.accent} lines={[
        { t: "Stage C — bridge", dy: 24, bold: true, size: 11.5 },
        { t: "train the MLP only · KL(πL ‖ πT)", dy: 46, color: C.muted, size: 10 },
        { t: "≈5 K samples/game · final KL ≈ 0.005", dy: 64, color: C.muted, size: 10 },
      ]} />
    </svg>
  );
}
