import { useState, useRef } from "react";
import { Play, RotateCcw, ArrowLeftRight } from "lucide-react";
import { GAMES, type GameResult } from "../data/games";
import research from "../data/research.json";

// Games that have replay videos, ordered by the L-over-T story.
const ORDER = ["roadrunner", "mspacman", "riverraid", "seaquest", "qbert",
               "spaceinvaders", "enduro", "metadrive"];
const REPLAY = ORDER
  .map(id => GAMES.find(g => g.id.startsWith(id) && g.videoSideBySide))
  .filter(Boolean) as GameResult[];

const RESEARCH: any = research;
// map a game card to its research.json key
function researchKey(g: GameResult): string {
  const n = g.name.replace(/[^A-Za-z]/g, "");
  const map: Record<string, string> = {
    RoadRunner: "RoadRunner", MsPacMan: "MsPacman", MsPacman: "MsPacman",
    RiverRaid: "Riverraid", Seaquest: "Seaquest", Qbert: "Qbert",
    QBert: "Qbert", SpaceInvaders: "SpaceInvaders", Enduro: "Enduro",
    MetaDrive: "MetaDrive",
  };
  return map[n] ?? n;
}

export default function ReplayTheater() {
  const [sel, setSel] = useState(0);
  const [mode, setMode] = useState<"sideBySide" | "split">("split");
  const fRef = useRef<HTMLVideoElement>(null);
  const lRef = useRef<HTMLVideoElement>(null);

  const g = REPLAY[sel];
  const rk = researchKey(g);
  const rdata = RESEARCH.games[rk] ?? {};
  const gap = g.L_vs_T_pct;

  const replayBoth = () => {
    for (const v of [fRef.current, lRef.current]) {
      if (v) { v.currentTime = 0; v.play().catch(() => {}); }
    }
  };

  return (
    <div>
      {/* game selector */}
      <div className="flex flex-wrap gap-2 mb-5">
        {REPLAY.map((gg, i) => {
          const gp = gg.L_vs_T_pct;
          return (
            <button key={gg.id} onClick={() => setSel(i)}
              className={`px-3 py-1.5 rounded-lg text-sm border transition ${
                i === sel ? "bg-accent text-bg border-accent font-semibold"
                          : "bg-panel border-border text-muted hover:border-accent/40"}`}>
              {gg.name}
              {gp != null && (
                <span className={`ml-2 text-[10px] font-mono ${
                  i === sel ? "text-bg/70"
                  : gp > 0 ? "text-good" : gp < 0 ? "text-bad" : "text-muted"}`}>
                  {gp > 0 ? `+${gp}%` : gp < 0 ? `${gp}%` : "="}
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="grid lg:grid-cols-[1.5fr_1fr] gap-6">
        {/* left: contrastive video */}
        <div className="bg-panel rounded-2xl border border-border p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-ink">{g.name}
              <span className="text-muted font-normal text-sm"> · F (fast-only) vs L (latent bridge)</span>
            </h3>
            <div className="flex gap-2">
              <button onClick={() => setMode(mode === "split" ? "sideBySide" : "split")}
                className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded
                           bg-panel-2 border border-border text-muted hover:text-ink transition">
                <ArrowLeftRight size={13} /> {mode === "split" ? "Pre-rendered" : "Synced split"}
              </button>
              {mode === "split" && (
                <button onClick={replayBoth}
                  className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded
                             bg-accent/15 border border-accent/40 text-accent hover:bg-accent/25 transition">
                  <RotateCcw size={13} /> Replay both
                </button>
              )}
            </div>
          </div>

          {mode === "split" ? (
            <div className="grid grid-cols-2 gap-3">
              <Panel label="F · fast-only" color="text-muted" score={g.F}>
                <video ref={fRef} src={g.videoF} playsInline loop muted controls
                       preload="metadata" className="w-full rounded bg-black" />
              </Panel>
              <Panel label="L · latent bridge" color="text-accent" score={g.L} highlight>
                <video ref={lRef} src={g.videoL} playsInline loop muted controls
                       preload="metadata" className="w-full rounded bg-black" />
              </Panel>
            </div>
          ) : (
            <video src={g.videoSideBySide} controls playsInline loop
                   preload="metadata" className="w-full rounded-lg bg-black" />
          )}

          <div className="mt-4 grid grid-cols-4 gap-2 text-center font-mono tabular-nums">
            <ScoreBox label="F" v={g.F} />
            <ScoreBox label="T" v={g.T} />
            <ScoreBox label="L" v={g.L} highlight />
            <div className="bg-panel-2 rounded-md py-1.5 px-2 border border-border">
              <div className="text-[10px] uppercase tracking-wider text-muted">L vs T</div>
              <div className={`text-sm leading-tight font-semibold ${
                gap == null ? "text-muted" : gap > 0 ? "text-good" : gap < 0 ? "text-bad" : "text-muted"}`}>
                {gap == null ? "—" : gap > 0 ? `+${gap}%` : `${gap}%`}
              </div>
            </div>
          </div>
          <p className="mt-3 text-xs text-muted leading-relaxed">{g.notes}</p>
        </div>

        {/* right: the reasoning the bridge transmits */}
        <div className="bg-panel rounded-2xl border border-border p-4 flex flex-col">
          <h3 className="font-semibold text-ink mb-1 flex items-center gap-2">
            <Play size={15} className="text-accent" /> What the slow model sends
          </h3>
          <p className="text-xs text-muted mb-3">
            The exact state snapshot the slow model reads (~1 Hz), and the strategic
            emission it returns. T sends this <em>text verbatim</em>; L sends its
            projected <em>latent</em> — same content, different channel.
          </p>

          <div className="text-[11px] uppercase tracking-wider text-muted mb-1">State snapshot → slow model</div>
          <pre className="bg-bg rounded-lg border border-border p-3 text-[11px] leading-snug
                          text-ink/90 font-mono whitespace-pre-wrap overflow-auto max-h-52 mb-3">
            {rdata.user_prompt ?? "—"}
          </pre>

          <div className="text-[11px] uppercase tracking-wider text-muted mb-1">Slow model emission (post-thinking)</div>
          <pre className="bg-bg rounded-lg border border-accent/30 p-3 text-[11px] leading-snug
                          text-accent/90 font-mono whitespace-pre-wrap overflow-auto max-h-52">
            {rdata.slow_emission ?? "—"}
          </pre>
        </div>
      </div>
    </div>
  );
}

function Panel({ label, color, score, highlight, children }: {
  label: string; color: string; score: number; highlight?: boolean; children: React.ReactNode;
}) {
  return (
    <div className={`rounded-lg border p-2 ${highlight ? "border-accent/40 bg-accent/5" : "border-border bg-panel-2"}`}>
      <div className="flex items-center justify-between px-1 pb-1.5">
        <span className={`text-xs font-semibold ${color}`}>{label}</span>
        <span className="text-xs font-mono tabular-nums text-ink">{Number.isInteger(score) ? score : score.toFixed(1)}</span>
      </div>
      {children}
    </div>
  );
}

function ScoreBox({ label, v, highlight }: { label: string; v: number; highlight?: boolean }) {
  return (
    <div className="bg-panel-2 rounded-md py-1.5 px-2 border border-border">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`text-sm leading-tight ${highlight ? "text-accent font-semibold" : "text-ink"}`}>
        {Number.isInteger(v) ? v : v.toFixed(1)}
      </div>
    </div>
  );
}
