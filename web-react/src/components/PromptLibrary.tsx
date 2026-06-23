import { useState } from "react";
import { Copy, Check, ChevronDown } from "lucide-react";
import research from "../data/research.json";

const RESEARCH: any = research;
const SYSTEMS: Record<string, string> = RESEARCH.systems;
const GAMES: Record<string, any> = RESEARCH.games;

const GAME_ORDER = ["MsPacman", "RoadRunner", "Seaquest", "Riverraid", "Qbert",
                    "SpaceInvaders", "Enduro", "MetaDrive"];
const GAME_LABEL: Record<string, string> = {
  MsPacman: "Ms. Pac-Man", RoadRunner: "Road Runner", Seaquest: "Seaquest",
  Riverraid: "River Raid", Qbert: "Q*bert", SpaceInvaders: "Space Invaders",
  Enduro: "Enduro", MetaDrive: "MetaDrive (driving)",
};

function CopyBtn({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button onClick={() => { navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1200); }}
      aria-label={done ? "Copied to clipboard" : "Copy prompt to clipboard"}
      className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded
                 bg-panel-2 border border-border text-muted hover:text-ink transition">
      {done ? <Check size={12} className="text-good" aria-hidden /> : <Copy size={12} aria-hidden />}
      {done ? "copied" : "copy"}
    </button>
  );
}

export default function PromptLibrary() {
  const [tab, setTab] = useState<"system" | "game">("system");
  const [openSys, setOpenSys] = useState<string | null>("Atari (default)");
  const [game, setGame] = useState("MsPacman");

  return (
    <div>
      <div className="flex gap-2 mb-5">
        <Tab active={tab === "system"} onClick={() => setTab("system")}>System prompts ({Object.keys(SYSTEMS).length})</Tab>
        <Tab active={tab === "game"} onClick={() => setTab("game")}>Per-game state &rarr; emission ({GAME_ORDER.length})</Tab>
      </div>

      {tab === "system" ? (
        <div className="space-y-3">
          <p className="text-sm text-muted">
            The verbatim system prompts that define the slow model's role. The fast (reactive)
            model gets no system prompt — it only sees pixels and, for T/L, the slow channel.
          </p>
          {Object.entries(SYSTEMS).map(([name, body]) => (
            <div key={name} className="bg-panel rounded-2xl border border-border overflow-hidden">
              <button onClick={() => setOpenSys(openSys === name ? null : name)}
                className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-panel-2 transition">
                <span className="font-semibold text-ink">{name}</span>
                <div className="flex items-center gap-2">
                  <CopyBtn text={body} />
                  <ChevronDown size={16}
                    className={`text-muted transition ${openSys === name ? "rotate-180" : ""}`} />
                </div>
              </button>
              {openSys === name && (
                <pre className="px-4 pb-4 text-[12px] leading-relaxed text-ink/90 font-mono
                                whitespace-pre-wrap border-t border-border pt-3">{body}</pre>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div>
          <div className="flex flex-wrap gap-2 mb-4">
            {GAME_ORDER.map(gk => (
              <button key={gk} onClick={() => setGame(gk)}
                className={`px-3 py-1.5 rounded-lg text-sm border transition ${
                  game === gk ? "bg-accent text-bg border-accent font-semibold"
                              : "bg-panel border-border text-muted hover:border-accent/40"}`}>
                {GAME_LABEL[gk]}
              </button>
            ))}
          </div>
          <div className="grid lg:grid-cols-2 gap-4">
            <div className="bg-panel rounded-2xl border border-border p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-ink">State snapshot (user prompt)</h4>
                <CopyBtn text={GAMES[game]?.user_prompt ?? ""} />
              </div>
              <p className="text-[11px] text-muted mb-2">
                Decoded from the live environment each ~1 s and handed to the slow model.
              </p>
              <pre className="bg-bg rounded-lg border border-border p-3 text-[11px] leading-snug
                              text-ink/90 font-mono whitespace-pre-wrap overflow-auto max-h-[28rem]">
                {GAMES[game]?.user_prompt ?? "—"}
              </pre>
            </div>
            <div className="bg-panel rounded-2xl border border-border p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-ink">Real slow-model emission</h4>
                <CopyBtn text={GAMES[game]?.slow_emission ?? ""} />
              </div>
              <p className="text-[11px] text-muted mb-2">
                Actual post-thinking output (seed 0). <span className="text-ink">T</span> sends this
                text; <span className="text-accent">L</span> sends its projected latent.
              </p>
              <pre className="bg-bg rounded-lg border border-accent/30 p-3 text-[11px] leading-snug
                              text-accent/90 font-mono whitespace-pre-wrap overflow-auto max-h-[28rem]">
                {GAMES[game]?.slow_emission ?? "—"}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Tab({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick}
      className={`px-4 py-2 rounded-lg text-sm font-medium border transition ${
        active ? "bg-panel-2 border-accent/50 text-ink" : "bg-panel border-border text-muted hover:text-ink"}`}>
      {children}
    </button>
  );
}
