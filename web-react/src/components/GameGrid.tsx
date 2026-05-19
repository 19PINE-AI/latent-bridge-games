import { GAMES, type GameResult } from "../data/games";

export default function GameGrid() {
  const withVideo = GAMES.filter(g => g.videoSideBySide);
  return (
    <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-5">
      {withVideo.map(g => <GameCard key={g.id} game={g} />)}
    </div>
  );
}

function GameCard({ game }: { game: GameResult }) {
  const gap = game.L_vs_T_pct;
  return (
    <article className="bg-panel rounded-xl border border-border overflow-hidden
                        hover:border-accent/40 transition">
      <div className="aspect-[16/6.24] bg-black">
        <video src={game.videoSideBySide!} controls playsInline
               preload="metadata" className="w-full h-full" />
      </div>
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <h3 className="font-semibold text-ink leading-tight">{game.name}</h3>
          {gap != null && (
            <span className={`text-xs font-mono px-2 py-0.5 rounded ${
              gap > 0 ? "bg-good/15 text-good" : gap < 0 ? "bg-bad/15 text-bad"
                                                          : "bg-muted/15 text-muted"
            }`}>
              {gap > 0 ? `+${gap}% L>T` : gap < 0 ? `${Math.abs(gap)}% T>L` : "L=T"}
            </span>
          )}
        </div>
        <div className="mt-2 grid grid-cols-3 gap-2 text-center font-mono tabular-nums">
          <Score label="F" v={game.F} />
          <Score label="T" v={game.T} />
          <Score label="L" v={game.L} highlight />
        </div>
        <p className="mt-3 text-xs text-muted leading-relaxed">{game.notes}</p>
      </div>
    </article>
  );
}

function Score({ label, v, highlight }: { label: string; v: number; highlight?: boolean }) {
  return (
    <div className="bg-panel-2 rounded-md py-1.5 px-2 border border-border">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`text-sm leading-tight ${highlight ? "text-accent font-semibold" : "text-ink"}`}>
        {Number.isInteger(v) ? v : v.toFixed(1)}
      </div>
    </div>
  );
}
