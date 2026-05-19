import { useMemo, useState } from "react";
import { ArrowDownAZ, ArrowUpAZ, ArrowDown01, ArrowUp01 } from "lucide-react";
import { GAMES, type GameResult } from "../data/games";

type SortKey = "name" | "F" | "T" | "L" | "L_vs_T_pct";
type SortDir = "asc" | "desc";

const CAT_BADGE: Record<GameResult["category"], { label: string; cls: string }> = {
  "win-L":   { label: "L > T",    cls: "bg-good/15 text-good border-good/40" },
  "win-T":   { label: "T > L",    cls: "bg-bad/15 text-bad  border-bad/40" },
  "collapse":{ label: "Collapse", cls: "bg-bad/15 text-bad  border-bad/40" },
  "partial": { label: "Partial recovery", cls: "bg-accent/15 text-accent border-accent/40" },
  "floor":   { label: "Reactive floor",   cls: "bg-muted/15 text-muted border-border" },
};

export default function ResultsTable() {
  const [sortKey, setSortKey] = useState<SortKey>("L_vs_T_pct");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    return [...GAMES].sort((a, b) => {
      const va = (a as any)[sortKey];
      const vb = (b as any)[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      const cmp = va < vb ? -1 : va > vb ? 1 : 0;
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [sortKey, sortDir]);

  const toggle = (k: SortKey) => {
    if (k === sortKey) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortKey(k); setSortDir(k === "name" ? "asc" : "desc"); }
  };

  return (
    <div className="overflow-x-auto bg-panel rounded-2xl border border-border">
      <table className="w-full text-sm">
        <thead className="bg-panel-2 text-muted">
          <tr>
            <Th label="Game" onClick={() => toggle("name")} active={sortKey === "name"}
                dir={sortDir} numeric={false} />
            <Th label="F (fast)" onClick={() => toggle("F")} active={sortKey === "F"}
                dir={sortDir} numeric />
            <Th label="T (text)" onClick={() => toggle("T")} active={sortKey === "T"}
                dir={sortDir} numeric />
            <Th label="L (latent)" onClick={() => toggle("L")} active={sortKey === "L"}
                dir={sortDir} numeric highlight />
            <Th label="L vs T" onClick={() => toggle("L_vs_T_pct")}
                active={sortKey === "L_vs_T_pct"} dir={sortDir} numeric highlight />
            <th className="px-4 py-2 text-left">Outcome</th>
            <th className="px-4 py-2 text-left font-normal">Note</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(g => {
            const badge = CAT_BADGE[g.category];
            return (
              <tr key={g.id} className="border-t border-border/60 hover:bg-panel-2/50 transition">
                <td className="px-4 py-3">
                  <div className="font-medium text-ink">{g.name}</div>
                  <div className="text-xs text-muted">Tier {g.tier} · {g.variant}</div>
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums">
                  {fmt(g.F)} <span className="text-muted">±{fmt(g.F_std)}</span>
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums">
                  {fmt(g.T)} <span className="text-muted">±{fmt(g.T_std)}</span>
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums font-semibold">
                  {fmt(g.L)} <span className="text-muted font-normal">±{fmt(g.L_std)}</span>
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums">
                  {g.L_vs_T_pct == null ?
                    <span className="text-muted">—</span> :
                    <span className={g.L_vs_T_pct > 0 ? "text-good font-semibold" :
                                    g.L_vs_T_pct < 0 ? "text-bad" : "text-muted"}>
                      {g.L_vs_T_pct > 0 ? "+" : ""}{g.L_vs_T_pct}%
                    </span>}
                </td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-1 rounded border font-medium ${badge.cls}`}>
                    {badge.label}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-muted max-w-sm leading-snug">{g.notes}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Th({ label, onClick, active, dir, numeric, highlight }: {
  label: string; onClick: () => void; active: boolean; dir: SortDir;
  numeric: boolean; highlight?: boolean;
}) {
  const Asc = numeric ? ArrowUp01 : ArrowUpAZ;
  const Desc = numeric ? ArrowDown01 : ArrowDownAZ;
  return (
    <th className={`px-4 py-2 cursor-pointer select-none ${numeric ? "text-right" : "text-left"} ${
      highlight ? "text-ink" : ""
    }`} onClick={onClick}>
      <span className={`inline-flex items-center gap-1 ${
        numeric ? "flex-row-reverse" : ""
      } ${active ? "text-ink" : "text-muted"}`}>
        {label}
        {active && (dir === "asc" ? <Asc size={12} /> : <Desc size={12} />)}
      </span>
    </th>
  );
}

function fmt(n: number) {
  if (Number.isInteger(n)) return n.toString();
  return n.toFixed(1);
}
