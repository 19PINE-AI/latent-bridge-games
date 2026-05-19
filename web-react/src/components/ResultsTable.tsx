import { useMemo, useState } from "react";
import { ArrowDownAZ, ArrowUpAZ, ArrowDown01, ArrowUp01, Info } from "lucide-react";
import { GAMES, type GameResult } from "../data/games";

type SortKey = "name" | "F" | "T" | "L" | "L_vs_T_pct" | "cohensD";
type SortDir = "asc" | "desc";

const CAT_BADGE: Record<GameResult["category"], { label: string; cls: string }> = {
  "win-L":    { label: "L > T",             cls: "bg-good/15 text-good border-good/40" },
  "win-T":    { label: "T > L",             cls: "bg-bad/15 text-bad border-bad/40" },
  "collapse": { label: "Collapse",          cls: "bg-bad/15 text-bad border-bad/40" },
  "partial":  { label: "Partial recovery",  cls: "bg-accent/15 text-accent border-accent/40" },
  "floor":    { label: "Reactive floor",    cls: "bg-muted/15 text-muted border-border" },
};

function pSymbol(p: number | undefined): { text: string; cls: string } {
  if (p == null) return { text: "—", cls: "text-muted" };
  if (p < 0.001) return { text: "***", cls: "text-good" };
  if (p < 0.01)  return { text: "**",  cls: "text-good" };
  if (p < 0.05)  return { text: "*",   cls: "text-good" };
  return { text: "n.s.", cls: "text-muted" };
}

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
            <Th label="Game"        onClick={() => toggle("name")}      active={sortKey === "name"} dir={sortDir} numeric={false} />
            <Th label="F (fast)"    onClick={() => toggle("F")}         active={sortKey === "F"} dir={sortDir} numeric />
            <Th label="T (text)"    onClick={() => toggle("T")}         active={sortKey === "T"} dir={sortDir} numeric />
            <Th label="L (latent)"  onClick={() => toggle("L")}         active={sortKey === "L"} dir={sortDir} numeric highlight />
            <Th label="ΔL−T (%, p)" onClick={() => toggle("L_vs_T_pct")} active={sortKey === "L_vs_T_pct"} dir={sortDir} numeric highlight />
            <Th label="Cohen's d"   onClick={() => toggle("cohensD")}    active={sortKey === "cohensD"} dir={sortDir} numeric />
            <th className="px-4 py-2 text-left">Outcome</th>
            <th className="px-4 py-2 text-left font-normal">Note</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(g => {
            const badge = CAT_BADGE[g.category];
            const sym = pSymbol(g.pvalue);
            return (
              <tr key={g.id} className="border-t border-border/60 hover:bg-panel-2/50 transition">
                <td className="px-4 py-3">
                  <div className="font-medium text-ink">{g.name}</div>
                  <div className="text-xs text-muted">Tier {g.tier} · {g.variant}</div>
                </td>
                <CellWithCI mean={g.F} std={g.F_std} ci={g.F_ci} />
                <CellWithCI mean={g.T} std={g.T_std} ci={g.T_ci} />
                <CellWithCI mean={g.L} std={g.L_std} ci={g.L_ci} bold />
                <td className="px-4 py-3 text-right font-mono tabular-nums whitespace-nowrap">
                  {g.L_vs_T_pct == null ? (
                    <span className="text-muted">—</span>
                  ) : (
                    <span className="flex items-center justify-end gap-1.5">
                      <span className={g.L_vs_T_pct > 0 ? "text-good font-semibold"
                                      : g.L_vs_T_pct < 0 ? "text-bad" : "text-muted"}>
                        {g.L_vs_T_pct > 0 ? "+" : ""}{g.L_vs_T_pct}%
                      </span>
                      <span className={`text-[10px] ${sym.cls}`}
                            title={g.pvalue == null ? undefined :
                              `${g.pvalueIsMWU ? "Mann-Whitney U" : "Welch's t"} p ≈ ${
                                g.pvalue < 0.001 ? g.pvalue.toExponential(1) : g.pvalue.toFixed(3)
                              }${g.pvalueIsMWU ? " (zero-variance fallback)" : ""}`
                            }>
                        {sym.text}{g.pvalueIsMWU ? "†" : ""}
                      </span>
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums">
                  {g.cohensD == null ? <span className="text-muted">—</span>
                    : <span className={Math.abs(g.cohensD) >= 0.8 ? "text-ink font-semibold"
                                       : "text-muted"}>{g.cohensD.toFixed(2)}</span>}
                </td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-1 rounded border font-medium ${badge.cls}`}>
                    {badge.label}
                  </span>
                  {g.smallScores && (
                    <div className="text-[10px] text-muted mt-1 flex items-center gap-1">
                      <Info size={10} /> small absolute scores
                    </div>
                  )}
                </td>
                <td className="px-4 py-3 text-xs text-muted max-w-sm leading-snug">{g.notes}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="px-4 py-3 text-[11px] text-muted border-t border-border/60 bg-panel-2/50">
        n = 12 episodes per cell (3 seeds × 4 episodes). Cell means show ± std;
        hover over a cell for the 95 % bootstrap CI. Significance: <span className="text-good font-semibold">*</span> p&nbsp;&lt;&nbsp;0.05,
        <span className="text-good font-semibold"> **</span> p&nbsp;&lt;&nbsp;0.01,
        <span className="text-good font-semibold"> ***</span> p&nbsp;&lt;&nbsp;0.001.
        † = Mann-Whitney U (used when one or both cells have zero variance, where Welch's t is undefined).
      </div>
    </div>
  );
}

function CellWithCI({ mean, std, ci, bold }: {
  mean: number; std: number; ci?: [number, number]; bold?: boolean;
}) {
  const title = ci ? `95% CI: [${fmt(ci[0])}, ${fmt(ci[1])}]` : undefined;
  return (
    <td className="px-4 py-3 text-right font-mono tabular-nums cursor-help" title={title}>
      <span className={bold ? "font-semibold text-ink" : ""}>{fmt(mean)}</span>
      <span className="text-muted"> ±{fmt(std)}</span>
    </td>
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
