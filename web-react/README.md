# Latent Bridge — React static site

Professional Vite + React + TypeScript + Tailwind + Recharts build of the
demo site. This is the canonical website (it superseded an earlier minimal
static page).

## What it includes

- **Hero** with embedded combined narrated 4-min video and 3 headline stats
- **Replay theater** — per-game F/T/L side-by-side playthroughs
- **Best-achievable result** (decoder tuned per channel) and the **behavioral predictor** (L−F vs T−F)
- **Bridge-replacement control** and **continuous-vs-categorical** charts
- **Sortable results table** across all game-conditions (click headers)
- **Interactive Recharts**: per-game F/T/L scores plus the latent-token-count and latency ablations
- **Stage A OOD diagnosis** section explaining the methodology finding
- **4-strategy comparison** table (S < F < T < L) plus the vision-cache latency sweep
- **Architecture** with whole-system + v1-vs-v2 diagrams, and a **prompt library**
- Smooth-scroll anchor nav · dark theme · mobile-responsive

## Build

```bash
cd web-react
npm install
npm run build
```

Output lands in `../web-dist/`. Static media folders (`demos/`, `paper/`,
`figs/`) are symlinked from `public/` so they appear at the correct URLs in
both dev and prod builds.

## Serve

```bash
# Built site
cd web-dist && python3 -m http.server 8000

# Or via the project's Flask + SSE server (serves web-dist/; build it first)
python3 scripts/live_demo_server.py --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/>.

## Dev

```bash
cd web-react
npm run dev   # http://localhost:5173
```

## Stack

- Vite 7 + React 19 + TypeScript 5
- Tailwind CSS 3 (custom palette in `tailwind.config.js`)
- Recharts 3 + lucide-react (icons)

## Layout

```
web-react/
├── public/
│   ├── demos/   -> ../../demos    (symlink)
│   ├── figs/    -> ../../figs
│   └── paper/   -> ../../paper
├── src/
│   ├── App.tsx                 # section composition
│   ├── data/games.ts           # all game-condition results
│   ├── data/strategies.ts      # 4-strategy + bandwidth + latency rows
│   └── components/
│       ├── Header.tsx · Hero.tsx · Footer.tsx
│       ├── ReplayTheater.tsx · BestAchievableSection.tsx · PredictorSection.tsx
│       ├── BridgeReplaceChart.tsx · ContinuousVsCategorical.tsx
│       ├── ResultsTable.tsx · ScoresChart.tsx · BandwidthChart.tsx  (Recharts)
│       ├── DiagnosisSection.tsx · ArchitectureSection.tsx · SystemDiagram.tsx
│       ├── StrategiesTable.tsx · PromptLibrary.tsx · ReproSection.tsx · ArchDiagram.tsx
└── vite.config.ts              # base="./", outDir="../web-dist"
```
