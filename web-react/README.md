# Latent Bridge — React static site

Professional Vite + React + TypeScript + Tailwind + Recharts build of the
demo site. Replaces the earlier minimal `web/index.html` (which stays in the
repo as a tiny fallback).

## What it includes

- **Hero** with embedded combined narrated 4-min video and 3 headline stats
- **Sortable results table** for all 9 game-conditions (click headers)
- **Interactive Recharts bar charts**: per-game F/T/L scores and bandwidth ablation
- **Stage A OOD diagnosis** section explaining the methodology finding
- **4-strategy comparison** table (S < F < T < L) plus the vision-cache latency sweep
- **Architecture** with ASCII diagram and v1-vs-v2 explanation
- **Paper figures gallery** with click-to-enlarge lightbox
- **Per-game playthrough grid** (7 side-by-side videos)
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

# Or via the project's Flask + SSE server (which now prefers web-dist when it
# exists, falling back to the static web/index.html otherwise)
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
│       ├── ResultsTable.tsx    # sortable
│       ├── ScoresChart.tsx · BandwidthChart.tsx  (Recharts)
│       ├── DiagnosisSection.tsx · ArchitectureSection.tsx
│       ├── StrategiesTable.tsx · FiguresGallery.tsx · GameGrid.tsx
└── vite.config.ts              # base="./", outDir="../web-dist"
```
