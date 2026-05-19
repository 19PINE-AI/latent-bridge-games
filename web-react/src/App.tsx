import Header from "./components/Header";
import Hero from "./components/Hero";
import ResultsTable from "./components/ResultsTable";
import ScoresChart from "./components/ScoresChart";
import BandwidthChart from "./components/BandwidthChart";
import GameGrid from "./components/GameGrid";
import ArchitectureSection from "./components/ArchitectureSection";
import DiagnosisSection from "./components/DiagnosisSection";
import StrategiesTable from "./components/StrategiesTable";
import FiguresGallery from "./components/FiguresGallery";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <Hero />

        <Section id="results"
                 title="Results across 9 game-conditions"
                 subtitle="Sortable table + chart. Click column headers to re-sort.">
          <ResultsTable />
          <div className="grid lg:grid-cols-[1.4fr_1fr] gap-6 mt-6">
            <ScoresChart />
            <BandwidthChart />
          </div>
        </Section>

        <Section id="diagnosis"
                 title="The Stage A OOD-brittleness diagnosis"
                 subtitle="One sub-finding that turned negative results into evidence and revealed a categorical exception.">
          <DiagnosisSection />
        </Section>

        <Section id="strategies"
                 title="Four-strategy comparison & latency"
                 subtitle="The full S < F < T < L story plus the vision-token cache that hits 15 Hz.">
          <StrategiesTable />
        </Section>

        <Section id="architecture"
                 title="Architecture"
                 subtitle="33 M trainable params; both base models frozen.">
          <ArchitectureSection />
        </Section>

        <Section id="figures"
                 title="Paper figures"
                 subtitle="From the 9-page arXiv-style PDF in paper/main.pdf. Click any to enlarge.">
          <FiguresGallery />
        </Section>

        <Section id="games"
                 title="All game playthroughs"
                 subtitle="Side-by-side F vs L for each game. The slow-model text overlay shows you the reasoning the bridge is transmitting.">
          <GameGrid />
        </Section>
      </main>
      <Footer />
    </div>
  );
}

function Section({ id, title, subtitle, children }: {
  id: string; title: string; subtitle?: string; children: React.ReactNode;
}) {
  return (
    <section id={id} className="border-b border-border/60">
      <div className="max-w-6xl mx-auto px-6 py-14">
        <div className="mb-6">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{title}</h2>
          {subtitle && <p className="text-muted mt-1.5 max-w-3xl">{subtitle}</p>}
        </div>
        {children}
      </div>
    </section>
  );
}
