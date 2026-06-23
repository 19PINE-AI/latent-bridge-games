import Header from "./components/Header";
import Hero from "./components/Hero";
import BestAchievableSection from "./components/BestAchievableSection";
import ResultsTable from "./components/ResultsTable";
import ScoresChart from "./components/ScoresChart";
import BandwidthChart from "./components/BandwidthChart";
import ArchitectureSection from "./components/ArchitectureSection";
import DiagnosisSection from "./components/DiagnosisSection";
import ContinuousVsCategorical from "./components/ContinuousVsCategorical";
import PredictorSection from "./components/PredictorSection";
import BridgeReplaceChart from "./components/BridgeReplaceChart";
import ReplayTheater from "./components/ReplayTheater";
import PromptLibrary from "./components/PromptLibrary";
import ReproSection from "./components/ReproSection";
import StrategiesTable from "./components/StrategiesTable";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <Hero />

        <Section id="replay"
                 title="Watch it play: fast vs text vs latent"
                 subtitle="Every game, three ways side by side — F (reactive only), T (text bridge), and L (latent bridge) — with the exact state snapshot and the slow model's reasoning shown alongside. Press “Replay all three” to re-run the contrast.">
          <ReplayTheater />
        </Section>

        <Section id="best-achievable"
                 title="The headline: tuned per channel, the latent bridge wins or ties"
                 subtitle="The action decoder is a deployment hyperparameter, so each channel gets its own best decoder, selected on held-out seeds. The latent bridge then significantly beats the text bridge on 2 of 7 games (MsPacman +57%, RoadRunner +28%), ties the other five, and never loses — and feeding both channels at once interferes, so couple via exactly one.">
          <BestAchievableSection />
        </Section>

        <Section id="predictor"
                 title="When is a latent bridge worth it?"
                 subtitle="Across 7 Atari games and a driving simulator (MetaDrive), the latent bridge helps iff slow reasoning beats fast reaction on the task (T > F) — Pearson r = 0.93 (0.96 over all 16 game/variant cells). MetaDrive is a controlled negative.">
          <PredictorSection />
        </Section>

        <Section id="bridge-replace"
                 title="Is the latent real? The bridge-replacement control"
                 subtitle="Replacing the trained latent with zeros or random vectors (matched norm) on every game. The trained latent carries genuine learned content exactly where slow reasoning helps (T > F), and is inert or harmful where it does not.">
          <BridgeReplaceChart />
        </Section>

        <Section id="results"
                 title="Full results across all game/variant cells"
                 subtitle="The per-cell, per-variant scores under greedy decoding, with 95% bootstrap CIs and Welch's-t / Mann-Whitney significance. (The tuned-decoder headline verdict is in the Best-achievable section above.)">
          <ResultsTable />
          <div className="grid lg:grid-cols-[1.4fr_1fr] gap-6 mt-6">
            <ScoresChart />
            <BandwidthChart />
          </div>
        </Section>

        <Section id="architecture"
                 title="Architecture"
                 subtitle="The full fast/slow runtime loop and the two bridge designs — 33 M trainable params; both base models frozen.">
          <ArchitectureSection />
        </Section>

        <Section id="diagnosis"
                 title="Why some cells collapse: action-head OOD-brittleness"
                 subtitle="The methodology behind the headline — why a few cells collapse, why greedy decoding flips low-variance cells, and why “bare vs robust” Stage A is a tuned hyperparameter, not cherry-picking.">
          <DiagnosisSection />
        </Section>

        <Section id="continuous-vs-categorical"
                 title="A predictor that fails: lexical diversity"
                 subtitle="We hoped lexical diversity of slow-model emissions would predict the sign of L − T a priori. It does not (Pearson r ≈ −0.08) — reported as an honest negative; the behavioural T > F predictor above is what holds.">
          <ContinuousVsCategorical />
        </Section>

        <Section id="strategies"
                 title="Four-strategy comparison & latency"
                 subtitle="The full S < F < T < L story plus the vision-token cache that hits the 15 Hz target.">
          <StrategiesTable />
        </Section>

        <Section id="prompts"
                 title="Every prompt in the research"
                 subtitle="The verbatim system prompts and, for each game, the live state snapshot the slow model reads plus the real strategic emission it returns. This is exactly the content the bridge carries.">
          <PromptLibrary />
        </Section>

        <Section id="reproducibility"
                 title="Reproducibility &amp; implementation"
                 subtitle="Stage A/C hyperparameters, prompt templates, fairness analysis for the text baseline.">
          <ReproSection />
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
