import { Code2, FileText, PlayCircle, Activity, Cpu, Scale, MessageSquareText } from "lucide-react";

const NAV = [
  { id: "replay", label: "Replay", icon: PlayCircle },
  { id: "predictor", label: "Predictor", icon: Scale },
  { id: "results", label: "Results", icon: Activity },
  { id: "prompts", label: "Prompts", icon: MessageSquareText },
  { id: "diagnosis", label: "Diagnosis", icon: FileText },
  { id: "architecture", label: "Architecture", icon: Cpu },
];

export default function Header() {
  return (
    <header className="sticky top-0 z-40 bg-bg/85 backdrop-blur-md border-b border-border">
      <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between gap-6">
        <a href="#hero" className="flex items-center gap-2 group">
          <div className="w-7 h-7 rounded-md bg-gradient-to-br from-accent to-good"
               aria-hidden />
          <span className="font-semibold tracking-tight group-hover:text-accent transition">
            Latent Bridge
          </span>
        </a>
        <nav className="hidden md:flex items-center gap-6 text-sm">
          {NAV.map(({ id, label, icon: Icon }) => (
            <a key={id} href={`#${id}`}
               className="text-muted hover:text-ink flex items-center gap-1.5 transition">
              <Icon size={14} />
              {label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-3 text-sm">
          <a href="paper/main.pdf" target="_blank" rel="noreferrer"
             className="text-muted hover:text-ink flex items-center gap-1.5 transition">
            <FileText size={14} /> Paper
          </a>
          <a href="https://github.com/bojieli/latent-bridge-games"
             target="_blank" rel="noreferrer"
             className="text-muted hover:text-ink flex items-center gap-1.5 transition">
            <Code2 size={14} /> GitHub
          </a>
        </div>
      </div>
    </header>
  );
}
