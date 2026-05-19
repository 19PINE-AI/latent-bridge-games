import { Code2, FileText, ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-border bg-panel/40 mt-16">
      <div className="max-w-6xl mx-auto px-6 py-10 grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
        <div>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-accent to-good" />
            <span className="font-semibold">Latent Bridge</span>
          </div>
          <p className="text-xs text-muted leading-relaxed">
            Fast/slow model coupling for real-time agents, demonstrated on Atari.
            Built on MiniCPM-o 4.5 + Qwen3-VL-8B-Thinking + 33 M trainable bridge params.
          </p>
        </div>
        <Col title="Resources">
          <FootLink href="https://github.com/bojieli/latent-bridge-games"
                    icon={<Code2 size={12} />}>GitHub repo</FootLink>
          <FootLink href="paper/main.pdf" icon={<FileText size={12} />}>Paper (PDF, 9 pages)</FootLink>
          <FootLink href="https://github.com/bojieli/latent-bridge-games/blob/main/docs/06_results.md"
                    icon={<ExternalLink size={12} />}>Full results doc</FootLink>
        </Col>
        <Col title="Reproduction">
          <FootLink href="https://github.com/bojieli/latent-bridge-games/blob/main/README.md">README</FootLink>
          <FootLink href="https://github.com/bojieli/latent-bridge-games/blob/main/docs/04_architecture.md">Architecture spec</FootLink>
          <FootLink href="https://github.com/bojieli/latent-bridge-games/tree/main/scripts">Pipeline scripts</FootLink>
        </Col>
        <Col title="Highlights">
          <li className="text-xs text-muted">9 games, 60+ commits</li>
          <li className="text-xs text-muted">Largest gap: <span className="text-good">+82 %</span> (RR robust)</li>
          <li className="text-xs text-muted">Single RTX Pro 6000 96 GB</li>
        </Col>
      </div>
      <div className="border-t border-border/60 py-4">
        <div className="max-w-6xl mx-auto px-6 text-xs text-muted flex items-center justify-between">
          <span>Hardware: RTX Pro 6000 96 GB · ALE 0.11.2 · Python 3.10</span>
          <span>Generated with the project's <code className="text-[11px] mx-1">render_demo_mp4.py</code> + Vite + Tailwind.</span>
        </div>
      </div>
    </footer>
  );
}

function Col({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="text-xs uppercase tracking-wider text-muted mb-3">{title}</h4>
      <ul className="space-y-2">{children}</ul>
    </div>
  );
}

function FootLink({ href, icon, children }: {
  href: string; icon?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <li>
      <a href={href} target={href.startsWith("http") || href.endsWith(".pdf") ? "_blank" : undefined}
         rel="noreferrer"
         className="text-xs text-muted hover:text-ink flex items-center gap-1.5 transition">
        {icon} {children}
      </a>
    </li>
  );
}
