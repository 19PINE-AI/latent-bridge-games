import { Layers, Settings2, ListChecks, Scale } from "lucide-react";

const STAGE_A_VAL_ACC = [
  { game: "MsPacman",     acc: 32, random: 11, ratio: "2.9×" },
  { game: "Seaquest",     acc: 24, random: 5.6, ratio: "4.3×" },
  { game: "SpaceInvaders", acc: 33, random: 17, ratio: "2.0×" },
  { game: "Pong",         acc: 25, random: 17, ratio: "1.5×" },
  { game: "RoadRunner",   acc: 59, random: 5.6, ratio: "10.5×" },
];

export default function ReproSection() {
  return (
    <div className="space-y-6">
      <div className="grid lg:grid-cols-2 gap-6">
        <Card icon={<Layers size={16} />} title="Stage C v2 (latent bridge training)">
          <KV k="Projection" v="2-layer MLP, LN → 4096 → GELU → 4096 → LN" />
          <KV k="Parameters" v="33.6 M trainable; both base models frozen" />
          <KV k="Optimizer" v="AdamW, lr 5·10⁻⁵, 1 epoch" />
          <KV k="Batch" v="size 1 with gradient accumulation 4" />
          <KV k="Loss" v="KL(π_L ‖ π_T), temperature τ = 1.0" />
          <KV k="Slow residuals" v="layer 24 of 36 (~67 % depth), last N = 8 positions" />
          <KV k="Training samples" v="~5 K (frame, slow text, slow residuals) per game" />
          <KV k="Final KL" v="~0.005 (converged across games)" />
        </Card>

        <Card icon={<Settings2 size={16} />} title="Stage A (behavioral cloning)">
          <KV k="Optimizer" v="AdamW, lr 1·10⁻⁴, 3 epochs" />
          <KV k="Batch" v="size 1 with gradient accumulation 8" />
          <KV k="Teacher" v="SB3-zoo expert trajectories (DQN/PPO)" />
          <KV k="Robust variant" v="half of training batches receive a synthetic slow-style suffix prepended to the prompt (suffix-probability 0.5)" />
          <div className="mt-4">
            <div className="text-[11px] uppercase tracking-wider text-muted mb-2">
              Val accuracy on bare game-state prompt
            </div>
            <table className="w-full text-xs">
              <thead className="text-muted">
                <tr>
                  <th className="text-left pb-1">Game</th>
                  <th className="text-right pb-1">acc</th>
                  <th className="text-right pb-1">random</th>
                  <th className="text-right pb-1">× random</th>
                </tr>
              </thead>
              <tbody>
                {STAGE_A_VAL_ACC.map(r => (
                  <tr key={r.game} className="border-t border-border/60">
                    <td className="py-1.5 text-ink">{r.game}</td>
                    <td className="py-1.5 text-right font-mono">{r.acc} %</td>
                    <td className="py-1.5 text-right font-mono text-muted">{r.random} %</td>
                    <td className={`py-1.5 text-right font-mono ${
                      r.game === "Pong" ? "text-bad font-semibold" : "text-ink"
                    }`}>{r.ratio}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-[11px] text-muted leading-snug">
              Pong's 1.5× explains the −21 floor: the head barely moves the paddle.
              Fast/slow coupling presupposes a working fast.
            </p>
          </div>
        </Card>
      </div>

      <Card icon={<ListChecks size={16} />} title="Prompts &amp; text-bridge suffix">
        <KV k="System prompt"
            v={`"You are the strategic-reasoning module... emit strategic guidance, not individual actions. 1–3 sentences identifying current threat/opportunity and recommended intent."`} />
        <KV k="User prompt (per game)"
            v="RAM-decoded structured state with explicit coordinates (e.g. positions, deltas, fuel, lives, ghost positions)." />
        <KV k="T suffix" v={`The full post-thinking slow-model emission is appended verbatim to the fast-model user prompt under a "strategic-guidance" tag. No truncation.`} />
        <KV k="Median emission length" v="302 characters (game medians 293–335)" />
      </Card>

      <Card icon={<Scale size={16} />} title="Fairness of the text baseline">
        <p className="text-sm text-ink/85 leading-relaxed mb-3">
          Common objection: <em>did we give T enough text to compete?</em> Two
          observations bound the answer.
        </p>
        <ol className="list-decimal pl-5 space-y-2 text-sm text-ink/85">
          <li>
            T uses the slow model's full unmodified emission, with no truncation.
            Median lengths are remarkably uniform across games (293–335 chars), so
            <strong> character budget is not the binding variable</strong>.
          </li>
          <li>
            We <em>did</em> run the longer-suffix sweep: feeding T a rolling window of the
            last 2–3 emissions does <strong>not</strong> close the gap — it widens it.
            MsPacman declines 14 % (w=1→3); RoadRunner collapses to <strong>0 on every
            episode</strong> at w=3 (stale older emissions mislead the frozen head). The
            latent sidesteps this because it carries only the most-recent emission's residuals.
          </li>
        </ol>
        <p className="text-xs text-muted mt-3 leading-relaxed">
          And the earlier lexical-entropy hypothesis (that T beats L on Q*bert because its
          emissions are lower-entropy) did not survive: under sampling Q*bert <em>inverts</em> to
          a 2.9× latent win, so the greedy T-win was a decoder artifact, not a property of the
          emission distribution.
        </p>
      </Card>
    </div>
  );
}

function Card({ icon, title, children }: {
  icon: React.ReactNode; title: string; children: React.ReactNode;
}) {
  return (
    <div className="bg-panel rounded-2xl border border-border p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-accent">{icon}</span>
        <h3 className="font-semibold text-ink">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="grid grid-cols-[140px_1fr] gap-3 py-1.5 text-sm border-b border-border/40 last:border-0">
      <span className="text-muted text-xs uppercase tracking-wider pt-0.5">{k}</span>
      <span className="text-ink/90 leading-snug">{v}</span>
    </div>
  );
}
