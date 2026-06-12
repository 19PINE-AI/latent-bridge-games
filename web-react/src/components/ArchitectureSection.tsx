export default function ArchitectureSection() {
  return (
    <div className="grid lg:grid-cols-2 gap-6">
      <div className="bg-panel rounded-2xl border border-border p-6">
        <h3 className="font-semibold text-ink mb-3">v2 LLaVA-style latent bridge</h3>
        <pre className="bg-bg/60 rounded-lg p-4 border border-border overflow-x-auto
                        text-xs leading-relaxed text-ink">
{`slow model (Qwen3-VL-8B-Thinking, 1 Hz)
   |--> residuals @ layer 24, last N = 8 positions (4096-d)
        |--> ThoughtProjection MLP: 4096 → 4096 → 4096
        |    LayerNorm bookends · 33.6 M trainable params
        |--> N = 8 latent tokens in fast model's embedding space
             |--> PREPENDED to fast model's input sequence
                  |--> fast model (MiniCPM-o 4.5, 15 Hz)
                       all 36 LLM layers attend over them
                       |--> action_head on last hidden state
                            |--> action logits (18-way)`}
        </pre>
        <ul className="mt-4 text-sm text-ink/90 space-y-1.5 list-disc pl-5">
          <li><strong>Trainable</strong>: only the slow model's ThoughtProjection (~33 M params)</li>
          <li><strong>Frozen</strong>: both base models entirely (no LoRA on backbones)</li>
          <li><strong>Loss</strong>: KL(student ‖ teacher) where the teacher is T-runtime,
              student is L-runtime</li>
        </ul>
      </div>
      <div className="bg-panel rounded-2xl border border-border p-6">
        <h3 className="font-semibold text-ink mb-3">Why v1 (cross-attention) failed</h3>
        <p className="text-sm text-ink/85 leading-relaxed">
          v1 used a 256-d ring buffer with cross-attention at LLM layers 12 and 24 only.
          KL converged to 0.004 on offline data but <strong>L = 225 &lt; F = 256</strong>{" "}
          at deployment, bimodal with 4/12 catastrophic episodes.
        </p>
        <p className="mt-3 text-sm text-ink/85 leading-relaxed">
          Two architectural privileges fixed it:
        </p>
        <ul className="mt-2 text-sm text-ink/90 space-y-1.5 list-disc pl-5">
          <li>Bridge tokens live in the LLM's native input-embedding space
              (4096-d, not 256-d) — inherits the LLM's pre-trained inductive bias</li>
          <li>All 36 layers attend through standard causal attention (not just 2 of 36)</li>
        </ul>
        <p className="mt-3 text-sm text-muted">
          These are exactly the privileges text tokens have. Latent tokens needed them too.
        </p>
      </div>
    </div>
  );
}
