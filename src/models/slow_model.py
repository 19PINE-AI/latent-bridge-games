"""Slow model: Qwen3-30B-A3B-Thinking (frozen) + projection head for thought vectors.

Operates at 1-2Hz asynchronously to the fast model. Emits both text reasoning tokens
(for diagnostic / text-bridge baseline) and projected thought vectors that the fast
model consumes via cross-attention.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class SlowModelConfig:
    hf_repo: str = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    dtype: str = "bfloat16"
    device: str = "cuda"
    projection_layer_idx: int = 32  # which residual stream to project
    bridge_dim: int = 256
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_proj", "v_proj")
    emission_rate_hz: float = 1.0  # how often to emit thought vectors
    max_emission_tokens: int = 128  # text tokens per emission


class ThoughtProjection(nn.Module):
    """Projects slow-model residual stream to thought vectors of bridge dim.

    Trained via COCONUT-style curriculum: latent vectors must reconstruct the
    predictive content of the text they replace.
    """
    def __init__(self, hidden_dim: int, bridge_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bridge_dim * 2),
            nn.GELU(),
            nn.Linear(bridge_dim * 2, bridge_dim),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """residual: [B, T, D_h] → [B, T, D_bridge]"""
        return self.proj(residual)


class SlowModel(nn.Module):
    """Wrapper around Qwen3-30B-A3B-Thinking with thought-projection head."""
    def __init__(self, cfg: SlowModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.projection: ThoughtProjection | None = None

    def load_pretrained(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.hf_repo, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.hf_repo,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            device_map=self.cfg.device,
        )
        for p in self.model.parameters():
            p.requires_grad = False
        hidden_dim = self.model.config.hidden_size
        self.projection = ThoughtProjection(hidden_dim, self.cfg.bridge_dim).to(
            self.cfg.device, dtype=torch.bfloat16
        )
        # TODO: inject LoRA via peft on attention layers
        return self

    def emit(self, context_text: str, perception_summary: torch.Tensor | None = None
             ) -> tuple[str, torch.Tensor]:
        """Run one emission cycle.

        Returns (text_emission, thought_vectors[L, D_bridge]) where L is the
        number of generated tokens at which we sampled the residual stream.
        """
        raise NotImplementedError("Wired up in Week 1.")
