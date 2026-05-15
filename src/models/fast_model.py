"""Fast model: MiniCPM-o 4.5 (9B) with LoRA + cross-attention bridge layers.

Frozen base, trainable LoRA on Qwen3-8B backbone attention, and new cross-attention
layers (at depths 12 and 24) that attend over the thought-vector ring buffer.

This is a skeleton. The model loading uses transformers' AutoModel; the cross-attn
injection is custom — implemented in `inject_bridge_xattn()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
from torch import nn


@dataclass
class FastModelConfig:
    hf_repo: str = "openbmb/MiniCPM-o-4_5"
    dtype: str = "bfloat16"
    device: str = "cuda"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    xattn_insert_layers: list = field(default_factory=lambda: [12, 24])
    bridge_dim: int = 256
    n_actions: int = 18  # ALE max action space size


class BridgeCrossAttention(nn.Module):
    """Cross-attention layer that reads the thought-vector ring buffer.

    Initialised so output projection is zero — model behaves identically to base at init.
    Bridge contribution emerges through training.
    """
    def __init__(self, hidden_dim: int, bridge_dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(bridge_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(bridge_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Zero-init the output projection so the layer is a no-op at start of training
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, hidden: torch.Tensor, thought_buffer: torch.Tensor,
                age_encoding: torch.Tensor | None = None) -> torch.Tensor:
        """
        hidden: [B, T, D_h]
        thought_buffer: [B, K, D_bridge]
        age_encoding: [B, K, D_bridge] additive positional encoding for entry age
        """
        if age_encoding is not None:
            thought_buffer = thought_buffer + age_encoding
        B, T, D = hidden.shape
        K = thought_buffer.size(1)
        q = self.q_proj(hidden).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(thought_buffer).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(thought_buffer).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


class FastModel(nn.Module):
    """Wrapper around MiniCPM-o 4.5 with bridge-cross-attn injection.

    NOTE: this is a skeleton. The actual injection requires hooking into the
    HF model's forward pass. Implemented in Week 1.
    """
    def __init__(self, cfg: FastModelConfig):
        super().__init__()
        self.cfg = cfg
        # Model loaded lazily via `load_pretrained()` to avoid huge cold-start cost
        # during unit tests
        self.model = None
        self.xattn_layers: dict[int, BridgeCrossAttention] = {}
        self.action_head = nn.Linear(cfg.bridge_dim, cfg.n_actions)  # placeholder

    def load_pretrained(self):
        """Load MiniCPM-o 4.5 from HF, freeze, attach LoRA + cross-attn layers."""
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            self.cfg.hf_repo,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(self.cfg.device)
        # Freeze base
        for p in self.model.parameters():
            p.requires_grad = False
        # TODO: inject LoRA via peft, inject xattn layers at specified depths
        return self

    def forward(self, frames: torch.Tensor, thought_buffer: torch.Tensor) -> torch.Tensor:
        """frames: [B, T_v, C, H, W]; thought_buffer: [B, K, D_bridge]
        Returns action logits: [B, n_actions]
        """
        raise NotImplementedError("Wired up in Week 1.")
