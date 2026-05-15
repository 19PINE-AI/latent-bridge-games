"""Slow model: Qwen3-VL-8B-Thinking (frozen) + projection head for thought vectors.

Operates at 1-2Hz asynchronously to the fast model. Emits both text reasoning tokens
(for diagnostic / text-bridge baseline) and projected thought vectors that the fast
model consumes via cross-attention.

Primary configuration is Qwen3-VL-8B-Thinking — same scale as MiniCPM-o's backbone, with
native vision. A capability-scaling ablation swaps in Qwen3-30B-A3B-Thinking (MoE) on one
Tier-3 game only; see `SlowModelConfig.from_scaling_ablation()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch import nn

try:
    from PIL import Image
except ImportError:  # PIL is a transformers dependency; guarded for type-only contexts
    Image = None  # type: ignore


@dataclass
class SlowModelConfig:
    hf_repo: str = "Qwen/Qwen3-VL-8B-Thinking"
    dtype: str = "bfloat16"
    device: str = "cuda"
    projection_layer_idx: int = 24  # which residual stream to project
    bridge_dim: int = 256
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_proj", "v_proj")
    emission_rate_hz: float = 1.0  # how often to emit thought vectors
    max_emission_tokens: int = 128  # text tokens per emission
    vision_enabled: bool = True     # primary model has native vision

    @classmethod
    def from_scaling_ablation(cls) -> "SlowModelConfig":
        """30B-A3B MoE configuration for the single Tier-3 scaling ablation.

        Not used in the main four-strategy sweep. Different hidden size (5120 vs 4096)
        and no native vision encoder — slow side falls back to text+perception channels.
        """
        return cls(
            hf_repo="Qwen/Qwen3-30B-A3B-Thinking-2507",
            projection_layer_idx=32,
            vision_enabled=False,
        )


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
    """Wrapper around the slow model (Qwen3-VL-8B-Thinking by default) with thought-projection head."""
    def __init__(self, cfg: SlowModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.projection: ThoughtProjection | None = None

    def load_pretrained(self):
        """Load the slow model. Qwen3-VL-8B-Thinking ships natively in transformers
        4.57+ as `Qwen3VLForConditionalGeneration` (no trust_remote_code); the 30B-A3B
        scaling-ablation also loads via the same auto class.
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor
        # Use cached files when available; fall back to download if not.
        import os
        local_only = os.environ.get("HF_LOCAL_ONLY", "1") == "1"
        self.tokenizer = AutoProcessor.from_pretrained(
            self.cfg.hf_repo, local_files_only=local_only
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.hf_repo,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=self.cfg.device,
            local_files_only=local_only,
        )
        for p in self.model.parameters():
            p.requires_grad = False
        # Qwen3-VL's text backbone hidden size is under `model.config.text_config`.
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        hidden_dim = text_cfg.hidden_size
        self.projection = ThoughtProjection(hidden_dim, self.cfg.bridge_dim).to(
            self.cfg.device, dtype=torch.bfloat16
        )
        # TODO: inject LoRA via peft on attention layers
        return self

    @torch.no_grad()
    def emit(self,
             messages: list[dict],
             frame: Optional[np.ndarray] = None,
             max_new_tokens: Optional[int] = None,
             ) -> tuple[str, torch.Tensor]:
        """Run one emission cycle of the slow model.

        Args:
            messages: chat-format message list (system/user/assistant turns) as built by
                `src.training.prompts.build_slow_model_messages`.
            frame: optional [H, W, 3] uint8 RGB frame to attach to the user turn.
                Qwen3-VL accepts images via the chat template's image content. Ignored
                if the model's `cfg.vision_enabled` is False (scaling-ablation case).
            max_new_tokens: cap on generated tokens; defaults to `cfg.max_emission_tokens`.

        Returns:
            text_emission: decoded text (everything generated after the prompt)
            thought_vectors: [L, D_bridge] bf16 tensor on `cfg.device`, one vector per
                generated token (from the residual stream at `cfg.projection_layer_idx`
                projected through `self.projection`).
        """
        if self.model is None or self.tokenizer is None or self.projection is None:
            raise RuntimeError("call load_pretrained() before emit()")

        max_new_tokens = max_new_tokens or self.cfg.max_emission_tokens
        proc = self.tokenizer  # AutoProcessor

        # Normalize all messages to multimodal content-block format. Qwen3-VL's
        # processor requires this when any message contains visual content (and
        # accepts it uniformly when there's no visual content).
        messages = [dict(m) for m in messages]
        for m in messages:
            if isinstance(m["content"], str):
                m["content"] = [{"type": "text", "text": m["content"]}]

        # Attach the frame to the final user turn if vision is enabled and a frame
        # was supplied.
        if frame is not None and self.cfg.vision_enabled:
            if Image is None:
                raise RuntimeError("PIL required for frame attachment but not installed")
            pil = Image.fromarray(frame.astype(np.uint8))
            last_user = next(i for i in range(len(messages) - 1, -1, -1)
                             if messages[i]["role"] == "user")
            messages[last_user]["content"] = [
                {"type": "image", "image": pil},
                *messages[last_user]["content"],
            ]

        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        ).to(self.cfg.device)

        prompt_len = int(inputs["input_ids"].shape[1])

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        # Decode only the newly generated tokens (skip the prompt).
        gen_ids = gen_out.sequences[0, prompt_len:]
        text_emission = proc.decode(gen_ids, skip_special_tokens=True)

        # `hidden_states` from generate() is a tuple of length n_new_tokens; each entry
        # is a tuple of length (num_layers + 1) — the embedding layer + each decoder
        # block's output. Index 0 is the embedding; index k is layer k's output.
        # We project the residual stream at our target layer for each generated token.
        target_layer = self.cfg.projection_layer_idx
        per_token_residuals = []
        for step_hidden_states in gen_out.hidden_states:
            # step_hidden_states[target_layer] has shape [B, T_step, D_h]; on the first
            # step T_step = prompt_len, on later steps T_step = 1 (KV-cache decoding).
            # We want the residual at the last position (the newly generated token).
            residual = step_hidden_states[target_layer][0, -1, :]  # [D_h]
            per_token_residuals.append(residual)
        residuals = torch.stack(per_token_residuals, dim=0).unsqueeze(0)  # [1, L, D_h]
        thought_vectors = self.projection(residuals.to(self.projection.proj[1].weight.dtype))
        thought_vectors = thought_vectors.squeeze(0)  # [L, D_bridge]

        return text_emission, thought_vectors
