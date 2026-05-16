"""Slow model: Qwen3-VL-8B-Thinking (frozen) + ThoughtProjection.

v2 (LLaVA-style latent-as-token bridge):
  - emit() returns (text, bridge_tokens) where bridge_tokens has shape
    [N=8, fast_hidden_dim=4096]. These tokens live in the FAST model's input
    embedding space and are prepended to the fast model's input sequence as if
    they were word tokens.
  - ThoughtProjection: MLP `4096 (slow hidden) → 4096 → 4096 (fast embedding)`
    with LayerNorm. Only this module is trainable in Stage C. ~32M params.

emit() also has a `return_raw_residuals=True` mode that returns un-projected
slow residuals at layer 24 (last N positions). This is what we save in the
T-condition trajectory file, so Stage C can re-project them through a trainable
projection without re-running the slow model.
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
    projection_layer_idx: int = 24  # which slow-model layer's residuals to project
    # v2: bridge_dim equals the FAST model's hidden_dim (4096 for Qwen3-8B in MiniCPM-o).
    # The slow projection maps slow's residual (4096) → fast's input embedding (4096).
    bridge_dim: int = 4096
    # v2: take the last N positions of the slow's emission as bridge tokens.
    # These positions are closest to the slow model's "answer" (after <think>).
    n_bridge_tokens: int = 8
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
    """Projects slow-model residuals into the fast model's input embedding space.

    v2 design (LLaVA-style):
        Input:  slow residuals at layer 24, shape [B, T, slow_hidden_dim=4096]
        Output: bridge tokens in fast LLM embedding space, shape [B, T, fast_hidden_dim=4096]

    The output LayerNorm is critical: it constrains the projected tokens to have
    statistics similar to text token embeddings (LayerNorm'd by the LLM's input
    embedding pipeline), giving the frozen LLM an immediate inductive bias for
    treating these tokens as "well-behaved" input.

    Trained via COCONUT-style curriculum in Stage C: latent tokens must produce
    the same logits as the text channel does at the action-prediction position.
    """
    def __init__(self, hidden_dim_in: int, hidden_dim_out: int):
        super().__init__()
        # 4-layer MLP with a LayerNorm bookend on each side
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim_in),
            nn.Linear(hidden_dim_in, hidden_dim_out),
            nn.GELU(),
            nn.Linear(hidden_dim_out, hidden_dim_out),
            nn.LayerNorm(hidden_dim_out),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """residual: [B, T, D_in] → [B, T, D_out]"""
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
             return_raw_residuals: bool = False,
             ) -> tuple[str, torch.Tensor]:
        """Run one emission cycle of the slow model.

        Args:
            messages: chat-format message list (system/user/assistant turns) as built by
                `src.training.prompts.build_slow_model_messages`.
            frame: optional [H, W, 3] uint8 RGB frame to attach to the user turn.
                Qwen3-VL accepts images via the chat template's image content. Ignored
                if the model's `cfg.vision_enabled` is False (scaling-ablation case).
            max_new_tokens: cap on generated tokens; defaults to `cfg.max_emission_tokens`.
            return_raw_residuals: if True, return un-projected slow residuals (last N
                positions × slow_hidden_dim). Useful for caching during T-trajectory
                collection so Stage C can re-project through a trainable projection
                without re-running the slow model. Default False = return projected
                bridge tokens.

        Returns:
            text_emission: decoded text (everything generated after the prompt)
            bridge_or_residuals: shape [N, D] where N = min(L_generated, n_bridge_tokens)
                and D = bridge_dim (projected) or slow_hidden_dim (raw).
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
        target_layer = self.cfg.projection_layer_idx
        per_token_residuals = []
        for step_hidden_states in gen_out.hidden_states:
            # step_hidden_states[target_layer] has shape [B, T_step, D_h]; on the first
            # step T_step = prompt_len, on later steps T_step = 1 (KV-cache decoding).
            # We want the residual at the last position (the newly generated token).
            residual = step_hidden_states[target_layer][0, -1, :]  # [D_h]
            per_token_residuals.append(residual)
        if not per_token_residuals:
            # No tokens generated (rare). Return empty tensor.
            empty_d = self.projection.proj[0].normalized_shape[0] if return_raw_residuals else self.cfg.bridge_dim
            return text_emission, torch.zeros(0, empty_d, device=self.cfg.device, dtype=torch.bfloat16)

        # v2: take only the LAST N positions (closest to the slow model's conclusion).
        n_keep = min(self.cfg.n_bridge_tokens, len(per_token_residuals))
        last_n_residuals = torch.stack(per_token_residuals[-n_keep:], dim=0)  # [N, D_h]

        if return_raw_residuals:
            # Return un-projected residuals so callers can save them and re-project later
            # through a trainable projection. Shape: [N, slow_hidden_dim=4096].
            return text_emission, last_n_residuals

        # Apply the projection: slow_hidden_dim → fast_hidden_dim
        residuals_batched = last_n_residuals.unsqueeze(0)  # [1, N, D_h]
        bridge_tokens = self.projection(
            residuals_batched.to(self.projection.proj[1].weight.dtype)
        ).squeeze(0)  # [N, bridge_dim]

        return text_emission, bridge_tokens

    def project_residuals(self, residuals: torch.Tensor) -> torch.Tensor:
        """Apply the (potentially trainable) ThoughtProjection to a batch of raw
        residuals. Used by Stage C to re-project cached residuals each forward pass.

        Args:
            residuals: [B, N, slow_hidden_dim] (typically bf16 or float32)

        Returns:
            bridge_tokens: [B, N, bridge_dim]
        """
        if self.projection is None:
            raise RuntimeError("call load_pretrained() before project_residuals()")
        target_dtype = self.projection.proj[1].weight.dtype
        return self.projection(residuals.to(target_dtype))
