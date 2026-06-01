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

import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from torch import nn


def _env_n_bridge_tokens() -> int:
    """Read LB_BRIDGE_N_TOKENS env var, falling back to 8."""
    v = os.environ.get("LB_BRIDGE_N_TOKENS", "")
    if v and v.isdigit():
        return int(v)
    return 8

try:
    from PIL import Image
except ImportError:  # PIL is a transformers dependency; guarded for type-only contexts
    Image = None  # type: ignore


def _default_hf_repo() -> str:
    """Default slow-model repo, overridable by env var for the scaling ablation.

    Set LB_USE_SCALING_SLOW=1 in the environment to switch to the 30B-A3B VL
    variant without having to pass a config flag through every entry point.

    Initially we tried Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 but transformers
    4.57.6 + compressed_tensors 0.14.0 silently drop the expert
    weight_scale_inv tensors during load, leaving them on meta and crashing
    at dispatch. We fall back to the bf16 checkpoint (60GB) which fits in
    96GB GPU alongside the 18GB MiniCPM-o fast model.
    """
    sv = os.environ.get("LB_USE_SCALING_SLOW", "0")
    if sv == "bf16":
        # Full-precision 30B (~60 GB GPU). Use when GPU has ~80+ GB free.
        # This is the only variant that fully loads: AWQ silently drops MoE
        # expert weights under transformers 4.57; FP8 drops weight_scale_inv.
        return "Qwen/Qwen3-VL-30B-A3B-Thinking"
    if sv == "fp8":
        return "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"
    if sv in ("1", "awq"):
        # AWQ 4-bit — historical default; KNOWN BROKEN on MoE under tf 4.57.
        return "QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ"
    # Local-staged path override (dodges shared HF-cache resolution contention).
    path = os.environ.get("LB_SLOW_MODEL_PATH")
    if path:
        return path
    return "Qwen/Qwen3-VL-8B-Thinking"


def _default_proj_layer_idx() -> int:
    """Projection layer index — depth-67% of the slow model's transformer stack.
    8B has 36 layers → 24; 30B has 48 layers → 32."""
    if os.environ.get("LB_USE_SCALING_SLOW", "0") in ("1", "bf16", "fp8", "awq"):
        return 32
    return 24


@dataclass
class SlowModelConfig:
    hf_repo: str = field(default_factory=_default_hf_repo)
    dtype: str = "bfloat16"
    device: str = "cuda"
    projection_layer_idx: int = field(default_factory=_default_proj_layer_idx)
    # v2: bridge_dim equals the FAST model's hidden_dim (4096 for Qwen3-8B in MiniCPM-o).
    # The slow projection maps slow's residual (4096) → fast's input embedding (4096).
    bridge_dim: int = 4096
    # v2: take the last N positions of the slow's emission as bridge tokens.
    # These positions are closest to the slow model's "answer" (after <think>).
    # Can be overridden by LB_BRIDGE_N_TOKENS env var (for bandwidth ablations).
    n_bridge_tokens: int = field(default_factory=_env_n_bridge_tokens)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = ("q_proj", "v_proj")
    emission_rate_hz: float = 1.0  # how often to emit thought vectors
    max_emission_tokens: int = 128  # text tokens per emission
    vision_enabled: bool = True     # primary model has native vision

    @classmethod
    def from_scaling_ablation(cls) -> "SlowModelConfig":
        """30B-A3B-FP8 VL MoE configuration for the single scaling ablation.

        - hf_repo: Qwen-official FP8 checkpoint (~30GB on disk vs bf16 ~60GB).
        - hidden_size: 2048 (auto-detected from config; ThoughtProjection adapts).
        - num_hidden_layers: 48 → projection_layer_idx=32 keeps the same ~67% depth
          ratio as the 8B's layer 24 / 36.
        - vision_enabled: True — Qwen3-VL-30B-A3B-Thinking is the VL variant.
        - bridge_dim: still 4096 (MiniCPM-o fast hidden size, unchanged).
        """
        return cls(
            hf_repo="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
            projection_layer_idx=32,
            vision_enabled=True,
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
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
        # Use cached files when available; fall back to download if not.
        local_only = os.environ.get("HF_LOCAL_ONLY", "1") == "1"
        self.tokenizer = AutoProcessor.from_pretrained(
            self.cfg.hf_repo, local_files_only=local_only
        )
        # Quantized checkpoints: device_map strategy depends on quant method.
        #   - AWQ: requires single-device {"": 0} (NOT "auto", which can place
        #     CPU/disk shards and trip a validate_environment check)
        #   - FP8 / compressed-tensors: still uses "auto" via accelerate dispatch
        cfg = AutoConfig.from_pretrained(self.cfg.hf_repo, local_files_only=local_only)
        qcfg = getattr(cfg, "quantization_config", None)
        is_quantized = qcfg is not None
        quant_method = (qcfg.get("quant_method") if isinstance(qcfg, dict)
                        else getattr(qcfg, "quant_method", None)) if qcfg else None
        if is_quantized:
            if quant_method in ("awq", "gptq"):
                device_map = {"": 0}
            else:
                device_map = "auto"
        else:
            device_map = self.cfg.device
        load_kwargs = dict(
            attn_implementation="sdpa",
            device_map=device_map,
            local_files_only=local_only,
        )
        if not is_quantized:
            # Quantized models manage their own dtype via the quant config.
            load_kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.hf_repo, **load_kwargs,
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
