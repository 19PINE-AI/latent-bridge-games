"""Fast model: MiniCPM-o 4.5 (9B) with LoRA + cross-attention bridge layers.

Frozen base, trainable LoRA on Qwen3-8B backbone attention, and new cross-attention
layers (at depths 12 and 24) that attend over the thought-vector ring buffer.

Architecture (probed live from MiniCPM-o-4_5):
  - Top-level model = `MiniCPMO`
  - LLM backbone:  model.llm = Qwen3ForCausalLM (36 Qwen3DecoderLayer, hidden=4096)
  - Vision tower:  model.vpm = SiglipVisionTransformer + model.resampler
  - Audio/TTS:     model.apm + model.tts (unused for Atari)

The cross-attention bridge is injected via `register_forward_hook` on
`model.llm.model.layers[12]` and `[24]`. The hook receives the post-block hidden state
and returns `hidden + bridge_xattn(hidden, thought_buffer)`. Output projection is
zero-initialised so the model behaves identically to base at init.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from torch import nn

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


ACTION_PROMPT = (
    "<image>./</image>\nThis is the current Atari frame. "
    "Choose the next action."
)


@dataclass
class FastModelConfig:
    hf_repo: str = "openbmb/MiniCPM-o-4_5"
    dtype: str = "bfloat16"
    device: str = "cuda"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    xattn_insert_layers: list = field(default_factory=lambda: [12, 24])
    xattn_n_heads: int = 8
    backbone_hidden_dim: int = 4096  # Qwen3-8B backbone in MiniCPM-o-4.5
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

    Trainable parameters: cross-attn weights (Q/K/V/O for each insertion layer) and the
    action head. LoRA on the Qwen3-8B attention layers is added separately by Stage A.
    Base weights are frozen.
    """
    def __init__(self, cfg: FastModelConfig):
        super().__init__()
        self.cfg = cfg
        # Model loaded lazily via load_pretrained()
        self.model = None
        # Cross-attention layers (one per inject point), registered as a ModuleDict
        # so .parameters() finds them for the optimizer.
        self.xattn_layers = nn.ModuleDict({
            str(li): BridgeCrossAttention(
                hidden_dim=cfg.backbone_hidden_dim,
                bridge_dim=cfg.bridge_dim,
                n_heads=cfg.xattn_n_heads,
            )
            for li in cfg.xattn_insert_layers
        })
        # Action head: reads the last hidden state, predicts 18-way action logits.
        self.action_head = nn.Linear(cfg.backbone_hidden_dim, cfg.n_actions)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        # Internal state holding the current thought buffer for the forward pass.
        # Set by `forward()`; read by the hooks.
        self._current_thought_buffer: Optional[torch.Tensor] = None
        self._current_age_encoding: Optional[torch.Tensor] = None
        self._hook_handles: list = []

    def load_pretrained(self):
        """Load MiniCPM-o 4.5 from HF, freeze, attach cross-attn hooks + processor."""
        import os
        local_only = os.environ.get("HF_LOCAL_ONLY", "1") == "1"
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(
            self.cfg.hf_repo,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map=self.cfg.device,
            local_files_only=local_only,
        ).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Initialize the integrated processor (image+text). Required for vision input.
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.hf_repo, trust_remote_code=True, local_files_only=local_only,
        )
        self.model.prepare_processor(tokenizer=tokenizer)
        # Cache the assembled prompt for action prediction so we don't rebuild every tick.
        self._action_prompt_str = self.model.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": ACTION_PROMPT}],
            tokenize=False, add_generation_prompt=True,
        )
        # Move our trainable bridge modules onto the same device + dtype as the LLM.
        target_dtype = next(self.model.llm.parameters()).dtype
        self.xattn_layers.to(self.cfg.device, dtype=target_dtype)
        self.action_head.to(self.cfg.device, dtype=target_dtype)
        self._attach_bridge_hooks()
        return self

    def _attach_bridge_hooks(self) -> None:
        """Register post-layer forward hooks at the configured Qwen3 decoder layers.

        Each hook adds `bridge_xattn(hidden_state, thought_buffer)` to the residual
        stream that flows into the next decoder block.

        Output of Qwen3DecoderLayer is a single Tensor (post-block hidden state), as
        confirmed by the topology probe.
        """
        # Remove any previously-attached hooks (idempotent if called more than once).
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

        decoder_layers = self.model.llm.model.layers

        def make_hook(layer_idx: int):
            xattn: BridgeCrossAttention = self.xattn_layers[str(layer_idx)]
            def hook(module, inputs, output):
                if self._current_thought_buffer is None:
                    return output
                if not isinstance(output, torch.Tensor):
                    # Defensive: some transformer versions return tuples.
                    hidden = output[0]
                    bridge_out = xattn(hidden, self._current_thought_buffer,
                                       self._current_age_encoding)
                    return (hidden + bridge_out, *output[1:])
                bridge_out = xattn(output, self._current_thought_buffer,
                                   self._current_age_encoding)
                return output + bridge_out
            return hook

        for li in self.cfg.xattn_insert_layers:
            handle = decoder_layers[li].register_forward_hook(make_hook(li))
            self._hook_handles.append(handle)

    def set_bridge_inputs(self, thought_buffer: Optional[torch.Tensor],
                          age_encoding: Optional[torch.Tensor] = None) -> None:
        """Set the thought buffer that the bridge hooks will read on the next forward.

        Calling with `thought_buffer=None` disables the bridge (hooks pass through).

        thought_buffer: [B, K, D_bridge] or None
        age_encoding:   [B, K, D_bridge] or None (sinusoidal age stamp)
        """
        self._current_thought_buffer = thought_buffer
        self._current_age_encoding = age_encoding

    def action_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, T, D_h] → [B, n_actions]  (uses the last position)."""
        return self.action_head(hidden[:, -1, :])

    def trainable_parameters(self):
        """Yield only the trainable parameters (cross-attn + action head + later LoRA)."""
        for p in self.xattn_layers.parameters():
            yield p
        for p in self.action_head.parameters():
            yield p

    def forward(self, *args, **kwargs):
        """Forward through the underlying MiniCPM-o model. Use `set_bridge_inputs()`
        before calling to enable the bridge.

        For Stage A imitation training the entrypoint is typically
        `forward_llm_only(input_ids)` for single-frame action prediction; this method
        delegates to `self.model.__call__` for full multimodal use.
        """
        return self.model(*args, **kwargs)

    def _build_action_prompt(self, slow_text_suffix: Optional[str] = None) -> str:
        """Build the chat-templated prompt string for action prediction.

        If `slow_text_suffix` is provided (T-condition), it is appended inside the
        user turn as `[strategic-guidance]: <text>` so the fast model can attend to it.
        """
        user_content = ACTION_PROMPT
        if slow_text_suffix:
            user_content = (
                f"{ACTION_PROMPT}\n[strategic-guidance]: {slow_text_suffix.strip()}"
            )
        return self.model.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )

    def predict_action(self,
                       frame: np.ndarray,
                       thought_buffer: Optional[torch.Tensor] = None,
                       age_encoding: Optional[torch.Tensor] = None,
                       legal_action_mask: Optional[torch.Tensor] = None,
                       slow_text_suffix: Optional[str] = None,
                       ) -> torch.Tensor:
        """Frame → 18-way action logits.

        Pipeline: frame -> MiniCPM-o processor -> vision tower (.vpm + .resampler) ->
        get_vllm_embedding (text token embeds with vision tokens scattered in) ->
        llm.model (bridge hooks fire at layers 12, 24) -> action_head on last position.

        Args:
            frame: [H, W, 3] uint8 RGB Atari frame.
            thought_buffer: [B=1, K, D_bridge] or None — slow-model thought-vector ring.
            age_encoding: [B=1, K, D_bridge] or None — sinusoidal age for each entry.
            legal_action_mask: [18] bool — illegal actions get logits → -inf.
            slow_text_suffix: optional T-condition text from slow model.

        Returns: [1, 18] action logits.
        """
        if self.model is None:
            raise RuntimeError("call load_pretrained() before predict_action()")
        if Image is None:
            raise RuntimeError("PIL is required for frame preprocessing")

        pil = Image.fromarray(frame.astype(np.uint8))
        prompt_str = (
            self._build_action_prompt(slow_text_suffix)
            if slow_text_suffix else self._action_prompt_str
        )
        inputs = self.model.processor(
            [prompt_str],
            [[pil]],
            [[]],
            [[]],
            max_slice_nums=2,
            use_image_id=None,
            return_tensors="pt",
            max_length=1024,
        )

        # Move tensors/lists-of-tensors to device.
        def to_dev(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.cfg.device)
            if isinstance(x, list):
                return [to_dev(y) for y in x]
            return x
        inputs = {k: to_dev(v) for k, v in inputs.items()}

        # Vision + token embedding fusion
        vllm_emb, _ = self.model.get_vllm_embedding(inputs)

        # Activate the bridge for this forward pass
        self.set_bridge_inputs(thought_buffer, age_encoding)
        out = self.model.llm.model(
            inputs_embeds=vllm_emb,
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        )
        self.set_bridge_inputs(None)  # clear so a subsequent forward without bridge is clean

        logits = self.action_logits_from_hidden(out.last_hidden_state)
        if legal_action_mask is not None:
            mask = legal_action_mask.to(logits.device)
            logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        return logits

    def forward_llm_only(self, input_ids: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the LLM backbone (frozen + bridge) on token IDs, return final hidden state.

        Useful for: (a) probing cross-attn injection with synthetic inputs (no images),
        (b) Stage A action-head training when the prompt is text-only.

        Returns: [B, T, D_h]
        """
        if self.model is None:
            raise RuntimeError("call load_pretrained() before forward_llm_only()")
        out = self.model.llm.model(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            use_cache=False,
        )
        return out.last_hidden_state
