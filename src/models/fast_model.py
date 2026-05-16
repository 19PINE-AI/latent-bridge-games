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
    xattn_gated: bool = False  # if True, output is multiplied by a learnable gate
    backbone_hidden_dim: int = 4096  # Qwen3-8B backbone in MiniCPM-o-4.5
    bridge_dim: int = 256
    n_actions: int = 18  # ALE max action space size


class BridgeCrossAttention(nn.Module):
    """Cross-attention layer that reads the thought-vector ring buffer.

    Initialised so output projection is zero — model behaves identically to base at init.
    Bridge contribution emerges through training.

    Optional gating (`gated=True`): the output is multiplied elementwise by
    `sigmoid(gate(hidden))`, letting the model learn to suppress the bridge in states
    where its contribution would be harmful. Gate weights start at zero, biases at
    -2 (sigmoid(-2) ≈ 0.12), so the bridge is initially mostly silent and the model
    can gradually open the gate where helpful.

    The plain (ungated) variant is what the original Stage C trained — it can produce
    catastrophic behavior when the bridge output is wrong because the frozen action
    head can't ignore the perturbation. The gated variant lets the model route around
    bad bridge outputs by closing the gate.
    """
    def __init__(self, hidden_dim: int, bridge_dim: int, n_heads: int = 8,
                 gated: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.gated = gated
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(bridge_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(bridge_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Zero-init the output projection so the layer is a no-op at start of training
        nn.init.zeros_(self.o_proj.weight)
        if gated:
            self.gate = nn.Linear(hidden_dim, hidden_dim, bias=True)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12

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
        out = self.o_proj(out)
        if self.gated:
            out = out * torch.sigmoid(self.gate(hidden))
        return out


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
                gated=cfg.xattn_gated,
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

        # Vision-token cache (v2 latency optimization)
        # Caches the most-recent (vllm_emb, attention_mask) tuple. Reused for the next
        # N-1 ticks when `predict_action(vision_refresh_every=N)` is passed and the
        # `slow_text_suffix` is unchanged.
        # Saves 80-150 ms per tick when N > 1, at the cost of 1-K-tick visual staleness.
        # Atari frames change by ~1 px per tick at 15 Hz so staleness is mild.
        self._vision_cache_emb: Optional[torch.Tensor] = None
        self._vision_cache_mask: Optional[torch.Tensor] = None
        self._vision_cache_age: int = 0
        self._vision_cache_prompt_id: Optional[str] = None
        # Latency budget for real-time tick rate (paper claim: 67ms @ 15Hz, 100ms @ 10Hz).
        # Current breakdown after max_slice_nums=1 (profiled): processor 14ms +
        # to_device 42ms + vision/get_vllm_embedding 80ms + llm.model forward 52ms +
        # action head <1ms = ~197ms. Optimization roadmap (future work):
        #   - torch.compile on llm.model:                potentially 52ms → ~26ms
        #   - Vision tower at lower input resolution:    80ms → ~30ms
        #   - Pinned-memory + async H2D transfer:        42ms → ~5ms
        #   - Optional vision-token caching every Nth tick (1-tick staleness tradeoff)
        # Realistic floor with these wins: ~80ms (>= LLM forward + processor minimum).

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

    def reset_vision_cache(self):
        """Clear the vision-token cache. Call between episodes."""
        self._vision_cache_emb = None
        self._vision_cache_mask = None
        self._vision_cache_age = 0

    def predict_action(self,
                       frame: np.ndarray,
                       thought_tokens: Optional[torch.Tensor] = None,
                       thought_buffer: Optional[torch.Tensor] = None,  # deprecated alias
                       age_encoding: Optional[torch.Tensor] = None,    # ignored in v2
                       legal_action_mask: Optional[torch.Tensor] = None,
                       slow_text_suffix: Optional[str] = None,
                       vision_refresh_every: int = 1,
                       ) -> torch.Tensor:
        """Frame → 18-way action logits.

        v2 pipeline (LLaVA-style):
          frame
            → MiniCPM-o processor
            → get_vllm_embedding fuses 64 SigLIP visual tokens into the text embed seq
            → PREPEND `thought_tokens` (N×4096) to the embedding sequence
            → llm.model with extended inputs_embeds (all 36 layers attend over bridge)
            → action_head on last hidden state position

        Args:
            frame: [H, W, 3] uint8 RGB Atari frame.
            thought_tokens: [B=1, N, hidden_dim] or None — bridge tokens in fast LLM's
                input embedding space, produced by SlowModel's ThoughtProjection.
                These are prepended to the sequence so all 36 LLM layers attend to them.
            thought_buffer: deprecated alias for thought_tokens (v1 name).
            age_encoding: unused in v2.
            legal_action_mask: [18] bool — illegal actions get logits → -inf.
            slow_text_suffix: optional T-condition text from slow model (text bridge).

        Returns: [1, 18] action logits.
        """
        if self.model is None:
            raise RuntimeError("call load_pretrained() before predict_action()")
        if Image is None:
            raise RuntimeError("PIL is required for frame preprocessing")

        # Backward-compat alias
        if thought_tokens is None and thought_buffer is not None:
            thought_tokens = thought_buffer

        pil = Image.fromarray(frame.astype(np.uint8))
        prompt_str = (
            self._build_action_prompt(slow_text_suffix)
            if slow_text_suffix else self._action_prompt_str
        )

        # Vision-token cache: if recent vllm_emb is fresh enough and the prompt
        # is unchanged, skip processor + vision tower entirely.
        prompt_id = prompt_str[:128]  # cheap identity key
        use_cache = (
            vision_refresh_every > 1
            and self._vision_cache_emb is not None
            and self._vision_cache_age < vision_refresh_every
            and self._vision_cache_prompt_id == prompt_id
        )

        if use_cache:
            vllm_emb = self._vision_cache_emb
            attention_mask = self._vision_cache_mask
            self._vision_cache_age += 1
        else:
            inputs = self.model.processor(
                [prompt_str],
                [[pil]],
                [[]],
                [[]],
                max_slice_nums=1,
                use_image_id=None,
                return_tensors="pt",
                max_length=1024,
            )

            def to_dev(x):
                if isinstance(x, torch.Tensor):
                    return x.to(self.cfg.device)
                if isinstance(x, list):
                    return [to_dev(y) for y in x]
                return x
            inputs = {k: to_dev(v) for k, v in inputs.items()}

            # Vision + token embedding fusion
            vllm_emb, _ = self.model.get_vllm_embedding(inputs)  # [1, T, 4096]
            attention_mask = inputs.get("attention_mask")

            # Refresh cache
            if vision_refresh_every > 1:
                self._vision_cache_emb = vllm_emb
                self._vision_cache_mask = attention_mask
                self._vision_cache_prompt_id = prompt_id
                self._vision_cache_age = 1

        # v2: prepend bridge tokens to the embedding sequence
        if thought_tokens is not None and thought_tokens.numel() > 0:
            bt = thought_tokens
            if bt.dim() == 2:  # [N, D] → [1, N, D]
                bt = bt.unsqueeze(0)
            bt = bt.to(vllm_emb.device, dtype=vllm_emb.dtype)
            vllm_emb = torch.cat([bt, vllm_emb], dim=1)  # [1, N+T, 4096]
            if attention_mask is not None:
                bridge_mask = torch.ones(
                    bt.shape[0], bt.shape[1],
                    device=attention_mask.device, dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([bridge_mask, attention_mask], dim=1)

        out = self.model.llm.model(
            inputs_embeds=vllm_emb,
            attention_mask=attention_mask,
            use_cache=False,
        )

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
