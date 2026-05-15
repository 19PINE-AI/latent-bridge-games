"""Week-1 validation script: confirm joint inference of MiniCPM-o 4.5 + Qwen3-30B-A3B
fits on the 96GB GPU at bf16, and measure forward-pass latency for each.

Run: `python scripts/validate_hardware.py`

Expected output:
- MiniCPM-o 4.5 weights load successfully
- Qwen3-30B-A3B-Thinking weights load successfully
- Both fit jointly in GPU memory
- Fast-model forward pass: <100ms per tick
- Slow-model forward pass (1 emission): <1500ms
"""
from __future__ import annotations

import sys
import time

import torch


def report(msg: str):
    print(f"[validate] {msg}")


def fmt_gb(bytes_: int) -> str:
    return f"{bytes_ / 1024**3:.2f}GB"


def main():
    if not torch.cuda.is_available():
        report("ERROR: CUDA not available")
        sys.exit(1)

    dev_props = torch.cuda.get_device_properties(0)
    total = dev_props.total_memory
    report(f"GPU: {dev_props.name}, total VRAM: {fmt_gb(total)}")

    if total < 80 * 1024**3:
        report("WARNING: <80GB VRAM; joint inference likely to OOM")

    # ---- Phase 1: load MiniCPM-o 4.5 ----
    report("Loading MiniCPM-o 4.5 in bf16...")
    t0 = time.time()
    from transformers import AutoModel
    fast = AutoModel.from_pretrained(
        "openbmb/MiniCPM-o-4_5",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    report(f"  loaded in {time.time()-t0:.1f}s")
    report(f"  VRAM used: {fmt_gb(torch.cuda.memory_allocated())}")

    # ---- Phase 2: load Qwen3-30B-A3B-Thinking ----
    report("Loading Qwen3-30B-A3B-Thinking in bf16...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-30B-A3B-Thinking-2507", trust_remote_code=True
    )
    slow = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    report(f"  loaded in {time.time()-t0:.1f}s")
    report(f"  VRAM used: {fmt_gb(torch.cuda.memory_allocated())}")
    report(f"  free VRAM remaining: {fmt_gb(total - torch.cuda.memory_allocated())}")

    # ---- Phase 3: measure forward-pass latencies ----
    # TODO: minimal forward-pass timing. Implementation deferred to first runnable day.
    report("Forward-pass latency measurement: TODO")

    report("Validation complete. Both models fit jointly.")


if __name__ == "__main__":
    main()
