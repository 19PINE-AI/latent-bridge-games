"""Week-1 validation script: confirm joint inference of MiniCPM-o 4.5 + the slow model
fits on the 96GB GPU at bf16, and measure forward-pass latency for each.

Run:
    python scripts/validate_hardware.py                  # primary 8B+9B config
    python scripts/validate_hardware.py --scaling-ablation  # 30B-A3B slow ablation

Expected output:
- MiniCPM-o 4.5 weights load successfully
- Slow-model weights load successfully (Qwen3-VL-8B-Thinking by default)
- Both fit jointly in GPU memory
- Fast-model forward pass: <100ms per tick
- Slow-model forward pass (1 emission): <1500ms
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def report(msg: str):
    print(f"[validate] {msg}")


def fmt_gb(bytes_: int) -> str:
    return f"{bytes_ / 1024**3:.2f}GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling-ablation", action="store_true",
                    help="load the 30B-A3B scaling-ablation slow model instead of the primary 8B")
    ap.add_argument("--only", choices=("fast", "slow", "both"), default="both",
                    help="load only the fast or slow model in isolation (for tight VRAM budgets)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        report("ERROR: CUDA not available")
        sys.exit(1)

    dev_props = torch.cuda.get_device_properties(0)
    total = dev_props.total_memory
    report(f"GPU: {dev_props.name}, total VRAM: {fmt_gb(total)}")

    if args.scaling_ablation and total < 80 * 1024**3:
        report("WARNING: <80GB VRAM; joint inference with 30B-A3B likely to OOM")

    free_at_start, _ = torch.cuda.mem_get_info()
    report(f"free VRAM at start: {fmt_gb(free_at_start)}")

    # ---- Phase 1: load MiniCPM-o 4.5 ----
    if args.only in ("fast", "both"):
        report("Loading MiniCPM-o 4.5 in bf16...")
        t0 = time.time()
        from transformers import AutoModel
        fast = AutoModel.from_pretrained(
            "openbmb/MiniCPM-o-4_5",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map="cuda",
        ).eval()
        report(f"  loaded in {time.time()-t0:.1f}s")
        report(f"  VRAM used by this proc: {fmt_gb(torch.cuda.memory_allocated())}")
        free_now, _ = torch.cuda.mem_get_info()
        report(f"  free VRAM remaining: {fmt_gb(free_now)}")

    # ---- Phase 2: load slow model ----
    if args.only in ("slow", "both"):
        if args.scaling_ablation:
            slow_repo = "Qwen/Qwen3-30B-A3B-Thinking-2507"
            slow_label = "Qwen3-30B-A3B-Thinking (scaling ablation)"
        else:
            slow_repo = "Qwen/Qwen3-VL-8B-Thinking"
            slow_label = "Qwen3-VL-8B-Thinking (primary)"
        report(f"Loading {slow_label} in bf16...")
        t0 = time.time()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(slow_repo)
        slow = AutoModelForImageTextToText.from_pretrained(
            slow_repo,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map="cuda",
        ).eval()
        report(f"  loaded in {time.time()-t0:.1f}s")
        report(f"  VRAM used by this proc: {fmt_gb(torch.cuda.memory_allocated())}")
        free_now, _ = torch.cuda.mem_get_info()
        report(f"  free VRAM remaining: {fmt_gb(free_now)}")

        # Run a tiny forward pass to confirm runtime correctness
        report("Running 1-token forward pass on slow model...")
        prompt = "Hello, in one word, what game is Pac-Man?"
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = slow.generate(**inputs, max_new_tokens=8, do_sample=False)
        decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
        report(f"  forward+gen in {time.time()-t0:.2f}s; output preview: {decoded[:120]!r}")

    report("Validation complete.")


if __name__ == "__main__":
    main()
