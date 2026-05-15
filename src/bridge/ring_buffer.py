"""Continuous-valued ring buffers carrying thought vectors (slow→fast) and
perception summaries (fast→slow).

Both buffers support:
- Append-only writes with monotonic timestamps
- Read of the most-recent K entries with age encodings
- Thread-safe (slow model writes from background; fast model reads on every tick)
"""
from __future__ import annotations

import math
import threading
import torch


class _RingBuffer:
    """Base ring buffer of fixed-dim vectors with age tracking."""

    def __init__(self, capacity: int, dim: int, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.buf = torch.zeros(capacity, dim, device=device, dtype=dtype)
        self.timestamps = torch.zeros(capacity, device=device, dtype=torch.float32)
        self.next_idx = 0
        self.size = 0
        self.lock = threading.Lock()

    def append(self, vec: torch.Tensor, timestamp: float):
        """vec: [dim] tensor (single entry)."""
        assert vec.shape == (self.dim,), f"expected shape ({self.dim},), got {vec.shape}"
        with self.lock:
            self.buf[self.next_idx] = vec.to(self.device, dtype=self.dtype)
            self.timestamps[self.next_idx] = timestamp
            self.next_idx = (self.next_idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def read(self, current_time: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (entries[K, dim], age_seconds[K]). Empty slots filled with zeros and
        large age."""
        with self.lock:
            entries = self.buf.clone()
            ages = current_time - self.timestamps
            # Mark uninitialised slots as very old so the cross-attn can learn to ignore
            if self.size < self.capacity:
                # Slots from `size` onward (in cyclic order) are uninitialised
                # For simplicity, just check timestamps == 0 as a proxy for unwritten
                ages = torch.where(self.timestamps == 0, torch.tensor(1e6, device=self.device), ages)
        return entries, ages

    def age_encoding(self, ages: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
        """Sinusoidal age encoding to be added to entries before cross-attn."""
        # ages: [K]
        K = ages.shape[0]
        freqs = torch.tensor(
            [1.0 / (2.0 ** i) for i in range(n_freq)],
            device=ages.device, dtype=ages.dtype,
        )  # [n_freq]
        phases = ages.unsqueeze(-1) * freqs.unsqueeze(0)  # [K, n_freq]
        sins, coss = torch.sin(phases), torch.cos(phases)
        encoding_short = torch.cat([sins, coss], dim=-1)  # [K, 2*n_freq]
        # Pad/repeat to bridge dim
        n_repeats = math.ceil(self.dim / encoding_short.size(-1))
        encoding = encoding_short.repeat(1, n_repeats)[:, :self.dim]
        return encoding.to(self.dtype)


class ThoughtBuffer(_RingBuffer):
    """slow → fast. Capacity K=16, dim=256 default."""
    def __init__(self, capacity: int = 16, dim: int = 256, **kwargs):
        super().__init__(capacity, dim, **kwargs)


class PerceptionBuffer(_RingBuffer):
    """fast → slow. Capacity K=8, dim=256 default."""
    def __init__(self, capacity: int = 8, dim: int = 256, **kwargs):
        super().__init__(capacity, dim, **kwargs)
