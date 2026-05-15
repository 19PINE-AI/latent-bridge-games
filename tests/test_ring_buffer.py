"""Unit tests for the ring buffer."""
from __future__ import annotations

import pytest
import torch

from src.bridge import ThoughtBuffer, PerceptionBuffer


def _has_cuda():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_thought_buffer_append_and_read():
    buf = ThoughtBuffer(capacity=4, dim=8)
    for i in range(3):
        v = torch.full((8,), float(i), device="cuda", dtype=torch.bfloat16)
        buf.append(v, timestamp=float(i))
    entries, ages = buf.read(current_time=5.0)
    assert entries.shape == (4, 8)
    assert ages.shape == (4,)
    # Empty slot should have very large age
    assert ages.max().item() > 1000


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_ring_wraparound():
    buf = ThoughtBuffer(capacity=4, dim=8)
    for i in range(10):
        v = torch.full((8,), float(i), device="cuda", dtype=torch.bfloat16)
        buf.append(v, timestamp=float(i))
    entries, ages = buf.read(current_time=11.0)
    # All slots filled
    assert (ages < 100).all()
    # Most recent values: 6, 7, 8, 9 in cyclic order
    values = entries.float().mean(dim=-1).cpu().tolist()
    assert sorted(values) == [6.0, 7.0, 8.0, 9.0]


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_age_encoding_shape():
    buf = ThoughtBuffer(capacity=4, dim=64)
    ages = torch.tensor([0.0, 0.5, 1.0, 2.0], device="cuda")
    enc = buf.age_encoding(ages, n_freq=8)
    assert enc.shape == (4, 64)
