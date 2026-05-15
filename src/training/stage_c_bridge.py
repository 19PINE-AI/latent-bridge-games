"""Stage C: Bridge supervised training via COCONUT-style curriculum.

C0: bridge carries text tokens (replicates T baseline)
C1: first half of slow-model emissions replaced with latents
C2: all slow-model emissions latents; MSE-against-C0 supervision

Status: skeleton, wired up in Week 2.
"""
from __future__ import annotations


def main():
    raise NotImplementedError("Stage C: implement in Week 2 after Stage B baseline.")


if __name__ == "__main__":
    main()
