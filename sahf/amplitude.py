"""
Stage 1 — Amplitude Interception Kernel.

Converts raw logits into the square-root-of-probability ("amplitude") representation,
psi_v = sqrt(softmax(z)_v). This lands every distribution on the positive orthant of
the unit hypersphere, where ordinary Euclidean distance / dot-products correspond to
real statistical distances (Hellinger distance, Bhattacharyya coefficient) instead of
being an arbitrary choice on raw probabilities. Every later stage (the divergence
gate, fast mean fusion, the geometric median) depends on operating in this space.
"""

import torch


def softmax_to_amplitude(logits: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    Args:
        logits: raw model output, shape (..., vocab_size).
        dim: dimension the softmax is taken over.
        eps: floor applied to probabilities before the sqrt, so that a token with
             (numerically) zero probability does not later produce a NaN gradient/
             division in the gate, mean-fusion, or Weiszfeld steps.

    Returns:
        psi: same shape as logits, non-negative, with sum(psi**2, dim=dim) == 1.
    """
    probs = torch.softmax(logits, dim=dim)
    probs = probs.clamp_min(eps)
    psi = torch.sqrt(probs)
    return psi
