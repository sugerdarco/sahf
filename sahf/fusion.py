"""
Stage 5 — Fast Mean Fusion.

psi* = normalize( sum_i w_i * psi_i )

A weighted arithmetic (chordal) mean of the amplitude vectors, re-projected back
onto the unit sphere. O(V), closed-form, no iteration — this is the cheapest
possible real fusion, and it is tried before anything more expensive. It is a good
approximation to the true spherical (Frechet) mean whenever the agents are
genuinely, but honestly, disagreeing (a small-angle regime on the sphere). It is
not robust to a single adversarial/corrupted agent — that is what Stage 6/7 exist
to catch.
"""

from typing import List, Optional

import torch


def fast_mean_fusion(psi_list: List[torch.Tensor], weights: Optional[List[float]] = None,
                      eps: float = 1e-12) -> torch.Tensor:
    n = len(psi_list)
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError("weights must have one entry per agent.")

    weighted_sum = sum(w * psi for w, psi in zip(weights, psi_list))
    norm = torch.norm(weighted_sum, p=2).clamp_min(eps)
    return weighted_sum / norm
