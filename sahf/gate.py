"""
Stage 2 — Divergence Gate.

Decides, per decoding step, whether the agents already agree (cheap Path A) or
need real fusion (Path B). Two conditions must both hold for "agree":
  - H  (entropy of the mean consensus distribution) < theta_H  — the group as a
    whole is confident about *something*.
  - D  (max pairwise spread between agents' amplitude vectors) < theta_D — the
    agents are confident about the *same* thing.

Either condition failing routes the token to Path B. This mirrors the two-factor
consensus criterion used in "When to Ensemble" (SAFE, arXiv:2510.15346): fusing
every single token is both slower and measurably worse (distributional drift) than
fusing only where it's actually needed.
"""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class GateThresholds:
    entropy: float = 2.0       # theta_H, in nats
    divergence: float = 0.05   # theta_D, Euclidean/Hellinger distance on the amplitude sphere


@dataclass
class GateDecision:
    agree: bool
    entropy: float
    divergence: float


def mean_entropy(psi_list: List[torch.Tensor], eps: float = 1e-12) -> float:
    """Entropy of the plain average of the agents' probability distributions."""
    probs = [psi**2 for psi in psi_list]
    mean_p = torch.stack(probs, dim=0).mean(dim=0).clamp_min(eps)
    return float(-(mean_p * mean_p.log()).sum().item())


def pairwise_amplitude_spread(psi_list: List[torch.Tensor]) -> float:
    """Max pairwise Euclidean distance between amplitude vectors (proportional to
    the Hellinger distance: ||psi_i - psi_j||_2 = sqrt(2) * H(P_i, P_j))."""
    n = len(psi_list)
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.norm(psi_list[i] - psi_list[j], p=2).item()
            max_d = max(max_d, d)
    return max_d


def divergence_gate(psi_list: List[torch.Tensor], thresholds: GateThresholds) -> GateDecision:
    if len(psi_list) < 2:
        raise ValueError("divergence_gate needs at least 2 agents to compare.")
    H = mean_entropy(psi_list)
    D = pairwise_amplitude_spread(psi_list)
    agree = (H < thresholds.entropy) and (D < thresholds.divergence)
    return GateDecision(agree=agree, entropy=H, divergence=D)
