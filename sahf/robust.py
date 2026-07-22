"""
Stage 6 — Outlier Check, and Stage 7 — Escalation Tier (Riemannian Weiszfeld
geometric median).

Known limitation (see HISTORY.md and ARCHITECTURE.md): the geometric median's
breakdown-point guarantee ("up to just under half the agents can be adversarial")
only means something with N >= 3 agents. The default fast-version config runs
exactly 2 real agents, so this tier is exercised and unit-tested here, but with
only 2 agents there is no honest majority to fall back on if one disagrees — it
degrades to "trust whichever agent is closer to the fused mean". Add a 3rd
(real or Mock) agent to get the actual robustness guarantee.
"""

from typing import List, Optional, Tuple

import torch


def detect_outliers(psi_list: List[torch.Tensor], psi_star: torch.Tensor,
                     mad_multiplier: float = 3.0) -> Tuple[List[bool], List[float]]:
    """
    Flags agents whose amplitude vector sits far from the fused mean, using a
    median-absolute-deviation threshold (robust to the outlier itself skewing the
    statistic, unlike a mean/std-based threshold).
    """
    dists = torch.tensor([torch.norm(psi - psi_star, p=2).item() for psi in psi_list])
    median = dists.median()
    mad = (dists - median).abs().median().clamp_min(1e-8)
    threshold = median + mad_multiplier * mad
    outlier_mask = (dists > threshold).tolist()
    return outlier_mask, dists.tolist()


def weiszfeld_geometric_median(psi_list: List[torch.Tensor], weights: Optional[List[float]] = None,
                                max_iter: int = 50, tol: float = 1e-6,
                                eps: float = 1e-8) -> Tuple[torch.Tensor, int]:
    """
    Iteratively reweighted averaging on the sphere: at each round, weight each
    agent inversely by its current geodesic (approximated here by chordal)
    distance from the running estimate, take the weighted mean, and re-project
    onto the unit sphere. Converges to the point minimizing total distance to all
    agents — breakdown point 1/2, vs. 0 for the plain mean.
    """
    n = len(psi_list)
    if weights is None:
        weights = [1.0] * n

    y = sum(psi_list) / n
    y = y / torch.norm(y, p=2).clamp_min(eps)

    iterations_run = 0
    for iteration in range(max_iter):
        iterations_run = iteration + 1
        dists = torch.tensor([torch.norm(y - psi, p=2).clamp_min(eps).item() for psi in psi_list])
        inv_dists = torch.tensor([w / d for w, d in zip(weights, dists.tolist())])
        weight_sum = inv_dists.sum().clamp_min(eps)

        y_new = sum(inv_dists[i].item() * psi_list[i] for i in range(n)) / weight_sum
        y_new = y_new / torch.norm(y_new, p=2).clamp_min(eps)

        shift = torch.norm(y_new - y, p=2).item()
        y = y_new
        if shift < tol:
            break

    return y, iterations_run
