"""
Stage 8's numpy transcriptions must agree with the repo's Stages 1/5/6/7.

Stage 8 runs Stages 5, 6 and 7 per byte-prefix-tree node -- hundreds of times per
token, on short vectors -- where torch's per-call overhead dominates the actual
arithmetic. They are therefore transcribed into numpy inside
`sahf.sheaf.reconciler` rather than called.

Transcription is a duplication risk: `sahf.fusion` / `sahf.robust` could be tuned
later and Stage 8 would silently keep the old semantics. These tests exist to make
that a failing build instead. They are the reason the transcription is acceptable
at all, so they should be treated as load-bearing, not as coverage padding.

If one of these fails after an intentional change to Stages 5/6/7, the fix is to
port the change into `sahf/sheaf/reconciler.py` -- not to loosen the tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sahf.amplitude import softmax_to_amplitude
from sahf.fusion import fast_mean_fusion
from sahf.robust import detect_outliers as torch_detect_outliers
from sahf.robust import weiszfeld_geometric_median
from sahf.sheaf.reconciler import detect_outliers as np_detect_outliers
from sahf.sheaf.reconciler import geometric_median, mean_fuse, to_amplitude

TOL = 1e-6


def _random_psis(n_agents: int, dim: int, seed: int):
    """n agents' amplitude vectors over a `dim`-symbol alphabet, as numpy + torch."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n_agents, dim)) * 2.0
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    psis = np.sqrt(probs)
    return psis, [torch.from_numpy(p.copy()) for p in psis]


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stage1_amplitude_parity(seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(1, 512)) * 3.0
    t_psi = softmax_to_amplitude(torch.from_numpy(logits))[0].numpy()

    p = np.exp(logits[0] - logits[0].max())
    p /= p.sum()
    n_psi = to_amplitude(p)

    assert np.allclose(t_psi, n_psi, atol=TOL)


# --------------------------------------------------------------------------
# Stage 5
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_agents,dim,seed", [(2, 256, 0), (3, 256, 1), (5, 64, 2), (8, 256, 3)])
def test_stage5_mean_fusion_parity(n_agents, dim, seed):
    psis, t_psis = _random_psis(n_agents, dim, seed)
    w = np.full(n_agents, 1.0 / n_agents)

    t_out = fast_mean_fusion(t_psis, w.tolist()).numpy()
    n_out = mean_fuse(psis, w)

    assert np.allclose(t_out, n_out, atol=TOL)


def test_stage5_weighted_parity():
    psis, t_psis = _random_psis(3, 256, 7)
    w = np.array([0.5, 0.3, 0.2])
    assert np.allclose(fast_mean_fusion(t_psis, w.tolist()).numpy(), mean_fuse(psis, w), atol=TOL)


# --------------------------------------------------------------------------
# Stage 6 -- the rule Stage 8 originally got wrong
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_agents,seed", [(3, 0), (4, 1), (5, 2), (8, 3)])
@pytest.mark.parametrize("mad_multiplier", [1.5, 3.0, 5.0])
def test_stage6_outlier_mask_parity(n_agents, seed, mad_multiplier):
    psis, t_psis = _random_psis(n_agents, 256, seed)
    w = np.full(n_agents, 1.0 / n_agents)
    psi_star = mean_fuse(psis, w)

    t_mask, t_d = torch_detect_outliers(t_psis, torch.from_numpy(psi_star), mad_multiplier)
    n_mask, n_d = np_detect_outliers(psis, psi_star, mad_multiplier)

    assert list(n_mask) == list(t_mask)
    assert np.allclose(np.asarray(t_d), n_d, atol=TOL)


def test_stage6_flags_a_planted_adversary_identically():
    """The case Stage 6 exists for: 3 honest agents and 1 confidently wrong one."""
    honest, _ = _random_psis(1, 256, 11)
    adversary = np.zeros(256)
    adversary[200] = 1.0
    psis = np.stack([honest[0], honest[0], honest[0], adversary])
    t_psis = [torch.from_numpy(p.copy()) for p in psis]
    w = np.full(4, 0.25)
    psi_star = mean_fuse(psis, w)

    t_mask, _ = torch_detect_outliers(t_psis, torch.from_numpy(psi_star), 3.0)
    n_mask, _ = np_detect_outliers(psis, psi_star, 3.0)

    assert list(n_mask) == list(t_mask)
    assert n_mask[3], "the adversary should be flagged"
    assert not any(n_mask[:3]), "honest agents should not be flagged"


def test_stage6_identical_agents_flag_nobody():
    """MAD is floored at 1e-8 in both implementations; that must not misfire."""
    psi = np.zeros(16)
    psi[3] = 1.0
    psis = np.stack([psi, psi, psi])
    t_psis = [torch.from_numpy(p.copy()) for p in psis]
    star = mean_fuse(psis, np.full(3, 1 / 3))

    t_mask, _ = torch_detect_outliers(t_psis, torch.from_numpy(star), 3.0)
    n_mask, _ = np_detect_outliers(psis, star, 3.0)
    assert list(n_mask) == list(t_mask) == [False, False, False]


# --------------------------------------------------------------------------
# Stage 7
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_agents,dim,seed", [(3, 256, 0), (4, 256, 1), (5, 64, 2), (8, 256, 3)])
def test_stage7_geometric_median_parity(n_agents, dim, seed):
    psis, t_psis = _random_psis(n_agents, dim, seed)
    w = np.ones(n_agents)

    t_out, _ = weiszfeld_geometric_median(t_psis, w.tolist())
    n_out = geometric_median(psis, w)

    assert np.allclose(t_out.numpy(), n_out, atol=TOL)


def test_stage7_parity_with_an_adversary_present():
    """Parity has to hold in the regime Stage 7 is actually invoked in."""
    honest, _ = _random_psis(1, 256, 21)
    adversary = np.zeros(256)
    adversary[100] = 1.0
    psis = np.stack([honest[0], honest[0], honest[0], adversary])
    t_psis = [torch.from_numpy(p.copy()) for p in psis]
    w = np.ones(4)

    t_out, _ = weiszfeld_geometric_median(t_psis, w.tolist())
    n_out = geometric_median(psis, w)
    assert np.allclose(t_out.numpy(), n_out, atol=TOL)

    # and it must still beat the plain mean at suppressing the adversary
    mean_p = mean_fuse(psis, w / w.sum()) ** 2
    gm_p = n_out**2
    assert gm_p[100] < mean_p[100]


def test_stage7_weighted_parity():
    psis, t_psis = _random_psis(4, 256, 31)
    w = np.array([1.0, 2.0, 0.5, 1.5])
    t_out, _ = weiszfeld_geometric_median(t_psis, w.tolist())
    assert np.allclose(t_out.numpy(), geometric_median(psis, w), atol=TOL)


# --------------------------------------------------------------------------
# Stage 2 is reused directly, not transcribed -- assert that stays true
# --------------------------------------------------------------------------


def test_stage2_is_imported_not_reimplemented():
    """`SheafOrchestrator` must call the repo's gate helpers, not its own copy."""
    import inspect

    from sahf.sheaf import orchestrator

    src = inspect.getsource(orchestrator)
    assert "from ..gate import" in src
    assert "mean_entropy" in src and "pairwise_amplitude_spread" in src
    # no local redefinition sneaking in
    assert "def mean_entropy" not in src
    assert "def pairwise_amplitude_spread" not in src


def test_per_node_spread_matches_the_repo_gate_helper():
    """The gate's per-node spread must be `pairwise_amplitude_spread`, per node.

    `SheafOrchestrator` maximises this over every active node rather than calling
    the torch helper node by node (which would dominate the gate's cost). This
    pins the numpy computation to the repo's definition.
    """
    from sahf.gate import pairwise_amplitude_spread

    psis, t_psis = _random_psis(4, 256, 5)
    diff = psis[:, None, :] - psis[None, :, :]
    numpy_spread = float(np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).max())

    assert numpy_spread == pytest.approx(pairwise_amplitude_spread(t_psis), abs=TOL)
