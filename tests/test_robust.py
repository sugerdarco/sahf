import torch

from sahf.amplitude import softmax_to_amplitude
from sahf.fusion import fast_mean_fusion
from sahf.robust import detect_outliers, weiszfeld_geometric_median


def make_amplitude(vocab_size, winner, strength=10.0, noise_seed=0):
    g = torch.Generator().manual_seed(noise_seed)
    logits = torch.randn(1, vocab_size, generator=g) * 0.05
    logits[0, winner] += strength
    return softmax_to_amplitude(logits)[0]


def test_outlier_check_flags_the_odd_agent_out():
    honest = [make_amplitude(30, winner=3, noise_seed=i) for i in range(4)]
    poisoned = make_amplitude(30, winner=25, noise_seed=99)
    psi_list = honest + [poisoned]

    psi_star = fast_mean_fusion(psi_list)
    outlier_mask, dists = detect_outliers(psi_list, psi_star, mad_multiplier=3.0)

    assert outlier_mask[-1] is True, "the poisoned agent should be flagged"
    assert sum(outlier_mask[:-1]) == 0, "honest agents should not be flagged"


def test_geometric_median_beats_plain_mean_under_one_poisoned_agent():
    """
    The core claim behind Stage 7: with an honest majority (4 honest, 1
    poisoned), the geometric median should land much closer to the honest
    consensus than the plain arithmetic mean does.
    """
    honest_winner = 3
    honest = [make_amplitude(30, winner=honest_winner, noise_seed=i) for i in range(4)]
    poisoned = make_amplitude(30, winner=25, noise_seed=99, strength=20.0)
    psi_list = honest + [poisoned]

    honest_reference = fast_mean_fusion(honest)  # what we'd get with zero poisoning

    plain_mean = fast_mean_fusion(psi_list)
    median, n_iter = weiszfeld_geometric_median(psi_list)

    dist_mean_to_honest = torch.norm(plain_mean - honest_reference, p=2).item()
    dist_median_to_honest = torch.norm(median - honest_reference, p=2).item()

    assert dist_median_to_honest < dist_mean_to_honest
    # and the median should actually pick the honest-majority token as top-1
    assert int(torch.argmax(median**2).item()) == honest_winner
    assert n_iter >= 1


def test_geometric_median_converges_quickly_when_all_agents_agree():
    psi_list = [make_amplitude(20, winner=7, noise_seed=i) for i in range(5)]
    median, n_iter = weiszfeld_geometric_median(psi_list, max_iter=50, tol=1e-6)
    assert n_iter < 50, "should converge well before the iteration cap on easy inputs"
    assert int(torch.argmax(median**2).item()) == 7
