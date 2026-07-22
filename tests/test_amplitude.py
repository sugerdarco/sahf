import torch

from sahf.amplitude import softmax_to_amplitude


def test_amplitude_lands_on_unit_sphere():
    logits = torch.tensor([[2.0, 0.5, -1.0, 3.0, 0.0]])
    psi = softmax_to_amplitude(logits)
    norm_sq = (psi**2).sum(dim=-1)
    assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-5)


def test_amplitude_is_nonnegative():
    logits = torch.randn(4, 50)
    psi = softmax_to_amplitude(logits)
    assert (psi >= 0).all()


def test_amplitude_matches_sqrt_softmax_manually():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    expected = torch.softmax(logits, dim=-1).sqrt()
    psi = softmax_to_amplitude(logits)
    assert torch.allclose(psi, expected, atol=1e-6)


def test_amplitude_handles_extreme_logits_without_nan():
    logits = torch.tensor([[100.0, -100.0, 0.0]])
    psi = softmax_to_amplitude(logits)
    assert not torch.isnan(psi).any()
