import torch

from sahf.amplitude import softmax_to_amplitude
from sahf.gate import GateThresholds, divergence_gate


def confident_logits(vocab_size, winner, strength=10.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, vocab_size, generator=g) * 0.1
    logits[0, winner] += strength
    return logits


def test_identical_confident_agents_agree():
    logits = confident_logits(50, winner=5, seed=1)
    psi_list = [softmax_to_amplitude(logits)[0], softmax_to_amplitude(logits.clone())[0]]
    decision = divergence_gate(psi_list, GateThresholds(entropy=2.0, divergence=0.05))
    assert decision.agree is True
    assert decision.divergence < 1e-6


def test_agents_confident_about_different_tokens_disagree():
    logits_a = confident_logits(50, winner=5, seed=1)
    logits_b = confident_logits(50, winner=40, seed=2)
    psi_list = [softmax_to_amplitude(logits_a)[0], softmax_to_amplitude(logits_b)[0]]
    decision = divergence_gate(psi_list, GateThresholds(entropy=2.0, divergence=0.05))
    assert decision.agree is False
    assert decision.divergence > 0.05


def test_uniform_uncertain_agents_do_not_agree_even_if_similar():
    # both agents equally unsure (near-uniform) -> low divergence, but high entropy
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(4)
    logits_a = torch.randn(1, 50, generator=g1) * 0.01
    logits_b = torch.randn(1, 50, generator=g2) * 0.01
    psi_list = [softmax_to_amplitude(logits_a)[0], softmax_to_amplitude(logits_b)[0]]
    decision = divergence_gate(psi_list, GateThresholds(entropy=1.0, divergence=0.5))
    assert decision.entropy > 1.0  # near-uniform over 50 tokens -> entropy close to ln(50) ~ 3.9
    assert decision.agree is False


def test_requires_at_least_two_agents():
    logits = confident_logits(10, winner=0)
    psi_list = [softmax_to_amplitude(logits)[0]]
    try:
        divergence_gate(psi_list, GateThresholds())
        assert False, "expected ValueError"
    except ValueError:
        pass
