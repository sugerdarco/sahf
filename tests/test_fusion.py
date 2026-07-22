import torch

from sahf.amplitude import softmax_to_amplitude
from sahf.fusion import fast_mean_fusion


def test_fusion_of_identical_vectors_returns_same_vector():
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
    psi = softmax_to_amplitude(logits)[0]
    fused = fast_mean_fusion([psi, psi.clone(), psi.clone()])
    assert torch.allclose(fused, psi, atol=1e-5)


def test_fusion_output_is_unit_norm():
    a = softmax_to_amplitude(torch.tensor([[1.0, 0.0, 0.0]]))[0]
    b = softmax_to_amplitude(torch.tensor([[0.0, 1.0, 0.0]]))[0]
    fused = fast_mean_fusion([a, b])
    assert torch.allclose(torch.norm(fused, p=2), torch.tensor(1.0), atol=1e-5)


def test_fusion_rejects_mismatched_weight_count():
    a = softmax_to_amplitude(torch.tensor([[1.0, 0.0]]))[0]
    b = softmax_to_amplitude(torch.tensor([[0.0, 1.0]]))[0]
    try:
        fast_mean_fusion([a, b], weights=[0.5, 0.3, 0.2])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_weighted_fusion_favors_higher_weight_agent():
    a = softmax_to_amplitude(torch.tensor([[10.0, -10.0]]))[0]  # confident token 0
    b = softmax_to_amplitude(torch.tensor([[-10.0, 10.0]]))[0]  # confident token 1
    fused_favor_a = fast_mean_fusion([a, b], weights=[0.9, 0.1])
    fused_favor_b = fast_mean_fusion([a, b], weights=[0.1, 0.9])
    probs_a = fused_favor_a**2
    probs_b = fused_favor_b**2
    assert probs_a[0] > probs_a[1]
    assert probs_b[1] > probs_b[0]
