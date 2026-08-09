import torch

from sahf.agents import MockAgent, PoisonedAgentWrapper


def test_poison_wrapper_invert_mode_negates_logits():
    honest = MockAgent("honest", vocab_size=20, seed=1, bias_token=5, bias_strength=10.0)
    wrapped = PoisonedAgentWrapper(honest, mode="invert")
    input_ids = torch.tensor([[1, 2, 3]])

    # Same underlying agent, called with a fresh copy so its own internal RNG
    # state advances identically either way — compare against a second,
    # independently-seeded but identically-configured MockAgent instead of
    # re-calling `honest` (which would advance its generator).
    reference = MockAgent("honest", vocab_size=20, seed=1, bias_token=5, bias_strength=10.0)
    expected = -reference.next_logits(input_ids)
    actual = wrapped.next_logits(input_ids)
    assert torch.allclose(actual, expected)


def test_poison_wrapper_preserves_identity_metadata():
    honest = MockAgent("honest-3", vocab_size=32)
    wrapped = PoisonedAgentWrapper(honest, mode="uniform_noise")
    assert wrapped.vocab_size == 32
    assert wrapped.tokenizer is None
    assert "honest-3" in wrapped.name
    assert "uniform_noise" in wrapped.name


def test_poison_wrapper_rejects_unknown_mode():
    honest = MockAgent("honest", vocab_size=10)
    try:
        PoisonedAgentWrapper(honest, mode="not_a_real_mode")
        assert False, "expected ValueError"
    except ValueError:
        pass
