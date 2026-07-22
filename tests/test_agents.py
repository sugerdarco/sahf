import torch

from sahf.agents import MockAgent, PoisonedAgentWrapper, assert_shared_vocab_size
from sahf.gate import GateThresholds
from sahf.logger import RunLogger
from sahf.orchestrator import FusionOrchestrator
import tempfile
import shutil


def test_assert_shared_vocab_size_passes_when_matching():
    agents = [MockAgent("a", vocab_size=50), MockAgent("b", vocab_size=50), MockAgent("c", vocab_size=50)]
    assert assert_shared_vocab_size(agents) == 50


def test_assert_shared_vocab_size_raises_when_mismatched():
    agents = [MockAgent("a", vocab_size=50), MockAgent("b", vocab_size=64)]
    try:
        assert_shared_vocab_size(agents)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "a=50" in str(e) and "b=64" in str(e)


def test_assert_shared_vocab_size_requires_at_least_two():
    try:
        assert_shared_vocab_size([MockAgent("a", vocab_size=50)])
        assert False, "expected ValueError"
    except ValueError:
        pass


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


def test_n3_orchestrator_survives_wrapped_poisoning_via_production_mechanism():
    """
    The integration test that actually closes the gap: 3 agents, one wrapped
    with the SAME PoisonedAgentWrapper class `run.py --poison-index` uses on
    real models (not a hand-crafted biased mock) — proving the honest majority
    still wins at N=3 through the real production code path, not just a
    synthetic unit test of the math in isolation.
    """
    tmp_out = tempfile.mkdtemp()
    try:
        honest_winner = 9
        honest_1 = MockAgent("honest-1", vocab_size=40, seed=1, bias_token=honest_winner, bias_strength=15.0)
        honest_2 = MockAgent("honest-2", vocab_size=40, seed=2, bias_token=honest_winner, bias_strength=15.0)
        target = MockAgent("honest-3", vocab_size=40, seed=3, bias_token=honest_winner, bias_strength=15.0)
        poisoned = PoisonedAgentWrapper(target, mode="invert")

        agents = [honest_1, honest_2, poisoned]
        assert_shared_vocab_size(agents)  # must pass before we even try to run

        thresholds = GateThresholds(entropy=2.0, divergence=0.01)  # force disagreement path
        logger = RunLogger(out_dir=tmp_out, run_label="poison_test")
        orch = FusionOrchestrator(agents, thresholds, max_new_tokens=5, logger=logger)

        prompt_ids = torch.tensor([[1, 2, 3]])
        _, history = orch.generate(prompt_ids)
        logger.close()

        assert all(h["path"] == "B_fusion_pipeline" for h in history)
        assert all(h["escalated"] for h in history), "inverted logits should read as a clear outlier"
        assert all(h["token_id"] == honest_winner for h in history), \
            "honest majority should win every step despite one wrapped-poisoned agent"
    finally:
        shutil.rmtree(tmp_out)
