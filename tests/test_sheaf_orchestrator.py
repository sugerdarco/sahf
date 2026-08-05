"""
SheafOrchestrator end-to-end, offline.

Mirrors `tests/test_orchestrator_mock.py` for the mismatched-tokenizer case:
no models, no internet, no GPU. Agents here are `StaticAgent`s with genuinely
different vocabularies, which is the condition Stage 8 exists for and the one
`MockAgent` cannot represent (it has no tokenizer at all).
"""

from __future__ import annotations

import numpy as np
import pytest

from sahf.gate import GateThresholds
from sahf.sheaf import SheafOrchestrator, StaticAgent, VocabSpec


def vocab(name: str, toks: list[bytes]) -> VocabSpec:
    return VocabSpec.from_mapping({i: b for i, b in enumerate(toks)}, name=name)


def dist(size: int, weights: dict[int, float]) -> np.ndarray:
    p = np.zeros(size)
    for i, w in weights.items():
        p[i] = w
    return p / p.sum()


def covering_vocab(name: str, target: bytes, max_len: int) -> VocabSpec:
    """A vocabulary that fully covers `target`, capped at `max_len` bytes.

    Stands in for tokenizers with different compression ratios: the coarse agent
    sees several bytes ahead per forward pass, the fine one only a single byte.
    """
    toks = sorted({target[i : i + n] for i in range(len(target)) for n in range(1, max_len + 1)})
    return vocab(name, list(toks))


def lead(vs: VocabSpec, remaining: bytes) -> np.ndarray:
    """Agent predicts its own longest token matching the remaining target."""
    best, blen = None, 0
    for i, tok in enumerate(vs.token_bytes):
        if tok and remaining.startswith(tok) and len(tok) > blen:
            best, blen = i, len(tok)
    p = np.full(vs.size, 0.02 / max(vs.size - 1, 1))
    if best is not None:
        p[best] = 0.98
    return p / p.sum()


TARGET = b" Paris is the capital"


class LazyAgent(StaticAgent):
    """Responds to ANY context by predicting its longest token continuing TARGET.

    `StaticAgent` needs every context registered up front, which cannot work with
    `generate()`: the contexts that arise depend on where the ensemble's joint
    horizon falls, and that is what is being tested. This computes on demand.
    """

    def __init__(self, name, vs):
        super().__init__(name, vs)
        self._vs = vs

    def next_token_probs(self, context: str):
        produced = context.encode("utf-8")
        remaining = TARGET[len(produced) :] or TARGET
        return lead(self._vs, remaining)

    def stop_probability(self, context: str) -> float:
        return 1.0 if context.encode("utf-8") == TARGET else 0.0


@pytest.fixture
def agents():
    specs = [
        covering_vocab("coarse", TARGET, 6),
        covering_vocab("medium", TARGET, 3),
        covering_vocab("fine", TARGET, 1),
    ]
    return [StaticAgent(v.name, v) for v in specs], specs


def _prime(agents, specs, produced: bytes):
    ctx = produced.decode("utf-8", errors="replace")
    remaining = TARGET[len(produced) :]
    for ag, vs in zip(agents, specs):
        ag.set(ctx, lead(vs, remaining))


# --------------------------------------------------------------------------


def test_rejects_single_agent():
    v = vocab("a", [b"x"])
    with pytest.raises(ValueError):
        SheafOrchestrator([StaticAgent("a", v)], GateThresholds())


def test_step_emits_consensus_and_records_gate(agents):
    ags, specs = agents
    _prime(ags, specs, b"")
    orch = SheafOrchestrator(ags, GateThresholds(), k=32)

    chunk, record, res = orch.step("", 0)

    assert chunk and TARGET.startswith(chunk)
    assert record["path"] in ("A_fast_passthrough", "B_fusion_pipeline")
    # the fast version's field names must survive, for existing log tooling
    for key in ("step", "entropy", "divergence", "path", "escalated"):
        assert key in record
    # and the Stage 8 specifics
    for key in ("tree_nodes", "fused_nodes", "coverage", "stop_mass", "n_bytes"):
        assert key in record
    assert res.tree_stats.n_nodes > 0


def test_generation_reconstructs_across_three_granularities(agents):
    """Three different compression ratios, one exact reconstruction."""
    ags, specs = agents
    orch = SheafOrchestrator(ags, GateThresholds(), k=32)

    produced = b""
    for _ in range(12):
        if len(produced) >= len(TARGET):
            break
        _prime(ags, specs, produced)
        chunk, _record, _res = orch.step(produced.decode("utf-8", errors="replace"), 0)
        if not chunk:
            break
        produced += chunk

    assert produced == TARGET, f"got {produced!r}"


def test_generate_returns_text_not_ids():
    """Unlike FusionOrchestrator, there is no shared id sequence to return."""
    specs = [
        covering_vocab("coarse", TARGET, 6),
        covering_vocab("medium", TARGET, 3),
        covering_vocab("fine", TARGET, 1),
    ]
    ags = [LazyAgent(v.name, v) for v in specs]
    orch = SheafOrchestrator(ags, GateThresholds(), k=32, max_new_bytes=len(TARGET))

    text, history = orch.generate("")

    assert isinstance(text, str)
    assert history and all("step" in h for h in history)
    assert text == TARGET.decode(), f"got {text!r}"
    assert sum(h["n_bytes"] for h in history if h.get("n_bytes")) >= len(TARGET)


def test_generate_halts_on_consensus_stop():
    v = [vocab(f"a{i}", [b"ab", b"cd"]) for i in range(3)]
    ags = [StaticAgent(s.name, s) for s in v]
    for ag in ags:
        ag.set("", dist(2, {0: 1.0}), stop=0.0)
        ag.set("ab", dist(2, {1: 1.0}), stop=0.99)

    orch = SheafOrchestrator(ags, GateThresholds(), k=8, min_support=1, max_new_bytes=64)
    text, history = orch.generate("")

    assert text == "ab"
    assert history[-1].get("stopped") == "eos_consensus"


def test_byte_gate_routes_agreement_to_path_a():
    """Unanimous agents should take Path A; a planted disagreement should not."""
    v = [vocab(f"a{i}", [b" yes", b" no"]) for i in range(3)]
    ags = [StaticAgent(s.name, s) for s in v]

    for ag in ags:
        ag.set("agree", dist(2, {0: 0.999, 1: 0.001}))
    orch = SheafOrchestrator(ags, GateThresholds(), k=8)
    _, rec_agree, _ = orch.step("agree", 0)

    ags[0].set("split", dist(2, {0: 0.99, 1: 0.01}))
    ags[1].set("split", dist(2, {0: 0.99, 1: 0.01}))
    ags[2].set("split", dist(2, {0: 0.01, 1: 0.99}))
    _, rec_split, _ = orch.step("split", 1)

    assert rec_agree["divergence"] < rec_split["divergence"]
    assert rec_agree["path"] == "A_fast_passthrough"
    assert rec_split["path"] == "B_fusion_pipeline"


def test_path_b_escalates_on_a_byzantine_agent():
    """Stage 6/7 must still fire inside Stage 8, per node."""
    v = [vocab(f"a{i}", [b" Paris", b" Berlin"]) for i in range(4)]
    ags = [StaticAgent(s.name, s) for s in v]
    for ag in ags[:3]:
        ag.set("ctx", dist(2, {0: 0.95, 1: 0.05}))
    ags[3].set("ctx", dist(2, {0: 0.02, 1: 0.98}))

    orch = SheafOrchestrator(ags, GateThresholds(), k=8, min_support=1)
    chunk, record, _ = orch.step("ctx", 0)

    assert record["path"] == "B_fusion_pipeline"
    assert record["escalated"], "Stage 6 should have flagged the byzantine agent"
    assert chunk.startswith(b" P"), "the honest majority should still win"


def test_upstream_agent_rejects_tokenizerless_mock():
    """MockAgent has no vocabulary, so it cannot participate in byte-level fusion."""
    from sahf.agents import MockAgent
    from sahf.sheaf import UpstreamAgent

    with pytest.raises(ValueError, match="tokenizer"):
        UpstreamAgent(MockAgent("mock", vocab_size=32))
