"""
The mismatched-tokenizer path, end to end, with REAL tokenizers and REAL torch
forward passes — no network, no GPU.

Every other Stage 8 test uses `StaticAgent` with synthetic byte vocabularies,
which does not exercise `UpstreamAgent` — the piece that actually bridges
`sahf.agents`-style agents to Stage 8 — encoding the shared context separately
per tokenizer, applying Stage 1, and dropping stop-token mass.

So this trains two genuinely different tokenizers (byte-level BPE and
Unigram/Metaspace, different algorithms and different vocabulary sizes), pairs
each with a small torch model over its own vocabulary, and drives the whole
Stage 8 orchestrator through them.

Skipped when the optional `tokenizers` package is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")

from tokenizers import decoders, models, pre_tokenizers, trainers

from sahf.gate import GateThresholds
from sahf.sheaf import SheafOrchestrator, UpstreamAgent, VocabSpec

CORPUS = [
    "The capital of France is Paris, a city on the Seine.",
    "The capital of Japan is Tokyo, the largest city in the world.",
    "Paris is famous for the Louvre and the Eiffel Tower.",
    "the cat sat on the mat while the dog slept",
    "a quick brown fox jumps over the lazy dog again and again",
    "distributed agents must agree on a shared vocabulary of bytes",
] * 40


class Encoding:
    def __init__(self, ids):
        self.input_ids = torch.tensor([ids], dtype=torch.long)


class TokenizerShim:
    """The slice of the HuggingFace tokenizer API that Stage 8 actually uses."""

    def __init__(self, tok, name):
        self._tok = tok
        self.name_or_path = name
        self.all_special_ids: list[int] = []
        self.added_tokens_decoder: dict[int, str] = {}
        self.eos_token_id = None

    def get_vocab(self):
        return self._tok.get_vocab()

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text).ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self._tok.encode(text).ids or [0]
        return Encoding(ids)


class TinyAgent:
    """Matches the `sahf.agents.HFAgent` interface: .name, .tokenizer, .next_logits.

    The "model" is a small deterministic embedding+linear stack over this agent's
    own vocabulary, biased toward whichever of ITS OWN tokens is the longest
    prefix of `goal`. Real weights are not the point — what is being tested is
    that agents with different vocabulary sizes and different segmentations can
    be reconciled at all.

    The bias deliberately does not track generation progress. Recovering progress
    would mean decoding `input_ids` back to text, and SentencePiece-family
    tokenizers prepend a dummy space to a sequence, so that decode is off by one
    byte for some agents and not others — an artefact of the test harness that
    would look like a Stage 8 failure. Reconstruction accuracy is covered exactly
    in `test_sheaf_orchestrator.py`; this file covers the plumbing.
    """

    def __init__(self, name, shim, vocab_spec, goal: bytes, seed: int):
        self.name = name
        self.tokenizer = shim
        self._vs = vocab_spec
        self._goal = goal
        self.vocab_size = vocab_spec.size
        g = torch.Generator().manual_seed(seed)
        self._emb = torch.nn.Embedding(vocab_spec.size, 16)
        self._head = torch.nn.Linear(16, vocab_spec.size)
        with torch.no_grad():
            self._emb.weight.normal_(0, 0.02, generator=g)
            self._head.weight.normal_(0, 0.02, generator=g)
            self._head.bias.zero_()

        self._best = None
        blen = 0
        for i, tok in enumerate(vocab_spec.token_bytes):
            if tok and goal.startswith(tok) and len(tok) > blen:
                self._best, blen = i, len(tok)

    @torch.no_grad()
    def next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self._emb(input_ids).mean(dim=1)
        logits = self._head(h)
        if self._best is not None:
            logits[0, self._best] += 12.0
        return logits


def _bpe_shim():
    tok = tokenizers.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        CORPUS,
        trainers.BpeTrainer(
            vocab_size=400,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return TokenizerShim(tok, "bpe-agent")


def _unigram_shim():
    tok = tokenizers.Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.train_from_iterator(
        CORPUS,
        trainers.UnigramTrainer(
            vocab_size=600, unk_token="[UNK]", special_tokens=["[UNK]"], show_progress=False
        ),
    )
    return TokenizerShim(tok, "unigram-agent")


GOAL = b" the capital of France"


@pytest.fixture(scope="module")
def mixed_agents():
    shims = [_bpe_shim(), _unigram_shim(), _bpe_shim()]
    names = ["bpe-A", "unigram-B", "bpe-C"]
    out = []
    for i, (shim, name) in enumerate(zip(shims, names)):
        vs = VocabSpec.from_hf(shim, name=name)
        out.append(UpstreamAgent(TinyAgent(name, shim, vs, GOAL, seed=i)))
    return out


# --------------------------------------------------------------------------


def test_vocabularies_really_do_differ(mixed_agents):
    """Guard the premise: if these matched, the test would prove nothing."""
    schemes = {a.name: a.vocab_spec().scheme for a in mixed_agents}
    sizes = {a.name: a.vocab_spec().size for a in mixed_agents}
    assert "byte_level" in schemes.values()
    assert "sentencepiece" in schemes.values()
    assert len(set(sizes.values())) > 1, f"expected differing vocab sizes, got {sizes}"

    # and they genuinely segment the same text differently
    segs = {a.name: tuple(a.tokenizer.encode(" the capital")) for a in mixed_agents}
    assert len(set(segs.values())) > 1


def test_byte_extraction_round_trips_for_every_agent(mixed_agents):
    """A mis-detected scheme shifts every token by a space and fails silently."""
    for agent in mixed_agents:
        assert agent.verify(["the capital of France", "a quick brown fox"]), agent.name


def test_upstream_agent_returns_a_valid_distribution(mixed_agents):
    """Stage 1 is applied via sahf.amplitude; the result must be a proper pmf."""
    for agent in mixed_agents:
        p = agent.next_token_probs("The capital of France is")
        assert p.shape == (agent.vocab_spec().size,)
        assert np.all(p >= 0)
        assert abs(p.sum() - 1.0) < 1e-9
        assert 0.0 <= agent.stop_probability("The capital of France is") <= 1.0


def test_agents_encode_the_same_context_with_their_own_tokenizer(mixed_agents):
    """The core difference from FusionOrchestrator: text in, not shared ids."""
    ctx = "The capital of France is"
    lengths = {a.name: len(a.tokenizer.encode(ctx)) for a in mixed_agents}
    assert len(set(lengths.values())) > 1, (
        f"agents produced identical token counts {lengths}; the mismatch this "
        "test exists to cover is not present"
    )
    for agent in mixed_agents:
        agent.next_token_probs(ctx)  # must not raise on any of them


def test_orchestrator_step_across_real_mismatched_tokenizers(mixed_agents):
    orch = SheafOrchestrator(mixed_agents, GateThresholds(), k=32, min_support=1)
    chunk, record, res = orch.step("", 0)

    assert chunk, "no consensus bytes produced"
    assert GOAL.startswith(chunk), (
        f"consensus {chunk!r} is not a prefix of the string all three agents "
        f"were biased toward ({GOAL!r})"
    )
    assert record["tree_nodes"] > 0
    assert record["fused_nodes"] > 0
    assert set(record["coverage"]) != {0.0}
    # every agent got a token-space projection back
    assert set(res.token_probs) == {a.name for a in mixed_agents}
    for name, q in res.token_probs.items():
        assert abs(q.sum() - 1.0) < 1e-9, name


def test_generation_runs_across_real_mismatched_tokenizers(mixed_agents):
    orch = SheafOrchestrator(
        mixed_agents, GateThresholds(), k=32, min_support=1, max_new_bytes=12
    )
    text, history = orch.generate("")

    assert history, "no steps taken"
    assert isinstance(text, str)
    assert "\ufffd" not in text, f"corrupted utf-8 in {text!r}"
    assert text, "generation produced nothing"
    # the byte budget must be respected (one final chunk may overshoot it)
    emitted = sum(h.get("n_bytes", 0) for h in history)
    assert emitted <= 12 + 16, emitted


def test_projection_back_is_usable_by_each_agent(mixed_agents):
    """The consensus must land on tokens each agent can actually emit."""
    orch = SheafOrchestrator(mixed_agents, GateThresholds(), k=32, min_support=1)
    _chunk, _record, res = orch.step("", 0)

    for agent in mixed_agents:
        q = res.token_probs[agent.name]
        top = int(np.argmax(q))
        assert agent.vocab_spec().token_bytes[top] is not None, (
            f"{agent.name}: projection put its mass on a token with no byte image"
        )


def test_distinct_tokenizer_guard_passes_on_mismatched_agents(mixed_agents):
    from sahf.sheaf import assert_distinct_tokenizers

    info = assert_distinct_tokenizers(mixed_agents)
    assert set(info) == {a.name for a in mixed_agents}


def test_distinct_tokenizer_guard_warns_when_stage8_is_pointless():
    """If everyone shares a tokenizer, Stage 8 is overhead and should say so."""
    from sahf.sheaf import assert_distinct_tokenizers

    shim = _bpe_shim()
    vs = VocabSpec.from_hf(shim, name="same")
    agents = [
        UpstreamAgent(TinyAgent(f"clone-{i}", shim, vs, GOAL, seed=i), name=f"clone-{i}")
        for i in range(3)
    ]
    with pytest.warns(UserWarning, match="share one tokenizer"):
        assert_distinct_tokenizers(agents)
    with pytest.raises(ValueError, match="share one tokenizer"):
        assert_distinct_tokenizers(agents, strict=True)
