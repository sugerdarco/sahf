"""Byte extraction against REAL trained tokenizers.

The synthetic vocabularies in `test_stage8.py` cannot catch a wrong display
scheme, because they are already bytes. These tests train a byte-level BPE, a
WordPiece and a Unigram/Metaspace tokenizer from scratch and round-trip every
one of them, which is the only way to know `VocabSpec` decodes real vocabularies
correctly.

Skipped when the optional `tokenizers` package is absent -- the core library
depends on numpy alone.
"""

from __future__ import annotations

import pytest

from sahf.sheaf import BytePrefixTree, SheafReconciler, VocabSpec

tokenizers = pytest.importorskip("tokenizers")

from tokenizers import decoders, models, pre_tokenizers, trainers

CORPUS = [
    "The capital of France is Paris, a city on the Seine.",
    "The capital of Japan is Tokyo, the largest city in the world.",
    "Paris is famous for the Louvre and the Eiffel Tower.",
    "Tokyo is famous for Shibuya crossing and its railways.",
    "the cat sat on the mat while the dog slept",
    "a quick brown fox jumps over the lazy dog again and again",
    "machine learning models predict the next token in a sequence",
    "distributed agents must agree on a shared vocabulary of bytes",
] * 40

SAMPLES = [
    "The capital of France is Paris",
    "the cat sat on the mat",
    "distributed agents must agree",
]


class Shim:
    """Minimal HuggingFace-tokenizer surface over a raw `tokenizers.Tokenizer`."""

    def __init__(self, tok, name):
        self._tok = tok
        self.name_or_path = name
        self.all_special_ids: list[int] = []
        self.added_tokens_decoder: dict[int, str] = {}

    def get_vocab(self):
        return self._tok.get_vocab()

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text).ids


def _bpe():
    tok = tokenizers.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        CORPUS,
        trainers.BpeTrainer(
            vocab_size=500,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return Shim(tok, "bpe")


def _wordpiece():
    tok = tokenizers.Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(
        CORPUS,
        trainers.WordPieceTrainer(vocab_size=500, special_tokens=["[UNK]"], show_progress=False),
    )
    return Shim(tok, "wordpiece")


def _unigram():
    tok = tokenizers.Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.train_from_iterator(
        CORPUS,
        trainers.UnigramTrainer(
            vocab_size=500, unk_token="[UNK]", special_tokens=["[UNK]"], show_progress=False
        ),
    )
    return Shim(tok, "unigram")


@pytest.fixture(scope="module")
def trained():
    return {"bpe": _bpe(), "wordpiece": _wordpiece(), "unigram": _unigram()}


@pytest.mark.parametrize(
    "kind,expected_scheme",
    [("bpe", "byte_level"), ("wordpiece", "wordpiece"), ("unigram", "sentencepiece")],
)
def test_scheme_detected_and_bytes_round_trip(trained, kind, expected_scheme):
    shim = trained[kind]
    vs = VocabSpec.from_hf(shim, name=kind)
    assert vs.scheme == expected_scheme
    assert vs.verify(shim, SAMPLES, verbose=False), f"{kind}: byte round-trip failed"


def test_cross_tokenizer_reconciliation_on_real_vocabs(trained):
    """Two real tokenizers of different families must glue into one section."""
    import numpy as np

    a = VocabSpec.from_hf(trained["bpe"], name="bpe")
    b = VocabSpec.from_hf(trained["unigram"], name="unigram")

    def peaked(vs, text, sharpness=0.9):
        target = text.encode()
        best, blen = None, 0
        for i, tok in enumerate(vs.token_bytes):
            if tok and target.startswith(tok) and len(tok) > blen:
                best, blen = i, len(tok)
        assert best is not None, f"{vs.name} cannot start {text!r}"
        p = np.zeros(vs.size)
        p[best] = sharpness
        rest = [i for i, t in enumerate(vs.token_bytes) if t and i != best][:50]
        for i in rest:
            p[i] = (1 - sharpness) / len(rest)
        return p / p.sum()

    tree = BytePrefixTree.from_vocabs([a, b])
    sec = SheafReconciler(fusion="auto", eps=1e-4).reconcile(
        tree, [peaked(a, " Paris"), peaked(b, " Paris")]
    )

    assert sec.conditionals, "no nodes were fused"
    best = sec.decode(min_support=1, max_len=8)[0][0]
    assert b" Pa".startswith(best[:3]) or best.startswith(b" Pa")

    for agent in (0, 1):
        q, rep = sec.project_to_vocab(agent)
        assert abs(q.sum() - 1.0) < 1e-9
        assert rep.unreachable >= 0.0
