"""
Stage 8 / part 1 -- vocabulary extraction.

The byte-prefix tree is only meaningful if every agent's tokens are expressed in
the SAME base alphabet: raw bytes. Getting `token id -> bytes` right is the part
that silently corrupts everything downstream, because tokenizers store their
vocab in a *display* form, not in bytes:

  byte-level BPE (GPT-2, Llama-3, Qwen, Mistral-v3)
      "Ġthe"        -> b" the"      via the bytes_to_unicode() permutation
  sentencepiece    (Llama-2, Gemma, T5)
      "\u2581the"   -> b" the"      via the U+2581 meta-space
      "<0x0A>"      -> b"\n"        byte-fallback tokens
  wordpiece        (BERT)
      "##ing"       -> b"ing"       continuation marker

A wrong scheme does not raise -- it produces a plausible tree that mis-aligns
agents by exactly one leading space, which is the single most common token in
English. Always run `VocabSpec.verify()` after building one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# GPT-2 byte <-> unicode permutation
# --------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte->printable-unicode map (verbatim from the original repo)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


BYTE_ENCODER: dict[int, str] = bytes_to_unicode()
BYTE_DECODER: dict[str, int] = {v: k for k, v in BYTE_ENCODER.items()}

META_SPACE = "\u2581"  # sentencepiece leading-space marker


# --------------------------------------------------------------------------
# per-scheme decoders: token display string -> raw bytes
# --------------------------------------------------------------------------


def decode_bytelevel(token: str) -> bytes:
    return bytes(BYTE_DECODER[ch] for ch in token)


def decode_sentencepiece(token: str) -> bytes:
    # byte-fallback tokens: <0x41>
    if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
        try:
            return bytes([int(token[3:5], 16)])
        except ValueError:
            pass
    return token.replace(META_SPACE, " ").encode("utf-8")


def decode_wordpiece(token: str) -> bytes:
    if token.startswith("##"):
        return token[2:].encode("utf-8")
    # WordPiece drops the space; a leading space is the usual reconstruction
    return (" " + token).encode("utf-8")


def decode_identity(token: str) -> bytes:
    return token.encode("utf-8")


SCHEMES: dict[str, Callable[[str], bytes]] = {
    "byte_level": decode_bytelevel,
    "sentencepiece": decode_sentencepiece,
    "wordpiece": decode_wordpiece,
    "identity": decode_identity,
}


#: characters that appear ONLY in byte-level display forms -- the images of
#: bytes whose printable representation differs from the byte itself (space ->
#: 'Ġ', newline -> 'Ċ', and so on). Their presence is a positive fingerprint of
#: byte-level BPE that no other scheme produces.
BYTE_LEVEL_MARKERS: set[str] = {BYTE_ENCODER[b] for b in range(256) if BYTE_ENCODER[b] != chr(b)}


def detect_scheme(tokens: Iterable[str]) -> str:
    """Guess the display scheme from a sample of vocabulary strings.

    Order matters, and the obvious order is wrong. Testing for the WordPiece
    ``##`` prefix first misclassifies byte-level vocabularies, because ``##`` is
    an ordinary byte-level token in any vocab trained on markdown or code -- and
    the resulting mis-decode shifts every token by a leading space.

    So the positive byte-level fingerprint is checked first: marker characters
    such as 'Ġ' that only byte-level display forms can contain.
    """
    sample = [t for t in tokens if t]
    if not sample:
        return "identity"

    # 1. sentencepiece: meta-space or byte-fallback tokens are unambiguous
    if any(META_SPACE in t for t in sample):
        return "sentencepiece"
    if any(len(t) == 6 and t.startswith("<0x") and t.endswith(">") for t in sample):
        return "sentencepiece"

    # 2. byte-level: marker characters no other scheme emits
    if any(any(ch in BYTE_LEVEL_MARKERS for ch in t) for t in sample):
        return "byte_level"

    # 3. wordpiece: needs a real population of continuation tokens, not one "##"
    hashed = sum(1 for t in sample if t.startswith("##"))
    if hashed / len(sample) > 0.01:
        return "wordpiece"

    # 4. byte-level fallback: vocab closed under the permutation's charset
    covered = sum(1 for t in sample if all(ch in BYTE_DECODER for ch in t))
    if covered / len(sample) > 0.98:
        return "byte_level"
    return "identity"


# --------------------------------------------------------------------------
# VocabSpec
# --------------------------------------------------------------------------


@dataclass
class VocabSpec:
    """One agent's vocabulary, expressed in the shared byte alphabet.

    Attributes
    ----------
    name        : label for diagnostics
    token_bytes : token id -> raw bytes; ``None`` for ids with no byte image
                  (special tokens, holes in the id space, added-token slots)
    special_ids : ids deliberately excluded from the byte base space
    scheme      : which display->bytes decoder was used
    """

    name: str
    token_bytes: list[bytes | None]
    special_ids: set[int] = field(default_factory=set)
    scheme: str = "identity"
    prefix_space: bool = False  # set by verify(): tokenizer prepends a dummy space

    def __len__(self) -> int:
        return len(self.token_bytes)

    @property
    def size(self) -> int:
        return len(self.token_bytes)

    def byte_ids(self) -> list[int]:
        """Ids that carry a byte image (i.e. participate in the prefix tree)."""
        return [i for i, b in enumerate(self.token_bytes) if b is not None]

    # ---- constructors ---------------------------------------------------

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[int, bytes],
        *,
        name: str = "vocab",
        size: int | None = None,
        special_ids: Iterable[int] = (),
    ) -> VocabSpec:
        """Build directly from ``{token_id: raw_bytes}``. No tokenizer needed."""
        n = size if size is not None else (max(mapping) + 1 if mapping else 0)
        tb: list[bytes | None] = [None] * n
        for i, b in mapping.items():
            tb[i] = b
        return cls(name=name, token_bytes=tb, special_ids=set(special_ids), scheme="explicit")

    @classmethod
    def from_hf(
        cls,
        tokenizer,
        *,
        name: str | None = None,
        scheme: str = "auto",
        drop_special: bool = True,
        drop_added: bool = True,
    ) -> VocabSpec:
        """Build from a HuggingFace ``PreTrainedTokenizer(Fast)``.

        Handles vocab holes, added tokens and control tokens. Call ``verify()``
        afterwards -- scheme detection is a heuristic and a silent miss costs you
        one leading space on every token.

        ``drop_added`` excludes chat/control tokens registered in
        ``added_tokens_decoder`` (plus anything shaped like ``<|...|>``). Without
        it, a token such as ``<|im_start|>`` decodes to its literal bytes and is
        fused as though the model had predicted that text -- corrupting the
        consensus and letting one agent's chat template bleed into the shared
        byte space.
        """
        vocab = tokenizer.get_vocab()  # str -> id
        size = getattr(tokenizer, "vocab_size", 0)
        size = max(size, max(vocab.values()) + 1 if vocab else 0)

        specials = set(getattr(tokenizer, "all_special_ids", []) or [])
        if drop_added:
            specials |= set(getattr(tokenizer, "added_tokens_decoder", {}) or {})
            specials |= {
                tid
                for tok, tid in vocab.items()
                if len(tok) > 4 and tok.startswith("<|") and tok.endswith("|>")
            }

        if scheme == "auto":
            scheme = detect_scheme(list(vocab.keys())[:4000])
        decoder = SCHEMES[scheme]

        token_bytes: list[bytes | None] = [None] * size
        for tok, tid in vocab.items():
            if tid >= size:
                continue
            if (drop_special or drop_added) and tid in specials:
                continue
            try:
                token_bytes[tid] = decoder(tok)
            except (KeyError, UnicodeEncodeError):
                token_bytes[tid] = None

        return cls(
            name=name or getattr(tokenizer, "name_or_path", "hf-tokenizer"),
            token_bytes=token_bytes,
            special_ids=specials,
            scheme=scheme,
        )

    @classmethod
    def from_tiktoken(cls, encoding, *, name: str | None = None) -> VocabSpec:
        """Build from a ``tiktoken`` encoding -- already byte-native, no decoding."""
        ranks = encoding._mergeable_ranks  # bytes -> id
        size = encoding.n_vocab
        token_bytes: list[bytes | None] = [None] * size
        for b, tid in ranks.items():
            if tid < size:
                token_bytes[tid] = b
        specials = set(encoding._special_tokens.values())
        return cls(
            name=name or encoding.name,
            token_bytes=token_bytes,
            special_ids=specials,
            scheme="tiktoken",
        )

    # ---- verification ---------------------------------------------------

    def verify(self, tokenizer, samples: Sequence[str], *, verbose: bool = True) -> bool:
        """Round-trip check: encode text, concatenate our byte images, compare.

        This is the only reliable way to know the scheme was right -- a wrong
        scheme does not raise, it just shifts every token by a leading space.

        One legitimate mismatch is tolerated: SentencePiece and WordPiece
        tokenizers prepend a dummy space to the start of a sequence
        (``add_dummy_prefix`` / ``prepend_scheme="always"``). That shows up as a
        reconstruction one space longer than the input. It is a property of the
        tokenizer, not an extraction error, so it is recorded on
        ``self.prefix_space`` and does not fail the check. It matters only for
        prompt encoding, never for the next-token bytes Stage 8 fuses.
        """
        ok = True
        prefix_space = False
        for text in samples:
            ids = tokenizer.encode(text, add_special_tokens=False)
            parts = []
            for i in ids:
                b = self.token_bytes[i] if i < len(self.token_bytes) else None
                if b is None:
                    if i in self.special_ids:
                        continue
                    ok = False
                    if verbose:
                        print(f"[verify] {self.name}: id {i} has no byte image")
                    continue
                parts.append(b)
            got, want = b"".join(parts), text.encode("utf-8")
            if got == want:
                continue
            if got == b" " + want:
                prefix_space = True
                continue
            ok = False
            if verbose:
                print(f"[verify] {self.name}: mismatch\n  want={want!r}\n  got ={got!r}")

        self.prefix_space = prefix_space
        if verbose and ok:
            note = "  [tokenizer prepends a dummy space]" if prefix_space else ""
            print(f"[verify] {self.name}: OK ({self.scheme}) on {len(samples)} samples{note}")
        return ok
