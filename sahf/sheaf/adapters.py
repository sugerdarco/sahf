"""
Stage 8 / part 4 -- getting the distributions out of the models.

An agent here is (its own tokenizer, its own next-token distribution). The
interface takes TEXT, never token ids: each agent tokenizes the shared context
with its own tokenizer, which is precisely the mismatch Stage 8 exists to fix.

`DirectHFAgent` is written against transformers/torch but imports them lazily, so the
rest of the package stays dependency-free (numpy only).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .vocab import VocabSpec


class Agent(Protocol):
    name: str

    def vocab_spec(self) -> VocabSpec: ...

    def next_token_probs(self, context: str) -> np.ndarray: ...

    def stop_probability(self, context: str) -> float:
        """Mass on end-of-text. Optional; agents without it never request a stop."""
        return 0.0


# --------------------------------------------------------------------------


class DirectHFAgent:
    """Wraps a HuggingFace causal LM + tokenizer directly as a Stage 8 agent.

    Inside this repo prefer `UpstreamAgent(sahf.agents.HFAgent(...))`, which
    reuses the repo's own agent wrapper and its Stage 1 amplitude conversion.
    This class exists for using Stage 8 standalone, without `sahf.agents`.

    Usage
    -----
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    >>> mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    >>> agent = DirectHFAgent("qwen", mdl, tok)
    >>> agent.vocab_spec().verify(tok, ["Hello world", " the cat sat"])
    """

    def __init__(
        self,
        name: str,
        model,
        tokenizer,
        *,
        temperature: float = 1.0,
        top_p: float | None = None,
        scheme: str = "auto",
        device: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self._vocab = VocabSpec.from_hf(tokenizer, name=name, scheme=scheme)
        self._cache: dict[str, np.ndarray] = {}
        self._stop_cache: dict[str, float] = {}
        eos = getattr(tokenizer, "eos_token_id", None)
        self._eos_ids: set[int] = set()
        if eos is not None:
            self._eos_ids.add(int(eos))
        eot = getattr(tokenizer, "eot_token_id", None)
        if eot is not None:
            self._eos_ids.add(int(eot))

    def vocab_spec(self) -> VocabSpec:
        return self._vocab

    def verify(self, samples: Sequence[str]) -> bool:
        """Round-trip the byte extraction. Run this once before trusting output."""
        return self._vocab.verify(self.tokenizer, samples)

    def next_token_probs(self, context: str) -> np.ndarray:
        import torch  # lazy: keeps the core numpy-only

        if context in self._cache:
            return self._cache[context]

        enc = self.tokenizer(context, return_tensors="pt", add_special_tokens=True)
        if self.device:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()

        if self.temperature != 1.0:
            logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)

        if self.top_p is not None:
            srt, idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(srt, dim=-1)
            cut = cum > self.top_p
            cut[1:] = cut[:-1].clone()
            cut[0] = False
            srt[cut] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, idx, srt)
            probs = probs / probs.sum()

        out = probs.detach().cpu().numpy().astype(np.float64)
        n = len(self._vocab.token_bytes)
        if out.shape[0] < n:  # model head narrower than the vocab table
            out = np.pad(out, (0, n - out.shape[0]))
        elif out.shape[0] > n:
            out = out[:n]

        # Stop tokens carry NO bytes, so they cannot be represented in the shared
        # base space -- there is no byte string for "the text ends". Zeroing them
        # and renormalising (as the byte pipeline must) would silently delete the
        # model's intention to stop and force the ensemble to generate forever.
        # So the stop mass is recorded here and surfaced on Stage8Result instead.
        out = out.copy()
        stop_ids = {i for i in self._eos_ids if i < out.shape[0]}
        self._stop_cache[context] = float(out[list(stop_ids)].sum()) if stop_ids else 0.0

        for sid in self._vocab.special_ids:
            if sid < out.shape[0]:
                out[sid] = 0.0
        s = out.sum()
        if s > 0:
            out /= s

        self._cache[context] = out
        return out

    def stop_probability(self, context: str) -> float:
        """Mass this agent placed on an end-of-text token at `context`.

        Read by `Stage8Pipeline` to decide termination, since stop tokens are
        structurally absent from the byte-level base space.
        """
        if context not in self._stop_cache:
            self.next_token_probs(context)
        return self._stop_cache.get(context, 0.0)


# --------------------------------------------------------------------------


class StaticAgent:
    """An agent whose distribution is supplied directly.

    Useful for tests, for replaying logged distributions, and for wiring Stage 8
    behind an inference server that already returns probabilities.
    """

    def __init__(self, name: str, vocab: VocabSpec, table: dict[str, np.ndarray] | None = None):
        self.name = name
        self._vocab = vocab
        self._table = table or {}
        self._stop: dict[str, float] = {}

    def vocab_spec(self) -> VocabSpec:
        return self._vocab

    def set(self, context: str, probs: np.ndarray, *, stop: float = 0.0) -> None:
        p = np.asarray(probs, dtype=np.float64)
        self._table[context] = p / p.sum()
        self._stop[context] = float(stop)

    def stop_probability(self, context: str) -> float:
        return self._stop.get(context, 0.0)

    def next_token_probs(self, context: str) -> np.ndarray:
        if context not in self._table:
            raise KeyError(f"{self.name}: no distribution registered for {context!r}")
        return self._table[context]


# --------------------------------------------------------------------------


def top_k_ids(probs: np.ndarray, k: int) -> np.ndarray:
    """Stage-2-style cheap summary: the ids worth putting in the tree."""
    if k >= probs.shape[0]:
        return np.flatnonzero(probs > 0)
    idx = np.argpartition(-probs, k)[:k]
    return idx[probs[idx] > 0]


# --------------------------------------------------------------------------
# bridge to the repo's Stage 1-7 agent interface
# --------------------------------------------------------------------------


class UpstreamAgent:
    """Adapts a `sahf.agents` agent (HFAgent / MockAgent / PoisonedAgentWrapper)
    to the Stage 8 interface.

    The two interfaces differ in one way that matters. Stages 1-7 pass
    `input_ids` around, which is only meaningful because every agent shares one
    tokenizer. Stage 8 exists precisely when that is false, so it passes TEXT and
    lets each agent encode it with its own tokenizer. This wrapper does that
    encoding, so `sahf.agents` needs no changes and poisoned agents keep working
    (a PoisonedAgentWrapper forwards `.tokenizer`, so it wraps transparently).

    Stage 1 is applied via `sahf.amplitude.softmax_to_amplitude` -- the repo's
    own implementation, not a copy -- and squared back to probabilities, which is
    what the byte-prefix tree consumes.
    """

    #: Names of end-of-turn tokens across the common instruct families. Needed
    #: because `tokenizer.eos_token_id` is a single id and several models stop on
    #: a different one: Llama-3 generates `<|eot_id|>` while its `eos_token_id`
    #: is `<|end_of_text|>`, so keying termination off `eos_token_id` alone means
    #: the model's stop signal is renormalised away and generation never ends.
    STOP_TOKEN_NAMES = frozenset({
        "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<|im_end|>",
        "<end_of_turn>", "</s>", "<|eom_id|>", "<|end|>",
    })

    #: Cap on the per-context probability cache. Each entry is a vocab-sized
    #: float64 array (~1.2 MB at 150k tokens) and generation visits a fresh
    #: context every step, so an unbounded dict grows without limit for the life
    #: of the process. Only the current step's contexts are ever re-read.
    CACHE_SIZE = 8

    def __init__(
        self,
        agent,
        *,
        name: str | None = None,
        scheme: str = "auto",
        stop_token_ids: "set[int] | None" = None,
    ):
        from ..amplitude import softmax_to_amplitude

        if getattr(agent, "tokenizer", None) is None:
            raise ValueError(
                f"{getattr(agent, 'name', agent)!r} has no tokenizer; Stage 8 needs one "
                "per agent to express its vocabulary in bytes. MockAgent has no real "
                "vocabulary -- use sahf.sheaf.StaticAgent for offline testing instead."
            )
        self._agent = agent
        self._to_amplitude = softmax_to_amplitude
        self.name = name or getattr(agent, "name", "agent")
        self.tokenizer = agent.tokenizer
        self._vocab = VocabSpec.from_hf(agent.tokenizer, name=self.name, scheme=scheme)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._stop: "OrderedDict[str, float]" = OrderedDict()
        self.stop_token_ids = (
            set(stop_token_ids) if stop_token_ids is not None else self._discover_stop_ids()
        )

    def _discover_stop_ids(self) -> set[int]:
        ids = set()
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.add(int(eos))
        vocab = {}
        try:
            vocab = self.tokenizer.get_vocab()
        except Exception:
            pass
        for tok, tid in vocab.items():
            if tok in self.STOP_TOKEN_NAMES:
                ids.add(int(tid))
        for tid, tok in (getattr(self.tokenizer, "added_tokens_decoder", {}) or {}).items():
            if str(tok) in self.STOP_TOKEN_NAMES:
                ids.add(int(tid))
        return ids

    def _remember(self, store, key, value):
        store[key] = value
        while len(store) > self.CACHE_SIZE:
            store.popitem(last=False)
        return value

    def vocab_spec(self) -> VocabSpec:
        return self._vocab

    def verify(self, samples) -> bool:
        """Round-trip the byte extraction. Run once before trusting any output."""
        return self._vocab.verify(self.tokenizer, samples)

    def next_token_probs(self, context: str) -> np.ndarray:
        if context in self._cache:
            return self._cache[context]

        import torch

        input_ids = self.tokenizer(context, return_tensors="pt").input_ids
        if input_ids.numel() == 0:
            # An empty prompt encodes to zero tokens for tokenizers that add no
            # BOS, and a zero-length forward pass raises inside the model. Seed
            # with BOS (or EOS, which is what most decoder-only models use as a
            # sequence opener) so generate(prompt="") works.
            seed = getattr(self.tokenizer, "bos_token_id", None)
            if seed is None:
                seed = getattr(self.tokenizer, "eos_token_id", None)
            input_ids = torch.tensor([[int(seed) if seed is not None else 0]], dtype=torch.long)
        with torch.no_grad():
            logits = self._agent.next_logits(input_ids)
        psi = self._to_amplitude(logits)[0]  # Stage 1, the repo's implementation
        probs = (psi**2).double().cpu().numpy()

        n = len(self._vocab.token_bytes)
        if probs.shape[0] < n:
            probs = np.pad(probs, (0, n - probs.shape[0]))
        elif probs.shape[0] > n:
            probs = probs[:n]
        probs = probs.copy()

        # Stop tokens carry no bytes, so they cannot exist in the byte base space.
        # Record the mass before dropping it, or the ensemble can never terminate.
        live = [i for i in self.stop_token_ids if i < probs.shape[0]]
        self._remember(self._stop, context, float(probs[live].sum()) if live else 0.0)

        for sid in self._vocab.special_ids:
            if sid < probs.shape[0]:
                probs[sid] = 0.0
        total = probs.sum()
        if total > 0:
            probs /= total

        return self._remember(self._cache, context, probs)

    def stop_probability(self, context: str) -> float:
        if context not in self._stop:
            self.next_token_probs(context)
        return self._stop.get(context, 0.0)


def assert_distinct_tokenizers(agents, *, strict: bool = False) -> dict:
    """Flags an ensemble that does not actually need Stage 8.

    If every agent really does share a tokenizer, byte-level reconciliation is
    pure overhead: token-level fusion would be faster and would not truncate to
    top-k. This project targets the mismatched case, so that situation almost
    certainly means the wrong models were configured.

    That is a performance note rather than a correctness bug, so by default this
    warns rather than raising. Pass ``strict=True`` to make it an error.

    Returns ``{agent_name: (vocab_size, scheme)}``.
    """
    import warnings

    info = {}
    fingerprints = set()
    for a in agents:
        vs = a.vocab_spec()
        info[a.name] = (vs.size, vs.scheme)
        # size + scheme is a cheap proxy; two tokenizers agreeing on both are
        # very likely the same one, and the sample below settles it.
        sample = tuple(a.tokenizer.encode("The capital of France is Paris"))
        fingerprints.add((vs.size, vs.scheme, sample))

    if len(fingerprints) == 1:
        msg = (
            "All agents share one tokenizer, so there is no vocabulary mismatch "
            "for Stage 8 to resolve — this is almost certainly a misconfigured "
            f"model list. Vocabularies: {info}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return info
