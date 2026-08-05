# Stage 8 Audit Report — `sahf_sheaf`

**Scope:** stage-by-stage verification of the byte-prefix tree pipeline built for
Stage 8 (Sheaf Reconciliation) of SAHF Distribution Fusion.
**Method:** adversarial probes per stage, checking mathematical invariants and
inputs the happy path never exercises — not a re-read of the code.
**Result:** **9 defects found, 9 fixed.** All now covered by regression tests.

| | |
|---|---|
| Stages audited | A vocab extraction · B prefix tree · C sheaf gluing · D projection & pipeline · E model adapter |
| Defects found | 9 (2 silent-corruption, 3 correctness, 2 misleading-diagnostic, 1 robustness, 1 performance) |
| Tests before → after | 16 → 25 |
| Invariants confirmed sound | 10 (listed in §3) |

---

## 1. Severity summary

| ID | Stage | Defect | Severity |
|---|---|---|---|
| AUDIT-1 | A | Byte-level vocab containing `##` misdetected as WordPiece | **Critical** |
| AUDIT-2 | A | Chat/control tokens fused as literal text | **Critical** |
| AUDIT-7 | D | `generate()` corrupts multi-byte UTF-8 characters | **High** |
| AUDIT-8 | E | End-of-text mass silently renormalised away | **High** |
| AUDIT-4 | B | Duplicate ids in `restrict` double-count probability | Medium |
| AUDIT-3 | B/C | Mis-sized distribution mis-indexes instead of failing | Medium |
| AUDIT-5 | D | `residual` diagnostic reported clamped noise | Medium |
| AUDIT-6 | D | Mass propagation ran twice per agent per step | Low |
| AUDIT-9 | — | No regression coverage for any of the above | Low |

"Critical" here means **silent** corruption: no exception, no warning, plausible-looking
output. Those are the two worth reading in full.

---

## 2. Defects in detail

### AUDIT-1 — Byte-level vocabulary misdetected as WordPiece *(Critical, Stage A)*

`detect_scheme()` tested for the WordPiece `##` prefix **before** testing for
byte-level BPE:

```python
if any(t.startswith("##") for t in sample):
    return "wordpiece"
```

`##` is an ordinary byte-level token in any vocabulary trained on markdown or
code — it is exactly how a level-2 heading begins. So a GPT-2/Llama-3/Qwen-style
tokenizer trained on realistic data is classified as WordPiece, and every token
is then decoded by `decode_wordpiece`, which **prepends a space to every
non-`##` token**.

Consequence: the entire vocabulary shifts by one leading space — the single most
common token in English. The prefix tree still builds, the gluing still runs,
every invariant still holds, and the consensus is quietly wrong.

The detection order was load-bearing and undocumented. It survived the original
validation only because the test corpus contained no `##`.

**Probe**

```python
enc = bytes_to_unicode()
vocab = ["".join(enc[b] for b in t) for t in [b"##", b" the", b"\n", b"def", b" x"]]
detect_scheme(vocab)   # -> 'wordpiece'   (expected 'byte_level')
```

**Fix.** Detect on a *positive fingerprint* rather than absence of evidence.
Byte-level display forms are the only ones that contain the permutation's marker
characters — `Ġ` for space, `Ċ` for newline, and the rest of the images of
non-printable bytes. No other scheme can produce them:

```python
BYTE_LEVEL_MARKERS = {BYTE_ENCODER[b] for b in range(256) if BYTE_ENCODER[b] != chr(b)}
```

New order: sentencepiece (`▁` / `<0xNN>`, unambiguous) → byte-level (marker
characters) → wordpiece (**>1 %** of tokens starting `##`, not one) → byte-level
charset fallback → identity. The `>1 %` floor matters independently: a single
stray `##` token no longer outvotes an entire vocabulary.

---

### AUDIT-2 — Control tokens fused as literal text *(Critical, Stage A)*

`VocabSpec.from_hf` excluded only `all_special_ids`. Chat and control tokens —
`<|im_start|>`, `<|endoftext|>`, tool-call markers — are registered in
`added_tokens_decoder`, not always in `all_special_ids`. Their display strings
are plain printable ASCII, so the byte-level decoder happily turned
`<|im_start|>` into the 12 literal bytes `b"<|im_start|>"`.

Consequences, both silent:

1. Those bytes enter the shared tree as if the model had predicted that *text*,
   and get fused into the consensus alongside real content.
2. Worse for a multi-agent system: agents with different chat templates
   contribute different control strings to the same base space, so one agent's
   template leaks into another's byte predictions. Cross-template contamination
   is precisely what the shared base space is supposed to prevent.

**Fix.** `drop_added=True` (default) excludes `added_tokens_decoder` ids plus
anything shaped `<|...|>`. Verified: `token_bytes` for such ids is now `None`
and they never reach the tree.

---

### AUDIT-7 — `generate()` corrupts multi-byte UTF-8 *(High, Stage D)*

The consensus emits a **byte** chunk, whose length is set by the ensemble's
joint horizon and has no reason to land on a character boundary. Byte-level BPE
vocabularies genuinely contain tokens that end mid-character (byte fallback).
The loop did:

```python
buf.extend(emitted)
text = buf.decode("utf-8", errors="replace")
```

so an incomplete trailing sequence became `U+FFFD` — and that corrupted string
was fed straight back to every agent as the next context. This breaks every
non-ASCII language: CJK, Indic scripts, emoji, accented Latin.

**Probe.** Vocabulary containing `b'caf\xc3'` (ends mid-`é`) → `generate()`
returned `'caf\ufffd'`, and the agents' next context was `'caf\ufffd'`.

**Fix.** An incremental UTF-8 decoder, which buffers a partial sequence until
the next chunk completes it:

```python
decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
text += decoder.decode(chunk)
```

The final flush is deliberately *omitted*: if a run ends mid-character those
bytes stay buffered rather than surfacing as visible corruption, and the raw
chunk remains on the last trace entry's `consensus_bytes` for a caller resuming
generation.

---

### AUDIT-8 — End-of-text mass silently discarded *(High, Stage E)*

`HFAgent` zeroed every special id and renormalised. That is *necessary* — a stop
token has no byte image, so it cannot exist in a byte-level base space. But it
was done silently, which means **the ensemble could never terminate**. A model
placing 0.99 on EOS had that mass deleted and redistributed across whatever
byte-carrying tokens remained; `generate()` then ran to `max_steps` emitting
noise.

This one is architectural rather than a typo: stop is genuinely outside the
shared base space, so it has to be carried *beside* it.

**Fix.** Stop mass is captured before renormalisation and surfaced:

- `HFAgent.stop_probability(context)` / `StaticAgent.set(..., stop=)`
- `Stage8Result.stop_mass` (per agent) and `Stage8Result.should_stop`
  (majority of agents above 0.5)
- `generate()` halts on `should_stop`

Termination is now a consensus decision made in the same majority style as
`min_support`, rather than an accident of renormalisation.

---

### AUDIT-4 — Duplicate ids double-count probability *(Medium, Stage B)*

`BytePrefixTree.from_vocabs(..., restrict=...)` walked the id list as given. A
repeated id was walked twice and had its probability added twice to every node
on its path.

This is not hypothetical: `restrict` is built by merging top-k lists, and the
intended top-k-union workflow produces repeats routinely.

**Probe.** `restrict=[[0, 0, 1]]` with `p = [0.4, 0.2, 0.3, 0.1]` gave root cover
`1.0000` instead of `0.6`.

Note the identity `cover(s) = term(s) + Σ children` **still held** — both sides
inflate equally — so the existing invariant tests could never have caught it.
Only the absolute normalisation was wrong, which then silently distorted
`coverage`, the eps pruning threshold, and every fused conditional.

**Fix.** `dict.fromkeys(...)` de-duplicates while preserving order.

---

### AUDIT-3 — Mis-sized distributions mis-index instead of failing *(Medium, Stages B/C)*

`cover_mass` indexed `probs[flat_tok]` with no shape check. A vector of the wrong
length either raised a bare `IndexError` from deep inside numpy, or — when
merely *longer* than the vocabulary — silently read the wrong probabilities. With
N agents of differing vocabulary sizes, passing distributions in the wrong order
is an easy and completely silent mistake.

**Fix.** Explicit validation naming the agent, its vocabulary and the expected
length. `reconcile` inherits the check, so `test_mis_sized_distribution_rejected`
covers both entry points.

---

### AUDIT-5 — `residual` reported clamped noise *(Medium, Stage D)*

`project_to_vocab` returned `residual = max(0, depth1_mass − claimed)`,
documented as "glued mass this vocabulary cannot express".

The pre-normalisation total is

```
claimed = Σ_s term_a(s) · glued_cover(s) / cover_a(s)
```

which is an **importance reweighting** of the agent's own distribution, not a
coverage figure. It equals 1 only when the consensus happens to match that
agent, and for a vocabulary with boundaries at several depths it routinely
exceeds 1 — at which point `max(0, ...)` clamps the reported residual to exactly
`0.0000`, i.e. "no loss", regardless of the truth.

**Probe.** Agent with tokens `a`, `ab`, `abc`, `abcd` — boundaries at four
depths along one path:

```
node b'a'     glued=1.0000  rate=0.2500 -> 0.2500
node b'ab'    glued=1.0000  rate=0.3333 -> 0.3333
node b'abc'   glued=1.0000  rate=0.5000 -> 0.5000
node b'abcd'  glued=0.9743  rate=1.0000 -> 0.9743
pre-normalisation total = 2.0577   reported residual = 0.0000
```

**Fix.** Split the two quantities into a `ProjectionReport`:

- `mass_ratio` — the pre-normalisation total, honestly labelled as reweighting
  drift (1.0 = the glued section matched this agent; >1 is legitimate).
- `unreachable` — the real loss figure, computed properly: a bottom-up pass
  marks whether any token of this agent can end at or below each node, then the
  glued mass entering each **maximal** boundary-free subtree is summed. That is
  mass this vocabulary genuinely cannot represent.

The returned token distribution was always correct; only the diagnostic lied.

---

### AUDIT-6 — Mass propagation ran twice per agent *(Low, Stage D)*

`Stage8Pipeline._reconcile` computed `coverage` with a fresh
`tree.cover_mass(a, probs[a])` call, re-running the per-step hot path that
`reconcile` had already completed.

**Fix.** Read `section.agent_cover[a][0]`. Measured saving: **5.5 ms/step** in
full mode (165k-node tree, 3×32k vocabularies); negligible in top-k mode, where
the tree is small.

---

### AUDIT-9 — No regression coverage

The original 16 tests checked invariants that all eight defects above satisfied.
Nine tests added, one per defect, each reproducing the specific failure.

---

## 3. What was verified as sound

These held under adversarial probing and needed no change:

1. `cover(s) = term(s) + Σ_b cover(s·b)` at every node.
2. Root cover equals 1 (with AUDIT-4 fixed).
3. The identity survives `max_depth` truncation.
4. Cover mass is monotone down the tree — this is what makes the depth-sorted
   gluing walk correct: an active node's parent is always active, so no node is
   ever visited before its parent's mass is final.
5. Every fused conditional is normalised.
6. Glued depth-1 mass sums to 1.
7. No byte supported by any agent is ever assigned zero fused mass (both mean
   fusion and Weiszfeld use non-negative weights on non-negative amplitudes).
8. The sole-surviving-agent fast path returns that agent's section unchanged.
9. Agent-count mismatch is rejected.
10. Projection always returns a valid probability distribution.

Also re-confirmed after the fixes: **max disagreement `0.000000`** across three
tokenizers segmenting one string three different ways (the boundary-as-horizon
design), Byzantine agent absorbed by per-node Stage 7 escalation
(`P('P')` 0.77 → 0.95), and exact multi-step reconstruction of
`' Paris is the capital'` across three granularities.

---

## 4. Unresolved — by design, not by omission

Carried forward from the original limitations; none are regressions.

- **Prompt-cover exactness.** Phan et al.'s correction enumerates the token
  sequences that encode the same prompt bytes. This implements exact cover-mass
  marginalisation for the next-token step and re-encodes the full context each
  step; it does not enumerate prompt covers.
- **WordPiece is lossy by construction.** It discards whitespace, so byte
  reconstruction is a heuristic. Fine as a fusion participant, unsuitable as
  ground truth.
- **Top-k truncation.** Watch `coverage`; on diffuse tokens `k=64` can reach only
  ~52 % of the mass.
- **Full-mode speed.** A Python loop over active nodes. Batching Weiszfeld across
  nodes would fix it; unnecessary while `topk_union` is the hot path.
- **`prefix_space`.** SentencePiece/WordPiece prepend a dummy space to a
  sequence. Recorded on `VocabSpec.prefix_space` rather than treated as an
  error; affects prompt encoding only, never the fused next-token bytes.

---

## 5. Reproducing

```bash
make audit    # 22 probes, all stages -> "no defects found"; non-zero exit on defect
make test     # 29 tests, including one regression test per defect below
make demo     # three scenarios, offline
```

## 6. Files changed

| File | Defects fixed |
|---|---|
| `vocab.py` | AUDIT-1, AUDIT-2 |
| `prefix_tree.py` | AUDIT-3, AUDIT-4 |
| `sheaf.py` | AUDIT-5 (`ProjectionReport`) |
| `pipeline.py` | AUDIT-5, AUDIT-6, AUDIT-7, AUDIT-8 |
| `sources.py` | AUDIT-8 |
| `tests/test_stage8.py` | AUDIT-9 (+9 regression tests) |
| `tests/test_real_tokenizers.py` | AUDIT-1, AUDIT-2 (real trained tokenizers) |
| `audit/audit_stage8.py` | new — the audit harness itself |

**API changes.** `project_to_vocab` returns `(probs, ProjectionReport)` rather
than `(probs, float)`; `Stage8Result.residual` is replaced by `.unreachable` and
`.mass_ratio`, and gains `.stop_mass` / `.should_stop`.
