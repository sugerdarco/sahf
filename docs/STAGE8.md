# Stage 8 — Sheaf Reconciliation

Stage 8 is what makes fusion defined at all when agents' tokenizers differ, which
is the only case this repository now targets. It reuses Stages 1 and 2 directly
and runs Stages 5/6/7 per byte-prefix-tree node.

Stage 8 was originally added as a pure addition alongside a single-tokenizer
pipeline. That pipeline has since been removed; `sahf/amplitude.py`, `fusion.py`,
`gate.py`, `robust.py`, `agents.py` and `logger.py` are all still here and still
unmodified, because Stage 8 depends on every one of them.

---

## Why token-space fusion cannot be pointed at mixed models

The removed single-tokenizer pipeline asserted a shared vocabulary and refused to
start without one. That check was right. Two things break the moment
vocabularies differ:

**Comparison.** Stages 5/6/7 operate on amplitude vectors indexed by token id.
Qwen's token 5921 and Gemma's token 5921 are unrelated strings, and the vectors
aren't even the same length. Averaging them is not merely inaccurate, it is
undefined.

**Feedback.** Token-space decoding appends one `token_id` and feeds the same
`input_ids` to every agent. With different tokenizers there is no shared id to
append.

Stage 8 fixes both by moving to a space every agent genuinely shares: **raw
bytes**.

---

## What it does

```
agents (own tokenizers)
   │  next-token distributions over incomparable vocabularies
   ▼
token id → RAW BYTES                                       sheaf/vocab.py
   │  byte-level BPE / sentencepiece / wordpiece / tiktoken
   ▼
union byte-prefix tree — the shared base space             sheaf/prefix_tree.py
   │  cover(s) = Σ p(v) over tokens whose bytes start with s
   ▼
per-node local sections: P[next byte | prefix s]           sheaf/reconciler.py
   │  every agent's section is on the SAME 256-simplex →
   │  Stages 1/5/6/7 apply per node, unchanged
   ▼
glue: chain conditionals down from the root
   ▼
consensus bytes  →  appended as text, every agent re-encodes
                 →  and projected back into each agent's token space
```

The sheaf structure, concretely:

| concept | here |
|---|---|
| base space | the tree's nodes (byte prefixes `s`) |
| stalk at `s` | distribution over the next **byte** |
| local section | one agent's conditional, where it still has mass |
| restriction | parent → child edge = conditioning one byte further |
| gluing | agreement on overlaps ⇒ one global section |

---

## How Stages 1–7 are reused

| Stage | In Stage 8 |
|---|---|
| 1 Amplitude | **imported.** `sahf.amplitude.softmax_to_amplitude`, called in `UpstreamAgent`. |
| 2 Divergence gate | **imported.** `sahf.gate.mean_entropy` and `GateThresholds`, applied in byte space. |
| 5 Mean fusion | transcribed to numpy, run **per node**. |
| 6 Outlier check | transcribed to numpy, run **per node**, same `mad_multiplier` as `config_sheaf.yaml`. |
| 7 Geometric median | transcribed to numpy, run **per node**, only where Stage 6 fires. |

Stages 5/6/7 are transcribed rather than called because they execute per tree
node — hundreds of times per token on 256-dim vectors — where torch's per-call
overhead dominates the arithmetic. Transcription is a duplication risk, so
`tests/test_sheaf_parity.py` pins the numpy versions to `sahf.fusion` and
`sahf.robust` numerically (30 parity assertions, tolerance 1e-6). **If you change
Stages 5/6/7, those tests fail** — port the change into
`sahf/sheaf/reconciler.py` rather than loosening the tolerance.

Writing those tests immediately caught two real mismatches:

- My standalone Stage 6 used a modified z-score (`0.6745·(d−med)/MAD > 3.5`,
  ≈ `med + 5.19·MAD`) where this repo uses `med + 3.0·MAD`. Stage 8 now uses the
  repo's rule.
- `torch.median` returns the **lower** of the two central values for even-length
  inputs; `np.median` averages them. With N=3 (the default) they agree, so this
  would have stayed invisible until someone added a fourth agent, at which point
  Stage 8 and Stage 6 would have flagged different agents. The transcription now
  replicates torch's definition exactly.

---

## Quickstart

The prefix tree is built **once** by a separate command, then reused by every
run — it depends only on the ensemble's vocabularies, never on the prompt:

```bash
python build_prefix_tree.py          # once, whenever `models` changes
python run_sheaf.py --prompt "..."   # loads artifacts/prefix_tree.npz
```

See `docs/STAGE8_PREBUILT_TREE.md`. Offline, no models:

```bash
python demo_sheaf_mock_run.py   # 3 different tokenizers, 1 poisoned agent, writes to out/
python -m sahf.sheaf.demo       # three focused scenarios
pytest tests/test_sheaf_parity.py -q
```

Real models:

```bash
python run_sheaf.py --prompt "Explain the water cycle in two sentences."
python run_sheaf.py --prompt "..." --poison-index 2 --poison-mode invert
```

`config_sheaf.yaml` is the only config. Records in `out/runs/*/steps.jsonl` keep
the original `path` / `escalated` / `entropy` / `divergence` field names and add
Stage 8 fields (`tree_nodes`, `fused_nodes`, `escalated_nodes`, `coverage`,
`unreachable`, `stop_mass`, `n_bytes`, `elapsed_ms`) alongside them, so existing
log tooling still reads them.

In code:

```python
from sahf.agents import HFAgent
from sahf.gate import GateThresholds
from sahf.sheaf import SheafOrchestrator, UpstreamAgent

agents = [UpstreamAgent(HFAgent(n)) for n in MODELS]
for a in agents:
    assert a.verify(["The capital of France is Paris"])   # do not skip this

orch = SheafOrchestrator(agents, GateThresholds(), k=16)
text, history = orch.generate("The capital of France is")
```

`generate` returns **text**, not `output_ids` — with mismatched tokenizers there
is no shared id sequence to return.

---

## Three things worth knowing before you rely on it

**1. Verify byte extraction, always.** Vocabularies store tokens in a display
form (`Ġthe`, `▁the`, `##ing`), not bytes. A mis-detected scheme does not raise —
it shifts every token by one leading space, the most common token in English, and
builds a plausible tree that is silently misaligned. `run_sheaf.py` aborts if the
round-trip fails. Detection originally misread *any* byte-level vocabulary
containing `##` (i.e. any vocab trained on markdown or code) as WordPiece; see
`docs/STAGE8_AUDIT_REPORT.md`, AUDIT-1.

**2. Token boundaries are horizons, not content.** If agent A holds `" the"` as
one token and agent B holds `" th"` + `"e"`, then at node `" th"` A continues and
B stops. Treating "stop" as a fusable symbol would report a 50/50 split — a
manufactured disagreement about where a tokenizer draws boundaries, when the
agents agree completely. So a stop means that agent has run out of what one
forward pass can tell us; it leaves the support at deeper nodes and the others
carry the prediction. `python -m sahf.sheaf.demo` asserts max disagreement of
exactly `0.000000` across three tokenizers segmenting one string three ways.

**3. The gate has to look at every node.** Byte-level tokenizers put a leading
space on most word-initial tokens, so `" Paris"` and `" Berlin"` share their
entire root section — two agents in flat disagreement look unanimous at depth 0.
Gating on the root alone sends nearly every step down Path A and Stage 6/7 never
fires. `_byte_level_gate` therefore maximises the pairwise spread over all active
nodes.

---

## Cost

Three tokenizers trained from scratch (byte-level BPE 32k / byte-level BPE 27k /
Unigram 32k), single core, 2.1 GHz Xeon, numpy. Full tables in
`docs/STAGE8_BENCHMARK_RESULTS.md`.

| k | ms/step | 1st-byte accuracy | correct bytes/step |
|---|---|---|---|
| 8 | 4.6 | 100% | 3.38 |
| **16 (default)** | **6.3** | 100% | 3.38 |
| 64 | 24.5 | 100% | 3.40 |
| 256 | 108.2 | 100% | 3.40 |

Quality is flat across the whole sweep while cost grows 23×, because a single
step cannot exceed the mean token length (~3.3 bytes) of lookahead — past its own
token boundary an agent has no information. Hence `top_k: 16`.

Latency is flat in entropy (tree size is set by `k`, not diffuseness), so
per-token cost is predictable. Agent scaling is sub-linear: 3 → 8 agents costs
2.4×. Against a 20–30 ms forward pass this is roughly 20–25% overhead.

`mode="full"` (no top-k truncation) costs **1.5–3.5 s/step** on real
vocabularies — offline evaluation only, never a decode loop.

---

## Known limitations

- **Exactness.** Phan et al. (ICLR 2025, arXiv:2410.09303) also correct for the
  *prompt's* cover. This implements exact cover-mass marginalisation for the
  next-token step and re-encodes the full context each step; it does not
  enumerate prompt covers.
- **The entropy half of the Stage 2 gate does not work in byte space.** Measured
  root entropy is either ~0 (first byte near-deterministic, which is common) or
  2.3-2.7 nats, against `theta_H = 2.0` and a ceiling of `ln(256) = 5.55`. It
  never discriminates; divergence carries the routing decision. It now has its
  own `sheaf.byte_entropy_threshold` key rather than inheriting the token-space
  value, and the observed entropy is logged for calibration. See
  `docs/STAGE8_INTEGRATION_AUDIT.md` INT-3.
- **`theta_D` is likewise uncalibrated for byte space**, though unlike `theta_H`
  it does vary usefully. Calibrate from `out/runs/*/steps.jsonl`.
- **Emission precision.** At the default `min_support` (majority), decode emits
  ~7 bytes of which ~3.4 match a simulated ground truth. `min_support` = agent
  count (unanimous) raises that from ~48% to ~72% for shorter chunks. Worth
  considering if wrong committed bytes cost more than extra steps; not made the
  default because the measurement is simulated.
- **No KV-cache**, inherited from `HFAgent` — and here every step re-encodes the
  full context by design, so caching matters more than it does for Stages 1–7.
- **`MockAgent` cannot be used.** It has no tokenizer, so its tokens have no byte
  image. Use `sahf.sheaf.StaticAgent` for offline testing.
- **`ARCHITECTURE.md` and `HISTORY.md` describe the removed single-tokenizer
  design.** They are kept deliberately as the historical record and carry a
  banner saying so; `README.md` and `SAHF_ARCHITECTURE_SPEC.md` describe what
  actually runs.
- **Import cost.** `sahf/__init__.py` imports torch, so `import sahf.sheaf` pulls
  torch in even though Stage 8's own code is numpy-only. Avoidable only by
  editing `sahf/__init__.py`, which this change deliberately does not do.

---

## Files added

```
sahf/sheaf/__init__.py        public API
sahf/sheaf/vocab.py           token id -> raw bytes, 4 schemes + verification
sahf/sheaf/prefix_tree.py     union byte-prefix tree, vectorized cover/term mass
sahf/sheaf/reconciler.py      local sections, Stages 5/6/7 per node, gluing
sahf/sheaf/adapters.py        UpstreamAgent bridge, StaticAgent, DirectHFAgent
sahf/sheaf/pipeline.py        Stage8Pipeline + diagnostics
sahf/sheaf/orchestrator.py    SheafOrchestrator — the per-token decode loop
sahf/sheaf/demo.py            three focused offline scenarios

tests/test_sheaf_parity.py           30 assertions pinning Stage 8 to Stages 1/5/6/7
tests/test_sheaf_stage8.py           25 invariant + regression tests
tests/test_sheaf_orchestrator.py     8 end-to-end tests (synthetic vocabularies)
tests/test_sheaf_upstream_agent.py   9 end-to-end tests through UpstreamAgent with
                                     REAL trained tokenizers + real torch forward passes
tests/test_sheaf_real_tokenizers.py  trains BPE/WordPiece/Unigram and round-trips
tests/test_sheaf_integration_regressions.py  11 regression tests, one per
                                     integration audit finding
tests/test_sheaf_prebuilt_tree.py    13 tests: artifact round-trip, sparse
                                     propagation, prebuilt decode path

build_prefix_tree.py          optional one-time tree build -> artifacts/prefix_tree.npz
run_sheaf.py                  generation CLI; loads the prebuilt tree if present
demo_sheaf_mock_run.py        offline example run
config_sheaf.yaml             the only config

audit/audit_stage8.py         22 probes over the library internals
audit/audit_integration.py    12 probes over the integration seams
benchmarks/                   tokenizer training + isolated Stage 8 benchmark
docs/STAGE8_AUDIT_REPORT.md   library audit: 9 defects found and fixed
docs/STAGE8_INTEGRATION_AUDIT.md  integration audit: 9 issues, 8 fixed
docs/STAGE8_PREBUILT_TREE.md  build-once tree: artifact, sparse propagation
docs/STAGE8_BENCHMARK_RESULTS.md
```

118 tests in total. `pytest` collects them automatically — `pytest.ini` is unchanged.

Optional dependency: `tokenizers>=0.15`, only for
`tests/test_sheaf_real_tokenizers.py` and the benchmarks. It is skipped when
absent, so `requirements.txt` did not need changing.
