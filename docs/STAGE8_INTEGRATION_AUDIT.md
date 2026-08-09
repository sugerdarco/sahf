# Stage 8 Integration — Verification Report

**Scope:** the code written to attach Stage 8 to this repo — `UpstreamAgent`,
`SheafOrchestrator`, the byte-space Stage 2 gate, and the run/logging path.
The library internals (vocab / prefix tree / sheaf / pipeline) were audited
separately; see `STAGE8_AUDIT_REPORT.md`.

**Method:** adversarial probes against the integration seams, plus timing
measurements — not a re-read of the code. Harness: `audit/audit_integration.py`.

**Result:** **9 issues found. 8 fixed, 1 is a design property that cannot be
fixed and is documented instead.**

| | |
|---|---|
| Issues found | 9 (2 correctness, 3 resource/robustness, 2 performance, 2 hygiene) |
| Tests | 110 → 112 (87 Stage 8, 25 pre-existing, all passing) |
| Orchestrator step | 17.56 ms → 16.04 ms; gate overhead 25% → 13% |
| Files modified outside Stage 8 | **zero**, still |

---

## 1. Issues found

| ID | Area | Issue | Severity | Status |
|---|---|---|---|---|
| INT-1 | orchestrator | Prefix tree built twice per decoding step | Perf | Fixed |
| INT-2 | orchestrator | Mass propagation run twice per agent per step | Perf | Fixed |
| INT-3 | orchestrator | `theta_H` silently inherited from token space, where it cannot discriminate | Design | Documented + mitigated |
| INT-4 | UpstreamAgent | Empty prompt crashes the forward pass | **Correctness** | Fixed |
| INT-5 | UpstreamAgent | Stop mass only counts `tokenizer.eos_token_id` | **Correctness** | Fixed |
| INT-6 | UpstreamAgent | Per-context cache unbounded | Resource | Fixed |
| INT-7 | adapters | Dead branch in `DirectHFAgent` stop-id collection | Hygiene | Fixed |
| INT-8 | demo | Docstring still names the standalone module path | Hygiene | Fixed |
| INT-9 | orchestrator | Per-node sections computed twice per step | Perf | Fixed |

Two of these would have bitten on the very first real run against
`config_sheaf.yaml`. They are worth reading in full.

---

## 2. The two that mattered

### INT-5 — generation would never terminate on Llama-3

`UpstreamAgent` read stop mass from `tokenizer.eos_token_id` alone:

```python
eos = getattr(self.tokenizer, "eos_token_id", None)
self._stop[context] = float(probs[eos]) if eos is not None else 0.0
```

Stop tokens carry no bytes, so they cannot live in the byte base space at all —
they are zeroed and the rest renormalised. Capturing their mass first is the only
thing that lets the ensemble terminate, which is why `should_stop` exists.

But **`eos_token_id` is a single id, and several instruct models stop on a
different one.** Llama-3 generates `<|eot_id|>` while its `eos_token_id` is
`<|end_of_text|>`. `config_sheaf.yaml` lists `meta-llama/Llama-3.2-1B-Instruct`.
So that agent's stop signal would have been renormalised away with no trace, and
generation would run to `max_new_bytes` every time, emitting text past the point
the model wanted to stop.

Measured before the fix: model puts ~1.0 on `<|eot_id|>`, `stop_probability`
returns **0.0000**.

**Fix.** Stop ids are now discovered from `eos_token_id` *plus* any vocabulary or
added-token entry matching a known end-of-turn name (`<|eot_id|>`,
`<|im_end|>`, `<|end_of_text|>`, `<|endoftext|>`, `</s>`, `<end_of_turn>`,
`<|eom_id|>`, `<|end|>`), with an explicit `stop_token_ids=` override for anything
unusual. Regression test asserts both the union and the override.

### INT-4 — `generate(prompt="")` crashed

An empty prompt encodes to zero tokens for tokenizers that add no BOS, and a
zero-length forward pass raises inside the model. Every Stage 8 test used
`StaticAgent`, which never encodes anything, so nothing caught it —
`demo_sheaf_mock_run.py` calls `generate("")` and passes only because its agents
are `StaticAgent`s.

**Fix.** Empty encodings are seeded with BOS, falling back to EOS (what most
decoder-only models use as a sequence opener), then id 0.

---

## 3. Performance: the gate was paying three times

`SheafOrchestrator.step` runs the Stage 2 gate before deciding Path A or Path B.
The gate has to build the prefix tree, propagate each agent's mass onto it, and
compute the per-node sections in order to measure divergence. It then threw all
three away and `reconcile` recomputed every one.

| | before | after |
|---|---|---|
| Trees built per step | 2 | 1 |
| `cover_mass` calls per step (3 agents) | 6 | 3 |
| Per-node section passes | 2 | 1 |
| Gate overhead | 4.39 ms (25% of step) | 2.16 ms (13%) |
| Orchestrator step | 17.56 ms | **16.04 ms** |

The breakdown is worth recording, because the naive assumption was wrong: the
tree build is cheap (0.48 ms) and mass propagation is nearly free (0.02 ms). The
cost was the **per-node section pass at 2.93 ms** — so fixing only the tree and
mass (INT-1/INT-2) moved the total by 0.1 ms. INT-9, sharing the sections, is
what actually paid.

Both reuse paths are pinned by equivalence tests: `reconcile` with supplied
`mass=` / `stalks=` must produce bit-identical conditionals, cover and per-node
support to recomputing them. Without that, a performance path could silently
change fusion results depending on whether the gate ran.

**Is the gate worth its remaining 13%?** Measured over 60 steps, Path A (mean
only) and Path B (mean → MAD → geometric median) produce a different consensus in
**6 of 60 steps (10%)**. So it is not redundant — but note that `SheafReconciler`
already skips the robust machinery per node when a node's sections are unanimous,
so the whole-step gate is partly duplicating a decision made at finer grain.

---

## 4. INT-3 — the entropy condition does not work in byte space

Not fixable; recording it plainly.

Stage 2 is a two-factor test: entropy `H < theta_H` **and** divergence
`D < theta_D`. `theta_H = 2.0` nats was calibrated (loosely — see `HISTORY.md`) for a
~150k-token vocabulary where the ceiling is `ln(151000) ≈ 11.9`.

In byte space the ceiling is `ln(256) ≈ 5.55`, and measured root entropy does not
sit anywhere useful relative to 2.0:

| regime | measured `H` | vs `theta_H = 2.0` |
|---|---|---|
| first byte near-deterministic (very common) | ~0.000 | always below → always Path A |
| realistic diffuse next-byte | 2.32 – 2.72 (median 2.63) | always above → always Path B |

In neither regime does it discriminate; it acts as a constant. The near-zero case
is common precisely because byte-level tokenizers put a leading space on most
word-initial tokens, so the first byte is frequently a foregone conclusion.
**Divergence carries the routing decision in practice.**

This is the same root cause as a bug fixed earlier in the integration: the gate
originally measured divergence at the root only, where `" Paris"` and `" Berlin"`
share their entire section, so two agents in flat disagreement looked unanimous.
That was fixed by maximising `D` over all active nodes. `H` has no equivalent fix
— entropy of the mean consensus is a single number about the first byte, and the
first byte is usually not where the information is.

**Mitigation, not a fix:**

- `sheaf.byte_entropy_threshold` is its own config key, so it does not silently
  inherit a token-space number.
- Left at 2.0, because that fails toward Path B — full fusion, slower but never
  less robust.
- The observed `entropy` is written to `steps.jsonl` on every step, so it can be
  calibrated against real models once you have run some.

---

## 5. Smaller items

**INT-6, unbounded cache.** `UpstreamAgent` memoised probabilities per context
string and never evicted. Generation visits a fresh context every step and each
entry is a vocab-sized float64 array (~1.2 MB at 150k tokens), so a 200-step
generation with 3 agents retains ~700 MB for the process lifetime. Measured: 50
distinct contexts → 50 entries retained. The cache only ever needs the current
step (the gate and reconcile both ask for the same context), so it is now an
`OrderedDict` capped at 8.

**INT-7.** `DirectHFAgent` looped over `("pad_token_id", "eot_token_id")` and then
discarded the first via `and extra == "eot_token_id"` — a no-op branch reading as
though pad were handled. Replaced with a direct lookup.

**INT-8.** `sahf/sheaf/demo.py` still said `python -m sahf_sheaf.demo`, the
standalone package path, which does not exist in this repo.

---

## 6. What was verified as sound

Probed and correct, no change needed:

1. `PoisonedAgentWrapper` wraps transparently — it forwards `.tokenizer`, so
   `--poison-index` works through `UpstreamAgent` with no special-casing.
2. `MockAgent` is rejected with a clear message pointing at `StaticAgent`
   (it has no tokenizer, so its tokens have no byte image).
3. Step records survive `json.dumps` — no numpy scalars leak into
   `steps.jsonl` via `RunLogger`.
4. bfloat16 logits (the `config_sheaf.yaml` default) preserve top-k probability ordering: max relative deviation
   from float32 is under 5%. Stage 8 only consumes the top-k, so bf16 is safe
   here — worth stating since byte-level cover mass sums many small values.
5. The gate's tree is the tree actually used for the consensus it routes.
6. All 30 Stage 5/6/7 parity assertions against `sahf.fusion` / `sahf.robust`
   still hold after the refactors.
7. The 25 pre-existing tests pass unchanged; `git status` shows zero
   modifications outside Stage 8.

---

## 7. Still unverified

Stated plainly, because it is the biggest remaining risk:

- **No run against real downloaded model weights.** Hugging Face is not
  reachable from the environment this was built in. Tokenizer handling is tested
  against real tokenizers trained from scratch (byte-level BPE, WordPiece,
  Unigram) driving real torch forward passes, but the first `run_sheaf.py`
  invocation on actual Qwen + Llama + Gemma weights is unproven. Do not pass
  `--skip-verify` on that run — if a display scheme is mis-detected the run
  aborts rather than producing plausible nonsense.
- **`theta_D` is uncalibrated for byte space** for the same reason as `theta_H`,
  though unlike `theta_H` it at least varies usefully. Calibrate from
  `steps.jsonl`.
- **Emission precision** (~48% of emitted bytes correct at the default
  `min_support`) is measured against a *simulated* ground truth. The latency
  numbers are hard; that one is directional.
- **No KV-cache**, inherited from `HFAgent`, and Stage 8 re-encodes the full
  context every step by design — so caching matters more here than it does for
  Stages 1–7.

---

## 8. Reproducing

```bash
python audit/audit_integration.py   # 12 integration probes -> "no issues found"
python audit/audit_stage8.py        # 22 library probes     -> "no defects found"
pytest -q                           # 118 tests
```

Both audits exit non-zero on failure, so either can gate CI.

## 9. Files changed by this pass

| File | Issues |
|---|---|
| `sahf/sheaf/adapters.py` | INT-4, INT-5, INT-6, INT-7 |
| `sahf/sheaf/orchestrator.py` | INT-1, INT-2, INT-3, INT-9 |
| `sahf/sheaf/reconciler.py` | INT-1, INT-2, INT-9 (shared `node_stalks`, `mass=`/`stalks=`) |
| `sahf/sheaf/pipeline.py` | INT-1, INT-2, INT-9 (passthrough) |
| `sahf/sheaf/demo.py` | INT-8 |
| `config_sheaf.yaml`, `run_sheaf.py` | INT-3 (`byte_entropy_threshold`) |
| `tests/test_sheaf_integration_regressions.py` | new — 11 regression tests |
| `audit/audit_integration.py` | new — the harness |

**API note.** `UpstreamAgent.__init__` gains `stop_token_ids=`;
`SheafOrchestrator.__init__` gains `byte_entropy_threshold=`;
`SheafReconciler.reconcile` and `Stage8Pipeline.step_from_probs` gain optional
`mass=` / `stalks=` / `tree=`. All are additive with defaults preserving previous
behaviour, except INT-5, which deliberately changes stop detection.
