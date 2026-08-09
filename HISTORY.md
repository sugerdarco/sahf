# History

Dated log of every non-trivial decision made while building this, and why.
Append new entries at the top; don't edit or delete past ones — if a later
decision reverses an earlier one, say so explicitly in the new entry rather
than rewriting history.

---

## 2026-07-22 — v0.2: fix — default config didn't have enough agents to prove Stage 7's own claim

**The gap, stated precisely:** v0.1 shipped with 2 real agents by default. All
25 (then 18) tests passed, `demo_mock_run.py` ran cleanly — but Stage 7's
entire justification is "an honest majority overrules a corrupted minority,"
and 2 agents can never have a majority on either side. Tests passing at N=2
proved the *code* was correct, not that the *claim* was true. This was flagged
directly rather than found by digging, and it was right — the fix below is a
real behavior change, not just added documentation.

**Fix, part 1: `config.yaml` now defaults to 3 agents**, not 2 — Qwen2.5-0.5B +
1.5B + 3B (Instruct). Reason for this specific triple over other options: still
comfortably same tokenizer family, still small (~10GB combined bf16, safely
under the 24GB budget, barely more than the old 2-model default), and 3
meaningfully different capability tiers rather than 3 near-duplicates.
Rejected keeping N=2 and only fixing this in documentation — a caveat that
says "this doesn't really work at N=2, but N=2 is the default" is not a fix.

**Fix, part 2: added `PoisonedAgentWrapper`** (`sahf/agents.py`). Reason: even
with 3 *honest* real models, there's no guarantee they'll disagree enough,
naturally, to ever exercise Stage 6/7 at all — you could run this for a long
time and never actually see the escalation path fire, let alone verify it
recovers correctly. Rather than waiting/hoping, this wraps any agent (real or
mock) and deliberately corrupts its logits on demand, so the exact condition
Stage 7 is supposed to survive can be forced and checked, on purpose, on real
hardware. Three modes were added (`invert`, `uniform_noise`, `random_bias`);
`invert` was chosen as the CLI default because it's the most realistic
failure — a well-formed distribution that's just confidently wrong, not an
obviously-broken one, which is closer to what an actually-misbehaving model
would look like than pure noise.

**Fix, part 3: extracted the tokenizer-match check into
`assert_shared_vocab_size()`.** This existed before only as an inline set
comprehension inside `run.py`'s `main()`, untested on its own. Writing a real
unit test for it surfaced a second, smaller latent bug: the original check
used `len(agent.tokenizer)`, which would have crashed (`len(None)`) the moment
anyone tried to run it against a `MockAgent`, since mocks have no real
tokenizer object. Fixed by giving every agent type (including
`PoisonedAgentWrapper`, which forwards from whatever it wraps) a `.vocab_size`
attribute, and comparing that instead. Small thing, but exactly the kind of
gap that only shows up when you actually try to write the test rather than
trust that the check "obviously" works.

**Decided against:** a `weights:` field in `config.yaml` for trusting some
agents more than others. Still explicitly a v2 problem (see the v0.1 entry
below) — didn't want to bundle an unrelated feature into a fix for the N≥3
gap.

**New test added:**
`tests/test_agents.py::test_n3_orchestrator_survives_wrapped_poisoning_via_production_mechanism`
is the one that actually closes the original gap — 3 agents, one wrapped with
the real `PoisonedAgentWrapper` class (not a hand-crafted biased mock, unlike
the v0.1 orchestrator test), asserting the honest majority's token wins at
every step through the same code path `run.py` uses. `demo_mock_run.py` was
also rewritten to use `PoisonedAgentWrapper` instead of a manually-biased
mock, so the no-GPU demo now dogfoods the identical mechanism intended for
real runs, rather than a parallel one-off.

All 25 tests pass (18 from v0.1 + 7 new in `tests/test_agents.py`).

---

## 2026-07-22 — v0.1: initial fast-version prototype

**Scope decision: 2 agents, shared tokenizer, single machine.**
Reason: the full 8-stage design has one genuinely hard, no-shortcut piece
(Stage 8, heterogeneous-tokenizer reconciliation via byte-level tries) and one
piece that only matters once you're multi-machine (a real Stage 4 allgather).
Building a prototype that includes both before anything else worked would mean
debugging three hard problems at once. Cutting both, in that order, was the
explicit build-order recommendation from earlier in the design conversation.

**Model choice: Qwen2.5-1.5B-Instruct + Qwen2.5-3B-Instruct.**
Reason: same tokenizer family (satisfies the scope decision above), small
enough that both fit in bf16 with room to spare on 24GB VRAM (~9GB combined),
and different enough in size that fusing them is actually meaningful (not just
two copies of the same model). Documented alternative pairs (up to 1.5B+7B) in
README.md for anyone who wants a bigger capability gap and still fits in 24GB.

**No KV-cache in `HFAgent.next_logits` (v1).**
Reason: correctness and simplicity first. A cached implementation has to get
per-agent cache alignment right, which is exactly the kind of subtle bug that's
hard to notice in a first build. Recomputing the full forward pass every step
is slower (O(seq_len) per token instead of O(1)) but unambiguously correct.
Flagged in README/ARCHITECTURE as the top follow-up once the pipeline itself is
verified correct.

**Gate thresholds: `entropy_threshold=2.0` nats, `divergence_threshold=0.05`.**
Reason: reasonable starting points (2.0 nats is meaningfully below ln(vocab
size) for any real tokenizer; 0.05 is a tight-but-not-absurd Euclidean distance
on the amplitude sphere), but **explicitly not calibrated** against any real
generation from these two specific models. First thing to look at and tune once
real `steps.jsonl` data exists — see README "Known rough edges."

**Outlier threshold: MAD multiplier = 3.0.**
Reason: standard robust-statistics default (roughly analogous to a 2-sigma-ish
cutoff for a Gaussian-like spread, but robust to the outlier itself skewing the
statistic, unlike a mean/std-based threshold would be). Not yet tuned against
real model disagreement patterns.

**Weiszfeld solver: `max_iter=50`, `tol=1e-6`, init at the arithmetic mean.**
Reason: generous caps, chosen for correctness in a prototype rather than
speed. Observed in `demo_mock_run.py` that a clearly-separated poisoned agent
can take 40+ iterations to converge to `tol=1e-6` — not slow in wall-clock
terms yet, but flagged as worth revisiting (looser tolerance, or warm-starting
from the Stage 5 fast-mean result instead of the plain arithmetic mean) if it
ever becomes a real bottleneck.

**Chose to fail loudly, not silently, on tokenizer mismatch.**
Reason: `run.py` checks `len(tokenizer)` across all configured agents at
startup and raises before doing any work if they differ, since this whole
prototype's correctness depends on the shared-tokenizer assumption holding —
silently reconciling or ignoring a mismatch would produce plausible-looking but
wrong output instead of an honest error.

**Testing approach: pure-math unit tests + a mock-agent end-to-end smoke test.**
Reason: this was built in a sandbox with no GPU and no access to
huggingface.co (confirmed directly — see below — rather than assumed). Every
stage that's pure math (amplitude conversion, gate, fast mean fusion, outlier
detection, Weiszfeld median) is fully unit-tested and passing (18/18) without
needing any model weights. `MockAgent` + `tests/test_orchestrator_mock.py`
additionally exercise the *entire* orchestrator loop and the `out/` logging
end-to-end, including the specific claim that matters most — that the
geometric median recovers the honest-majority token under one poisoned agent
while the plain mean does not (`test_geometric_median_beats_plain_mean_under_one_poisoned_agent`).
The real `HFAgent` + `run.py` path is written but has not been run anywhere
yet — that first real run is on the user's machine, not this one.

**Confirmed (not assumed) that Hugging Face is unreachable from the build
sandbox:** `urllib.request.urlopen('https://huggingface.co')` → `HTTP 403`.
This is why `demo_mock_run.py` exists as a separate, no-download entry point —
it lets the pipeline be sanity-checked on a new machine before spending time on
a multi-GB model download.

**N=2 vs. N=3 in tests.** The two agent-fusion tests
(`test_two_agreeing_mock_agents_...`, `test_disagreeing_mock_agents_...`) use
N=2 to match the default real-run config. The escalation/robustness test
(`test_three_agents_one_poisoned_...`) deliberately uses N=3, because the
geometric median's breakdown-point guarantee requires an honest majority, which
doesn't exist at N=2. Documented explicitly in ARCHITECTURE.md so this isn't
mistaken for an oversight later.

**`out/` layout: `logs/` for process-level logging, `runs/<timestamp>/` per
generation.** Reason: `steps.jsonl` (one JSON object per line, not one big JSON
array) so a run can be tailed while still in progress and a crash mid-generation
doesn't corrupt everything collected so far. Kept `meta.json` (config used) and
`result.json` (final summary) separate from the step-by-step trace so a quick
"how did this run go" check doesn't require parsing the full trace.

---

## Cross-tokenizer only — single-tokenizer variant removed

The project now targets exclusively ensembles whose agents do **not** share a
tokenizer. The single-tokenizer pipeline has been deleted rather than kept
alongside: with one tokenizer there is no vocabulary mismatch to resolve, and
maintaining two pipelines meant every stage had two behaviours to reason about.

**Removed**

- `sahf/orchestrator.py` (`FusionOrchestrator`) — fused in token space, which is
  only meaningful under a shared vocabulary.
- `run.py`, `config.yaml`, `demo_mock_run.py` — that pipeline's entry points.
- `assert_shared_vocab_size` and its tests — the guard existed to protect the
  shared-vocabulary assumption, which no longer applies. Its deliberate mirror,
  `sahf.sheaf.assert_distinct_tokenizers`, flags the opposite mistake.
- `tests/test_orchestrator_mock.py` — covered the removed orchestrator. The
  equivalent robustness claim (honest majority survives a poisoned agent at N=3)
  is covered for the current pipeline by
  `tests/test_sheaf_orchestrator.py::test_path_b_escalates_on_a_byzantine_agent`.
- `tests/test_sheaf.py` — dead. It imported `SoftVocabularyMapper` and a
  `SheafReconciler(agents, reference_agent_index=...)` signature, neither of
  which exists anywhere in the package; it was left behind by an earlier Stage 8
  design and broke collection for the whole suite.

**Ported rather than removed.** `run_batch_prompts.py`, `run_deepen_benchmark.py`
and `run_full_evaluation_experiment.py` all used `FusionOrchestrator`, but their
*function* — batch prompting, single-dataset evaluation, and the full
clean-vs-poisoned sweep — is not tied to a shared tokenizer. All three now use
`SheafOrchestrator` with `UpstreamAgent`, verify byte extraction per agent before
starting, and read `config_sheaf.yaml`. Their record schemas keep the original
field names so existing summaries and plots still read.

**Two defects found while doing this**

1. All three evaluation runners defaulted to `--config config_local.yaml`, a file
   that has never existed in the repository. They would have failed unless a
   config was passed explicitly. Now defaulted to `config_sheaf.yaml`.
2. `config_sheaf.yaml` described the wrong tokenizers: TinyLlama was labelled
   byte-level BPE ~128k and SmolLM2 sentencepiece ~256k. In fact TinyLlama is
   SentencePiece ~32k and SmolLM2 is byte-level BPE ~49k — the labels were
   left over from an earlier model list. Corrected.

**Worth recording as an open question.** Every evaluation run committed under
`out/runs/` names three Qwen snapshots (largely the same 3B path repeated) and
carries no `stage8` flag, and the runners as committed called
`assert_shared_vocab_size`, which would have rejected a mixed-family ensemble
outright. Whatever produced the cross-tokenizer benchmark numbers is therefore
not reproducible from this branch as it stood. The ported runners are, but the
existing result files should not be assumed to have come from the Stage 8 path.

**Kept.** `ARCHITECTURE.md` and this file remain as the historical record;
`ARCHITECTURE.md` now carries a banner saying so. `sahf/amplitude.py`,
`gate.py`, `fusion.py`, `robust.py`, `agents.py` and `logger.py` are all retained
unchanged apart from the removed guard — Stage 8 uses every one of them.

