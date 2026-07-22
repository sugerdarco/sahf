# History

Dated log of every non-trivial decision made while building this, and why.
Append new entries at the top; don't edit or delete past ones — if a later
decision reverses an earlier one, say so explicitly in the new entry rather
than rewriting history.

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
