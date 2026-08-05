# Optional: documentation now out of date in pre-existing files

Stage 8 was added **without modifying any file that existed before** — that was
the constraint, and `git status` confirms it. The cost is that several existing
documents still say Stage 8 does not exist. None of it affects behaviour;
Stages 1-7 run exactly as before and `run.py` is unchanged. But anyone reading
the repo will hit these statements first, so they are listed here with suggested
replacements.

Apply with `git apply docs/stage8_docs.patch`, edit by hand, or ignore.

---

## 1. `sahf/__init__.py`, lines 2-6

> sahf_lite — fast, **single-tokenizer** variant of the SAHF distribution-fusion pipeline.
> This package intentionally implements only Stages 1-7 of the full 8-stage design
> (see ARCHITECTURE.md). **Stage 8 (Sheaf Reconciliation) is dropped** because both
> default agents share one tokenizer, so there is no vocabulary mismatch to resolve.

Now inaccurate: Stage 8 is present as `sahf.sheaf`. Suggested:

> sahf_lite — the SAHF distribution-fusion pipeline.
>
> Stages 1-7 are the fast, single-tokenizer path: with a shared tokenizer there is
> no vocabulary mismatch, so agents' distributions can be fused token-for-token.
> That is what `config.yaml`, `run.py` and `FusionOrchestrator` implement, and it
> is the right path whenever the agents' tokenizers match.
>
> Stage 8 (Sheaf Reconciliation) lives in `sahf.sheaf` and handles the case they
> do not — reconciling mismatched vocabularies through a shared byte-prefix tree.
> See `run_sheaf.py`, `config_sheaf.yaml` and `docs/STAGE8.md`.

## 2. `sahf/orchestrator.py`, line 9

> ...safe here because both default agents share one tokenizer — **no Stage 8
> byte-level reconciliation is needed**; see ARCHITECTURE.md

Still true *of this class*, but reads as though Stage 8 does not exist. Suggested
ending: "...see ARCHITECTURE.md. For agents whose tokenizers differ, use
`sahf.sheaf.SheafOrchestrator` instead."

## 3. `sahf/agents.py`, lines ~143 and ~157

`assert_shared_vocab_size`'s docstring and error message both point at
ARCHITECTURE.md's "Why no Stage 8?". The check itself is still correct and should
stay — it is exactly right for `run.py`. Only the pointer is stale. Suggested
error-message addition:

> "...Got: {details}. If this is intentional, use run_sheaf.py, which reconciles
> mismatched vocabularies at the byte level (Stage 8)."

`sahf.sheaf.assert_distinct_tokenizers` is the deliberate mirror of this function:
it flags the opposite mistake, running Stage 8 on agents that *do* share a
tokenizer, where `run.py` would be faster and would not truncate to top-k.

## 4. `README.md`, lines 5, 19-20, 84, and the section at 199

The "Why no Stage 8 (Sheaf Reconciliation)?" section ends:

> If you later want to mix models from *different* tokenizer families (e.g. a Qwen
> agent + a Llama agent), you'll need to build that piece — it's the one explicitly
> deferred in the original build-order recommendation, and for good reason: it's
> the most involved part.

That piece is now built. Suggested: retitle to "Stage 8 (Sheaf Reconciliation)"
and replace the closing paragraph with a pointer to `docs/STAGE8.md`, noting that
Stages 1-7 remain the faster path when tokenizers match.

Line 5 ("N agents (default 3) that **share a tokenizer**") should note that
mixed-tokenizer ensembles are supported via `run_sheaf.py`.

## 5. `ARCHITECTURE.md`, lines 68-76 and 135-142

> **Stage 8 (Sheaf Reconciliation) is dropped entirely.**

The paragraph's reasoning is still correct as an explanation of why Stages 1-7
can skip it; it just needs to stop claiming the stage is absent. Note also that
line 142's "with a shared tokenizer this is a non-issue" is exactly right, and
line 135's "byte→token projections (Stage 8). This is the piece with no shortcut"
now has an implementation to point at.

---

## What was deliberately NOT changed, and why

`assert_shared_vocab_size` stays as-is. It is not stale — it is the correct guard
for `run.py`, and weakening it would let mismatched tokenizers reach Stages 1-7,
which is the silent-garbage failure it exists to prevent. `run_sheaf.py` simply
does not call it.

`config.yaml` stays as-is. Its three Qwen sizes share a tokenizer, which is right
for the fast path. Stage 8 settings live in `config_sheaf.yaml`.

`requirements.txt` stays as-is. Stage 8's runtime needs are numpy (already a torch
dependency) and nothing else; `tokenizers` is used only by two test files and the
benchmarks, and both skip cleanly when it is absent.

`pytest.ini` stays as-is — `testpaths = tests` collects the new tests already.
