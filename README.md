# SAHF Distribution Fusion — cross-tokenizer ensembling

A running implementation of the SAHF pipeline for ensembling N language models
**whose tokenizers do not agree**.

That constraint is the whole point. When agents share a tokenizer you can average
their next-token distributions directly, because "token 4711" denotes the same
string to everyone. When they don't, those vectors are indexed by unrelated
vocabularies and aren't even the same length — fusion isn't inaccurate, it's
undefined. Stage 8 (Sheaf Reconciliation) builds the shared byte-level space that
makes it defined, and Stages 1/5/6/7 then run *inside* that space, per tree node.

> **Scope.** This repository is cross-tokenizer only. An earlier single-tokenizer
> variant (`FusionOrchestrator`, `run.py`, `config.yaml`, `demo_mock_run.py`) has
> been removed — with one tokenizer there is no mismatch to resolve.
> `ARCHITECTURE.md` and `HISTORY.md` are kept as the historical record of that
> design and describe behaviour that no longer exists here.

---

## Quickstart

Offline — no models, no GPU, no downloads:

```bash
pip install -r requirements.txt
python demo_sheaf_mock_run.py       # 3 different tokenizers, 1 poisoned agent
python -m sahf.sheaf.demo           # three focused scenarios
pytest -q                           # 118 tests
python audit/audit_stage8.py        # 22 library probes
python audit/audit_integration.py   # 13 integration probes
```

With real models (edit `config_sheaf.yaml` first):

```bash
python run_sheaf.py --prompt "Explain the water cycle in two sentences."
python run_sheaf.py --prompt "..." --poison-index 2 --poison-mode invert
```

Optionally prebuild the byte-prefix tree. It depends only on the ensemble's
vocabularies, never on the prompt, so it can be built once and reused — though
per-step construction is cheap and is the default:

```bash
python build_prefix_tree.py         # writes artifacts/prefix_tree.npz
```

---

## Layout

```
sahf/
  amplitude.py    Stage 1 — psi = sqrt(softmax(z))
  gate.py         Stage 2 — divergence gate
  fusion.py       Stage 5 — chordal mean on the sphere
  robust.py       Stages 6, 7 — MAD screen, Weiszfeld geometric median
  agents.py       HFAgent, MockAgent, PoisonedAgentWrapper
  logger.py       out/runs/<ts>/{meta,steps.jsonl,result}
  sheaf/          Stage 8
    vocab.py        token id -> raw bytes; 4 tokenizer display schemes
    prefix_tree.py  union byte-prefix tree, persistence, sparse propagation
    reconciler.py   local sections, Stages 5/6/7 per node, gluing
    adapters.py     UpstreamAgent bridge, StaticAgent, DirectHFAgent
    pipeline.py     Stage8Pipeline and per-step diagnostics
    orchestrator.py SheafOrchestrator — the per-token decode loop
    demo.py         three offline scenarios

run_sheaf.py                      generate from a prompt
build_prefix_tree.py              optional one-time tree artifact
run_batch_prompts.py              many prompts, models loaded once
run_deepen_benchmark.py           one dataset (gsm / mmlu / arc), fine-grained control
run_full_evaluation_experiment.py full clean-vs-poisoned sweep + plots
demo_sheaf_mock_run.py            offline example run

config_sheaf.yaml   the only config; models, thresholds, Stage 8 settings
audit/              two adversarial audit harnesses, non-zero exit on defect
benchmarks/         tokenizer training + isolated Stage 8 benchmark
docs/               Stage 8 design, audits, benchmark results
SAHF_ARCHITECTURE_SPEC.md   the full 8-stage architecture
```

---

## How a token is produced

```
context TEXT — each agent encodes it with its OWN tokenizer
  → next-token distribution per agent, over its own vocabulary
  → Stage 1: psi = sqrt(softmax(z))
  → push each agent's top-k onto a shared byte-prefix tree
  → Stage 2 gate, in byte space: entropy + divergence over ALL active nodes
      ├─ agree     → Path A: mean fusion per node
      └─ contested → Path B: per node, mean → MAD screen → geometric median
  → glue the fused conditionals down from the root
  → emit the byte chunk a majority of agents can still see ahead of
  → append as text; every agent re-encodes for the next step
```

The organising principle is unchanged from the original design: **gate before
communicating, mean before geometric median** — escalate only on evidence that
the cheaper tool failed. In this system that judgement is made per tree node
rather than per token.

---

## Things that will bite you

**Verify byte extraction.** Vocabularies store tokens in a display form (`Ġthe`,
`▁the`, `##ing`), not bytes, and scheme detection is a heuristic. A wrong guess
does not raise — it shifts every token by one leading space and builds a
plausible, silently misaligned tree. `run_sheaf.py` and the evaluation runners
verify each agent up front and abort on failure. Don't pass `--skip-verify`.

**Stop tokens have no byte image.** They cannot exist in the byte space, so their
mass is captured before renormalisation and surfaced separately. Note that a
model's end-of-turn token is not always the one its tokenizer nominates as EOS —
Llama-3 generates `<|eot_id|>` while `eos_token_id` is `<|end_of_text|>`.

**N ≥ 3.** Stage 7's guarantee is that an honest majority overrules a corrupted
minority. At N = 2 there is no majority on either side; the code runs and the
claim is simply untrue.

**Thresholds are uncalibrated.** `θ_H`, `θ_D`, `byte_entropy_threshold` and the
MAD multiplier were chosen by reasoning, not swept against a validation set.
Entropy and divergence are logged every step so they can be calibrated from
`out/runs/*/steps.jsonl`.

**The entropy half of the gate barely discriminates in byte space.** Entropy over
256 bytes tops out at ln(256) ≈ 5.55 against ln(151000) ≈ 11.9 for a token
vocabulary. Divergence carries the routing decision in practice. See
`docs/STAGE8_INTEGRATION_AUDIT.md`.

**No KV-cache**, and each step re-encodes the full context by design, so this
costs more here than it would with a shared tokenizer.

---

## Documentation

| | |
|---|---|
| `SAHF_ARCHITECTURE_SPEC.md` | the full 8-stage architecture: inputs, outputs, parameters, invariants |
| `docs/STAGE8.md` | how Stage 8 works and how it reuses Stages 1/2/5/6/7 |
| `docs/STAGE8_PREBUILT_TREE.md` | the prefix-tree artifact and sparse propagation |
| `docs/STAGE8_AUDIT_REPORT.md` | library audit — 9 defects found and fixed |
| `docs/STAGE8_INTEGRATION_AUDIT.md` | integration audit — 9 issues, 8 fixed |
| `docs/STAGE8_BENCHMARK_RESULTS.md` | isolated Stage 8 timing |
| `ARCHITECTURE.md`, `HISTORY.md` | historical record of the removed single-tokenizer design |
