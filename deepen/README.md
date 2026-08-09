# DeePEn benchmarking

Everything for evaluating SAHF against **DeePEn** (Huang et al., NeurIPS 2024,
arXiv:2404.12715) lives here, separate from the pipeline itself. Nothing in
`sahf/` depends on this folder.

DeePEn is the closest published system solving the same problem — ensembling
models whose tokenizers do not agree. It does it by projecting each model's
distribution into a shared relative-representation space via dense transfer
matrices; SAHF does it by reconciling raw bytes in a prefix tree. These scripts
run SAHF over DeePEn's own benchmark datasets so the two are measured on the
same tasks.

```
deepen/
  run_deepen_benchmark.py            one dataset, fine-grained control
  run_full_evaluation_experiment.py  full clean-vs-poisoned sweep + plots
  datasets/                          DeePEn's benchmark data (NOT vendored — see below)
  results/
    verbose/                         per-question generations and step traces
    charts/                          generated comparison plots
    *_eval_summary.json              run summaries
```

## Getting the datasets

**The benchmark data is not vendored in this repository.** Fetch it from the
DeePEn project and place it at `deepen/datasets/`, keeping DeePEn's own layout:

```
deepen/datasets/
  GSM/data/test.cleand.jsonl
  MMLU/dev-jsonl/<category>.jsonl      (test-jsonl/ is used as a fallback)
  ARC-Challenge/test.jsonl
```

`deepen/datasets/` is preferred; the repository root is checked as a fallback, so
an existing top-level `datasets/` directory keeps working.

## Running

Both scripts work from the repository root or from inside this folder — paths are
anchored to the script, not to the working directory. They read
`config_sheaf.yaml` at the repository root by default.

```bash
# one dataset, with category and poison control
python deepen/run_deepen_benchmark.py --dataset gsm  --max-questions 10
python deepen/run_deepen_benchmark.py --dataset mmlu --category elementary_mathematics --max-questions 5
python deepen/run_deepen_benchmark.py --dataset arc  --max-questions 5 --poison-index 2

# full sweep: every dataset, clean and poisoned, with plots
python deepen/run_full_evaluation_experiment.py --num-samples 20
```

Summaries, per-question generations and charts are written to `deepen/results/`.
Per-step run logs go to the shared `out/runs/` tree at the repository root, in the
same format as every other run, so existing log tooling reads them unchanged.

## Reading the results with appropriate caution

Two things about the committed results are worth knowing before citing them.

**The accuracy comparison is not settled.** On ARC-Challenge SAHF answered
correctly where DeePEn produced no parseable option letter, which is the strongest
result in hand — but a 0% is a parsing outcome as much as a model outcome, and it
should be re-scored with a lenient extractor and the raw generations read before
it is relied on. On GSM8K, DeePEn nominally scored higher on a smaller sample.
Both sample sizes are small enough that a single question moves the number
several points, and the models involved (0.36B–1.5B) sit near the floor on GSM8K
regardless of the ensembling method.

**Provenance of the existing result files is uncertain.** Every run recorded under
`out/runs/` names three Qwen snapshots and carries no `stage8` flag, and the
evaluation runners as previously committed asserted a shared vocabulary — which
would have rejected a mixed-tokenizer ensemble outright. Whatever produced the
cross-tokenizer numbers is therefore not reproducible from the branch as it stood.
The scripts here now run the Stage 8 path and verify byte extraction per agent
before starting, so results produced from this point are reproducible; the older
files should not be assumed to have come from that path.

## Latency

Comparative timings are in `../docs/STAGE8_BENCHMARK_RESULTS.md`. Note that the
end-to-end figure includes three model forward passes and has not been decomposed
into forward-pass time versus Stage 8 reconciliation time — so it is a measure of
the whole pipeline, not of the reconciliation method on its own.
