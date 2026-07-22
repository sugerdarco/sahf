# sahf-lite

A **running prototype** of the SAHF distribution-fusion pipeline, simplified to
the case that's actually buildable on one machine right now: **N LLM agents
(default 3) that share a tokenizer**, fused token-by-token. See
`ARCHITECTURE.md` for exactly what that simplifies away, and `HISTORY.md` for a
dated log of every design decision and why it was made.

> **N=3 by default, not 2.** Stage 7's whole point — surviving a corrupted
> agent — only means something with an honest *majority*, which needs N ≥ 3.
> An earlier version of this defaulted to 2 agents; see `HISTORY.md` for the
> fix and `ARCHITECTURE.md` for why N=2 can't actually prove the claim even
> though the code runs fine at that size.

Target hardware: **1 machine, 24GB VRAM, 128GB RAM.**

## What's actually implemented

Stages 1–7 of the original 8-stage design (Stage 8, Sheaf Reconciliation, isn't
needed — see "Why no Stage 8?" below):

1. **Amplitude Interception Kernel** — ψ = √(softmax(z))
2. **Divergence Gate** — entropy + disagreement check, decides Path A vs B
3. **Path A: Fast Passthrough** — skip fusion when agents already agree
4. **Path B: collect** — (in-process, not a real network allgather — see below)
5. **Fast Mean Fusion** — cheap weighted average on the amplitude sphere
6. **Outlier Check** — median-absolute-deviation screen
7. **Escalation Tier** — Riemannian Weiszfeld geometric median

Every stage is unit-tested (`tests/`), and the full per-token decode loop is
tested end-to-end with mock agents that need no GPU or internet access
(`tests/test_orchestrator_mock.py`, `demo_mock_run.py`).

## Status of this build

Built and tested in a sandboxed environment with **no GPU and no access to
Hugging Face** (only PyPI is reachable there). So, honestly:

| Piece | Status |
|---|---|
| Math kernel (Stages 1, 2, 5, 6, 7) | ✅ 25/25 unit tests passing |
| Orchestrator loop + `out/` logging | ✅ tested end-to-end with mock agents |
| Poisoning wrapper + N=3 majority recovery | ✅ tested end-to-end via the actual `PoisonedAgentWrapper` production code path |
| Real N-model run (`run.py` + `HFAgent`) | ⬜ **not yet run anywhere** — needs your machine's GPU + internet |

The code path for real models is written and should work as-is, but you will be
the first one to actually run it. Please report back anything that breaks —
most likely candidates are listed in "Known rough edges" below.

## Setup (on your machine)

```bash
pip install -r requirements.txt
```

First real run will download ~3GB + ~6GB of model weights from Hugging Face
(no token needed — both default models are ungated).

## Quickstart — no GPU needed, right now

Sanity-check the whole pipeline (gate, fusion, outlier check, escalation,
logging) with fake models before waiting on any download:

```bash
python demo_mock_run.py
```

This writes a real run to `out/runs/run_<timestamp>_mock_demo/` — open
`steps.jsonl` there to see exactly what a poisoned-agent escalation looks like
in the logs.

## Quickstart — real models

```bash
python run.py --prompt "Explain the water cycle in two sentences."
```

Uses `config.yaml` by default: **Qwen2.5-1.5B-Instruct + Qwen2.5-3B-Instruct**
(same tokenizer family, ~9GB combined in bf16 — comfortable on 24GB VRAM).

## Choosing models for your 24GB budget

All configured models must come from the **same tokenizer family** — this is
what lets us skip Stage 8 entirely. Rough bf16 VRAM budget (weights only; add a
few GB for activations/KV-cache):

| Agents | Combined VRAM | Notes |
|---|---|---|
| Qwen2.5-0.5B + 1.5B + 3B (default, N=3) | ~10GB | Safest, fastest, good for first run |
| Qwen2.5-0.5B + 1.5B + 7B (N=3) | ~18GB | Bigger capability gap, still fits |
| Qwen2.5-1.5B + 3B + 7B (N=3) | ~23GB | Tight — leaves little headroom for KV-cache |
| Qwen2.5-3B + 3B + 14B (N=3) | ~35GB | **Too big for 24GB** — don't use |

To swap models, edit `config.yaml`'s `models:` list — nothing else needs to
change as long as every entry shares a tokenizer. The orchestrator itself
doesn't assume any particular N (it works for any N ≥ 2); N=3 is the config
default specifically because it's the smallest size where Stage 7 is actually
proving something (see the callout at the top of this file).

## Testing Stage 7 (the median stage) on your own hardware

Don't just trust that 3 honest real models will happen to disagree enough to
prove the escalation tier works — force the condition it's meant to survive:

```bash
# Baseline: all 3 configured agents run clean
python run.py --prompt "What is the capital of France?"

# Same prompt, but agent index 2 (Qwen2.5-3B) is deliberately corrupted
python run.py --prompt "What is the capital of France?" --poison-index 2 --poison-mode invert
```

`--poison-mode invert` takes that agent's real logits and negates them — a
well-formed distribution, still derived from a real model, but confidently
pushing for whatever the honest models think is *least* likely. Compare the
two runs' `out/runs/*/result.json` and `steps.jsonl`:

- `meta.json` records exactly which agent was poisoned and how (`poisoning`
  field; `null` for a clean run) — never silent.
- `steps.jsonl` shows `"escalated": true` and `"outliers": [false, false,
  true]` at the steps where the poisoned agent got caught.
- With a real honest majority, the final output text should be close to the
  clean-run baseline despite the poisoning. If it isn't, that's a real signal
  — either the gate/outlier thresholds need retuning for these specific
  models, or something in the pipeline needs a second look.

Other `--poison-mode` options: `uniform_noise` (replace the agent's logits with
pure noise) and `random_bias` (the agent insists on one arbitrary wrong token
every step) — `invert` is the default because it's the most realistic: it
doesn't look obviously broken, it just confidently disagrees.

## Running the tests

```bash
pytest -v
```

All 18 tests run on CPU with no downloads. `test_robust.py` is the one worth
reading first — it's the actual proof that the geometric median recovers the
honest-majority answer under a poisoned agent while the plain mean gets dragged
off, which is the entire justification for Stage 6/7 existing.

## Directory structure

```
sahf_lite/
├── README.md            you are here
├── ARCHITECTURE.md       what's implemented, what's simplified, why
├── HISTORY.md             dated changelog of every design decision
├── config.yaml             model pair + gate thresholds
├── requirements.txt
├── run.py                   CLI entry point (real models)
├── demo_mock_run.py           CLI entry point (mock models, no GPU/internet)
├── sahf/                       the actual package
│   ├── amplitude.py             Stage 1
│   ├── gate.py                   Stage 2
│   ├── fusion.py                   Stage 5
│   ├── robust.py                    Stage 6 + 7
│   ├── agents.py                     HFAgent, MockAgent, PoisonedAgentWrapper,
│   │                                   assert_shared_vocab_size
│   ├── orchestrator.py                per-token decode loop
│   └── logger.py                       everything below writes here
├── tests/                       one file per stage + agents/poisoning + end-to-end
└── out/                          all results and logs land here
    ├── logs/app.log               process-level log
    └── runs/run_<timestamp>_.../   one folder per run
        ├── meta.json                config + prompt used
        ├── steps.jsonl               one JSON line per decoding step
        └── result.json                 final text + summary counts
```

## Known rough edges (read before you run this for real)

- **No KV-cache.** Each step recomputes the full forward pass on the whole
  sequence so far, on *every* agent. Correct, but O(seq_len) slower per token
  than a cached implementation. Fine for short generations; will feel slow past
  a few hundred tokens. This is the top follow-up — see `HISTORY.md`.
- **Gate thresholds are not calibrated.** `entropy_threshold: 2.0` and
  `divergence_threshold: 0.05` in `config.yaml` are reasonable starting guesses,
  not fit to any real data from these specific models. Expect to tune them
  after looking at a few real runs' `steps.jsonl` — if Path B fires on almost
  every token, `divergence_threshold` is probably too tight for this agent set.
- **If you drop back to N=2, you lose Stage 7's actual guarantee.** The
  geometric median is only meaningfully Byzantine-robust with an honest
  *majority*, i.e. N ≥ 3 — this is *why* the default config now ships 3 agents
  (see the callout at the top of this file and `HISTORY.md`). The code still
  runs correctly at N=2, but "outlier" degrades to "further from the mean than
  the other one," which isn't the same claim. Don't reduce `config.yaml`'s
  `models:` list below 3 entries unless you're deliberately testing that
  degraded case.
- **"Allgather" is just a Python list comprehension.** There's no real
  distributed communication here since everything runs in one process — see
  `ARCHITECTURE.md` for what changes if you ever split agents across machines.
- **Weiszfeld iteration counts can climb.** In the mock demo it takes 40+
  iterations to hit `tol=1e-6` — not slow in wall-clock terms, but worth
  watching; loosening `tol` or warm-starting from the fast mean is an easy win
  if this ever shows up as a bottleneck.

## Why no Stage 8 (Sheaf Reconciliation)?

Because the default config uses two Qwen2.5 sizes, which share one tokenizer —
"token 4711" already means the same thing to both agents, so there's no
vocabulary mismatch to reconcile. If you later want to mix models from
*different* tokenizer families (e.g. a Qwen agent + a Llama agent), you'll need
to build that piece — it's the one explicitly deferred in the original build-order
recommendation, and for good reason: it's the most involved part.
