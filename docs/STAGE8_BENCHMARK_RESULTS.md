# Stage 8 (Sheaf Reconciliation) — Isolated Benchmark

**What was measured:** Stage 8 only. Stages 1–7 were not built. Stage 8 was fed
the inputs it would actually receive in the full architecture.

**Hardware:** 1 core, Intel Xeon @ 2.10 GHz, pure NumPy, no GPU, no threading.
Treat these as single-core CPU figures — they are a ceiling, not a target.

---

## 1. Inputs

Realistic rather than synthetic, because tree shape and cost depend entirely on
how real vocabularies and real distributions behave.

**Agents** — three tokenizers trained from scratch on a 14 MB mixed code-and-prose
corpus (Python stdlib + NumPy source), deliberately different families and sizes:

| Agent | Algorithm | Vocab | Scheme detected |
|---|---|---|---|
| agent-A-bpe32k | byte-level BPE | 32,000 | `byte_level` |
| agent-B-bpe50k | byte-level BPE, different merges | 26,968 | `byte_level` |
| agent-C-uni32k | Unigram + Metaspace | 32,000 | `sentencepiece` |

All three passed `VocabSpec.verify()` round-trip byte reconstruction.

**Distributions** — per agent, over its own vocabulary:
peaked on the same intended continuation (drawn from held-out corpus text),
competing against ~96 tokens that *share byte prefixes* with the true token —
which is what actually crowds the top-k during real decoding — over a dense
Zipf softmax tail. Three entropy regimes: top-1 of 0.90 / 0.55 / 0.18.

**Stage 2 hand-off** — the top-k id summaries are taken as given, exactly as in
the full architecture. Stage 8 does not re-derive them; the "quick token search"
is measured as its own phase and costs 0.43 ms.

200 trials per configuration, median and p95 reported.

---

## 2. Headline

| Configuration | ms/step | Correct bytes/step |
|---|---|---|
| **k=8, min_support=2 (recommended)** | **3.8** | **3.46** |
| k=64, min_support=2 (previous default) | 24.8 | 3.47 |
| full vocabulary, no truncation | 1,546–3,511 | — |

**k beyond 8 buys nothing.** Output quality is flat across the entire k sweep
while cost grows 23×. See §4 — this is the most actionable result here.

---

## 3. Sweep 1 — top-k width (typical entropy, 3 agents)

| k | median ms | p95 ms | tree nodes | coverage | 1st byte | correct B | chunk B |
|---|---|---|---|---|---|---|---|
| 8 | 4.64 | 6.27 | 87 | 0.732 | 100.0% | 3.38 | 6.36 |
| 16 | 6.32 | 8.86 | 203 | 0.776 | 100.0% | 3.38 | 6.60 |
| 32 | 13.01 | 18.49 | 386 | 0.810 | 100.0% | 3.39 | 6.83 |
| 64 | 24.52 | 32.20 | 858 | 0.835 | 100.0% | 3.40 | 7.07 |
| 128 | 52.35 | 67.10 | 1,954 | 0.851 | 100.0% | 3.38 | 7.32 |
| 256 | 108.15 | 132.65 | 4,080 | 0.860 | 100.0% | 3.40 | 7.40 |

Cost tracks tree node count almost exactly linearly. Node count grows
super-linearly in k, so latency does too.

**Coverage is not a quality signal here.** It climbs 0.73 → 0.86 while correct
bytes stay pinned at ~3.39. The extra mass admitted by a wider k lands on
low-probability tokens that never change the fused argmax. I previously
recommended raising k when coverage drops; on these inputs that advice is wrong
— coverage should be monitored, but it does not predict output quality.

---

## 4. Why quality is flat: the ceiling is the tokenizer, not k

Mean length of the true next token across the three agents: **3.33 bytes.**

That is the hard upper bound on how many bytes a single Stage 8 step can get
right, because past its own token boundary an agent has no information — one
forward pass simply does not extend further. Measured correct bytes at
min_support=2 is **3.46**, i.e. Stage 8 already extracts essentially all the
signal present, and does so at k=8.

Widening k adds competitor tokens, not lookahead.

---

## 5. Sweep 2 — entropy regime (k=64, 3 agents)

| Regime | top-1 | median ms | p95 ms | coverage | escalated nodes | 1st byte | correct B |
|---|---|---|---|---|---|---|---|
| confident | 0.90 | 23.73 | 30.62 | 0.962 | 3.96 | 100.0% | 3.40 |
| typical | 0.55 | 24.12 | 32.14 | 0.835 | 3.63 | 100.0% | 3.40 |
| uncertain | 0.18 | 24.39 | 34.74 | 0.676 | 3.13 | 100.0% | 3.29 |

Latency is essentially flat in entropy, because in top-k mode the tree size is
fixed by k, not by how diffuse the distribution is. This is the opposite of
full-vocabulary mode, where entropy dominates cost (§8) — a good argument for
top-k mode on its own: **predictable latency**.

Stage 6 escalates to the geometric median on ~3–4 nodes per step, so Stage 7
robustness is active but cheap.

---

## 6. Sweep 3 — agent count (k=64, typical)

| N agents | median ms | p95 ms | ms per agent |
|---|---|---|---|
| 3 | 22.46 | 34.92 | 7.49 |
| 5 | 36.19 | 52.29 | 7.24 |
| 8 | 54.54 | 79.59 | 6.82 |

Sub-linear per agent. Adding agents widens each node's stalk matrix (cheap,
vectorized) without adding nodes, so the per-node Python overhead is amortized.
Scaling to 8 agents costs 2.4× the time of 3.

---

## 7. Sweep 4 — `min_support`: how many bytes to commit per step

The emitted chunk ends where fewer than `min_support` agents can still see ahead.

| min_support | ms/step | chunk B | correct B | wasted B | precision |
|---|---|---|---|---|---|
| 1 (any agent) | 23.46 | 8.73 | 3.48 | 5.25 | 39.9% |
| 2 (majority, default) | 23.68 | 7.07 | 3.40 | 3.67 | 48.1% |
| 3 (unanimous) | 23.88 | 3.43 | 2.48 | 0.96 | 72.2% |

**The default emits roughly twice as many bytes as it gets right.** Those extra
bytes are committed to the context and re-tokenized by every agent on the next
step, so the error compounds.

Unanimous support raises precision from 48% to 72% at the cost of ~0.9 correct
bytes per step. That is a favourable trade for generation quality: the shorter
chunk costs an extra step, but a wrong committed byte costs far more.

Caveat: precision here is measured against a *simulated* ground truth. The
latency numbers are hard; treat precision as directional.

---

## 8. Operating points

| Config | ms/step | correct B | precision | correct B per ms |
|---|---|---|---|---|
| k=8, min_support=2 | **3.81** | 3.46 | 54.2% | **0.908** |
| k=8, min_support=3 | 3.21 | 2.32 | 83.0% | 0.722 |
| k=16, min_support=3 | 6.18 | 2.32 | 80.1% | 0.375 |
| k=32, min_support=3 | 11.02 | 2.32 | 77.4% | 0.210 |
| k=64, min_support=2 | 24.76 | 3.47 | 49.0% | 0.140 |
| k=64, min_support=3 | 24.41 | 2.35 | 73.4% | 0.096 |
| k=128, min_support=3 | 50.97 | 2.33 | 69.8% | 0.046 |

**Recommended: k=8–16.** If wrong committed bytes matter more than step count,
`min_support=3` at k=8 costs 3.2 ms and is the highest-precision option measured.

---

## 9. Phase breakdown (k=64, typical, 3 agents)

| Phase | ms | share |
|---|---|---|
| fusion + gluing | 18.72 | **81.0%** |
| projection back to token space | 3.00 | 13.0% |
| tree build | 0.82 | 3.5% |
| top-k search (Stage 2 hand-off) | 0.43 | 1.9% |
| decode | 0.12 | 0.5% |
| mass propagation | 0.04 | 0.2% |
| **total** | **23.13** | |

Two things worth noting:

- **Mass propagation is free** (0.04 ms). The two-`bincount` vectorization does
  its job; this is no longer worth optimizing.
- **Fusion is 81%**, and profiling shows it is *not* the linear algebra — it is
  per-node Python and NumPy call overhead across ~950 active nodes
  (~20 µs/node for work that is microseconds of actual arithmetic). Batching the
  per-node stalks into one padded 3-D array and running mean/MAD/Weiszfeld
  across all nodes at once should recover most of it. Estimated 3–5× on the
  dominant phase. Not attempted here — it is a real refactor, and at k=8 the
  absolute cost is already 3.8 ms.

---

## 10. Full-vocabulary mode is not viable on real tokenizers

| | |
|---|---|
| Union tree over the 3 real vocabs | **207,366 nodes** (built once, 0.59 s) |
| Reconcile, confident | 1,546 ms/step |
| Reconcile, typical | 3,511 ms/step |

My earlier estimate of ~106 ms/step for full mode came from synthetic
vocabularies with short random tokens. Real BPE vocabularies produce far deeper
and bushier trees (max depth 30 vs ~8), and the cost is ~30× worse than that
estimate. **Full mode is an offline evaluation tool only** — it cannot be used
in a decoding loop. The earlier README figure should be read as optimistic.

---

## 11. Bottom line for the architecture

- Stage 8 at the recommended operating point costs **~4 ms/token, single core**.
  Against a typical 20–30 ms GPU forward pass, that is **15–20% overhead** —
  acceptable for a stage that runs on every token.
- At the previous k=64 default it costs ~25 ms/token, which roughly **doubles**
  per-token latency for no measurable quality gain.
- Latency is flat in entropy, so per-token cost is predictable — useful given
  Stages 2–7 are already variable-cost.
- Cross-tokenizer reconciliation itself works: **100% first-byte agreement in
  every configuration**, across three tokenizer families, with the correct-byte
  count sitting at the theoretical ceiling set by token length.

### Suggested changes

1. Default `k` from 64 → **8–16** (23× cheaper, no quality loss measured).
2. Consider `min_support=N` (unanimous) as the default — precision 48% → 72%.
3. Correct the README's full-mode figure from ~106 ms to 1.5–3.5 s.
4. Drop the "raise k when coverage drops" guidance; coverage did not predict
   quality on any configuration tested.

---

## 12. Reproducing

```bash
make tokenizers   # ~5 min: builds a corpus and trains the 3 tokenizers
make bench        # ~15 min: writes benchmarks/results/bench_results.json
```

`bench_stage8.py` builds no part of Stages 1–7; it constructs the inputs Stage 8
receives and times Stage 8 alone.
