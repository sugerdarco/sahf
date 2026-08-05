# The prefix tree is built once, not per query

Correcting how Stage 8 was originally wired. The byte-prefix tree is a property
of the ensemble's **vocabularies**, not of any prompt: it does not change between
queries, between runs, or between decoding steps. It is now built by a separate
command, saved as an artifact, and loaded by the pipeline.

```bash
python build_prefix_tree.py            # once, whenever `models` changes
python run_sheaf.py --prompt "..."     # loads artifacts/prefix_tree.npz
```

---

## What was wrong

The original integration defaulted to `mode="topk_union"`, which **rebuilt the
trie on every decoding step** from that step's top-k tokens. It was fast, but it
is the wrong architecture: it rediscovers a fixed structure thousands of times
per generation, and the tree it builds is only ever a keyhole view of the
vocabularies rather than the real shared base space.

The alternative already present, `mode="full"`, built one tree per *process* —
never persisted, and it propagated the entire vocabulary through it every step,
which measured at **6.6 seconds per step**. Unusable, and correctly labelled so.

Neither was "build once, reuse".

---

## What changed

**1. The tree is serializable.** `BytePrefixTree.save()` / `.load()` write a
single `.npz`: the trie (CSR form), each token's precomputed root-to-leaf path,
and the extracted token→bytes tables. Because the vocabularies travel with the
tree, a run never loads a tokenizer just to reconcile bytes. `_terminals` is not
stored — it is recoverable from the end-node arrays.

**2. A prebuilt tree propagates sparsely.** `cover_mass(agent, probs,
restrict=ids)` walks only the given tokens' precomputed paths, so a step costs
O(k × depth) and does not grow with the size of the tree. This is what makes a
persisted full-vocabulary tree usable at all — propagating 150k tokens through a
200k-node trie every step is the 6.6 s figure above.

**3. `mode="prebuilt"`.** `Stage8Pipeline(agents, mode="prebuilt", tree=...)` and
`SheafOrchestrator(..., tree=...)`. `run_sheaf.py` loads the artifact by default
and **refuses to start** if it was built for a different model list — reconciling
against the wrong vocabularies would produce silently wrong output. `--no-tree`
falls back to per-step rebuilding for debugging.

---

## Making it actually fast

A prebuilt tree was initially **slower per step than rebuilding a small one**
(15.8 ms vs 8.3 ms), which is worth recording because the first two explanations
were both wrong.

| Hypothesis | Fix | Effect |
|---|---|---|
| Stalk vectors padded with structural zeros (a full-tree node has hundreds of children, a step touches a handful) | `live_children` drops zero-cover children | 15.8 → 15.6 ms |
| Materialising each node's children from a Python dict via `np.fromiter` | CSR child arrays, sliced as O(1) views | 15.6 → 16.1 ms |
| **`reconcile` allocated two `n_nodes`-sized arrays per step** — 1.6 MB zeroed twice per token, invisible on a 196-node tree, dominant on a 206k-node one | reusable scratch buffers, clearing only the candidate entries | **16.1 → 10.8 ms** |

Only the third mattered. The first two were kept anyway: they are correct, and
they matter more as vocabularies grow.

Both fixes that reuse buffers (`cover_mass(restrict=...)` and `glue_scratch`)
return arrays owned by the tree, valid until the next step. That is documented on
both methods; `GlobalSection` only reads them within the step that produced it.

### Current cost

Three real tokenizers (byte-level BPE 32k / BPE 32k / Unigram 32k), 206,775-node
union tree, single core:

| | |
|---|---|
| build (one-time) | 0.91 s |
| save | 1.30 s → **4.3 MB artifact** |
| load (once per run) | 1.09 s |

| per step | ms |
|---|---|
| prebuilt, k=16 | **12.9** |
| prebuilt, k=64 | 51.3 |
| topk_union, k=16 (rebuild each step) | 10.5 |
| full (propagate whole vocab) | 6,583 |

Rebuilding a keyhole tree is still marginally cheaper per step, because a 196-node
tree has better cache behaviour than a 206k-node one. That is not a reason to
prefer it: the prebuilt tree is the real shared base space, its cost does not
depend on the prompt, and the gap is ~2 ms against a 20–30 ms forward pass. With
production vocabularies (Qwen 151k + Llama 128k + Gemma 256k) the per-step build
in `topk_union` grows while the prebuilt path does not.

**Both modes now produce identical consensus on 25/25 cases.**

---

## A real bug this surfaced

The two modes originally disagreed on 1 of 25 cases. It was an exact tie — two
byte strings at probability 0.471886 — broken by whichever order the tree
happened to iterate a node's children. A prebuilt tree and a per-step tree
enumerate children differently, so the same input could yield different output.

`decode` now breaks ties by byte string, not insertion order. Determinism should
not depend on where the tree came from.

A second bug came from the same work: the Stage 2 gate filtered zero-cover
children while `reconcile` did not, so a cached stalk of width 5 met a child list
of width 6 and failed as a raw shape error deep in the fusion loop. The stalk
cache now carries the geometry it was computed against, so the two cannot drift.

---

## Rebuild when the vocabularies change

The artifact is only valid for the exact model list it was built from.
`run_sheaf.py` compares `[v.name for v in tree.vocabs]` against `config["models"]`
and exits with an explicit message on a mismatch, rather than reconciling against
the wrong vocabulary. `save`/`load` also carry a format version and refuse a
stale artifact.

Byte extraction is verified at **build** time — once, in the command that has the
tokenizers loaded anyway — rather than at the start of every run.
