# SAHF Distribution Fusion — Architecture Specification

Per-token, N-agent distribution fusion.

N language models each produce a next-token distribution over their own
vocabulary. The system combines them into one consensus token, per token, paying
cost proportional to how contested that token actually is.

This document specifies the architecture only: what each stage receives, what it
produces, what governs it, and what must hold for it to be correct. No
implementation detail, no benchmarks.

---

## 1. Design principle

**Cascading cost.** Every stage — including the gate itself — asks whether a
cheaper tool is good enough for *this* token, and escalates only on evidence of
failure.

| Escalation | Cheap tier | Expensive tier | Trigger |
|---|---|---|---|
| Communication | local summaries | full distribution exchange | gate disagreement |
| Fusion | arithmetic mean | geometric median | outlier detected |
| Vocabulary | shared token ids | byte-prefix reconciliation | tokenizers differ |

The rule is uniform: **never pay for machinery without evidence it is needed.**
The corollary matters as much — every escalation must be *detectable cheaply*, or
the gate is not honest.

---

## 2. Pipeline

```
                  N ≥ 3 AGENTS — PARALLEL FORWARD PASS
                                │
                                ▼
             STAGE 1 — Amplitude Interception Kernel
                        ψᵥ = √(softmax(z)ᵥ)
                                │
                                ▼
                  STAGE 2 — Divergence Gate
              cheap top-k summaries per agent
                                │
              ┌─────────────────┴─────────────────┐
            AGREE                              CONTESTED
              │                                    │
              ▼                                    ▼
   STAGE 3 — Path A                    PATH B — FUSION PIPELINE
   Fast Passthrough                     STAGE 4 — Allgather ψ
              │                                 │
              │                          STAGE 5 — Fast Mean Fusion
              │                                 │
              │                          STAGE 6 — Outlier Check
              │                          ┌──────┴──────┐
              │                    NO OUTLIER      FLAGGED
              │                          │             │
              │                     use Stage 5   STAGE 7 — Escalation
              │                          │        (Geometric Median)
              └─────────────┬────────────┴─────────────┘
                            ▼
              STAGE 8 — Sheaf Reconciliation
          byte-prefix tree resolves vocab mismatch
                            │
                            ▼
              FINAL CONSENSUS TOKEN — EMITTED
```

---

## 3. Two configurations

The pipeline has two forms, selected by one question: **do the agents share a
tokenizer?**

| | Homogeneous | Heterogeneous |
|---|---|---|
| Precondition | all agents share one tokenizer | tokenizers differ |
| Comparison space | token ids | byte prefixes |
| Stages 1–7 operate on | full vocabulary vectors | per-node byte conditionals |
| Stage 8 | not required | required, and Stages 5/6/7 run inside it |
| Feedback to agents | one shared token id | consensus **bytes**, re-encoded per agent |
| Emitted unit | one token | a byte chunk (one or more bytes) |

**Ordering note.** The pipeline diagram places Stage 8 last. That reflects its
role as the translation back into token space. But in the heterogeneous
configuration, Stage 8 must also come *first*: the agents' amplitude vectors are
indexed by different token ids and are not the same length, so there is no fused
object to reconcile — the fusion cannot be computed until the shared space
exists. Stage 8 therefore **brackets** Stages 5/6/7 rather than following them:
it creates the base space before, and projects back after.

---

## 4. Stages

### Stage 0 — Agent ensemble

| | |
|---|---|
| **Purpose** | produce N independent next-token distributions over the same context |
| **Input** | context (token ids if homogeneous, text if heterogeneous) |
| **Output** | N logit vectors `z⁽ⁱ⁾ ∈ ℝ^{Vᵢ}` |
| **Parameters** | `N` (agent count), model identities, fusion weights `wᵢ`, precision, sampling temperature |

**Constraint — N ≥ 3.** Stage 7's guarantee is that an honest majority overrules a
corrupted minority. At N = 2 there is no majority on either side: outlier
detection can only say "one of you is further from the average," which is a
strictly weaker claim. N = 2 runs correctly and demonstrates nothing.

**Weights** `wᵢ` express relative trust in each agent. They must be non-negative
and are normalised at point of use. Uniform weights are the default; unequal
weights are meaningful for Stages 5 and 7 but do not change any threshold.

**Heterogeneous configuration:** each agent encodes the *shared context text* with
its own tokenizer. There is no shared id sequence.

---

### Stage 1 — Amplitude Interception Kernel

**ψᵥ = √( softmax(z)ᵥ )**

| | |
|---|---|
| **Purpose** | move from probabilities to a space where distance is statistically meaningful |
| **Input** | logits `z⁽ⁱ⁾` |
| **Output** | amplitude vectors `ψ⁽ⁱ⁾` on the unit hypersphere |
| **Parameters** | none (temperature, if used, is applied to logits beforehand) |
| **Guarantees** | `‖ψ‖₂ = 1`; `p = ψ²` recovers the distribution (Born rule) |

Euclidean distance between √p vectors is the **Hellinger distance**; their dot
product is the **Bhattacharyya coefficient**. Averaging becomes geometrically
meaningful rather than an ad-hoc heuristic on raw probabilities.

**This must run first.** Every later stage — gate, mean, median — assumes a metric
space. Nothing downstream is defined on raw softmax output.

---

### Stage 2 — Divergence Gate

**H = local entropy;  D = spread over the top-k union**

| | |
|---|---|
| **Purpose** | decide whether this token needs full fusion, using only cheap data |
| **Input** | per-agent summary: local entropy `Hᵢ`, top-`k` candidate ids and probabilities |
| **Output** | routing decision `{agree, contested}`, plus `(H, D)` for logging |
| **Parameters** | `k` (summary width), `θ_H` (entropy threshold), `θ_D` (divergence threshold) |
| **Cost** | O(N·k) — independent of vocabulary size |

**Rule:** `agree ⟺ H < θ_H AND D < θ_D`

Both conditions are required. `H` asks *is everyone individually confident*; `D`
asks *do they agree with each other*. Confident-but-divergent and
agreeing-but-uncertain both route to fusion.

**What makes it honest:** the gate must decide whether to pay for the allgather
using data that is itself cheap to obtain. If the gate inspects full
distributions, it has already paid the cost it exists to avoid. In a
single-machine deployment full vectors are already local and this distinction is
invisible; **in a distributed deployment it is the entire point.**

**Heterogeneous configuration:** the gate operates on byte-level sections. `H` is
the entropy of the consensus distribution over the next byte. `D` is the maximum
pairwise spread **over every active node of the tree, not the root** — byte-level
tokenizers place a leading space on most word-initial tokens, so distributions
that disagree completely can share an identical root section and appear unanimous
at depth 0.

**Threshold scale is space-dependent.** Entropy over a 150k-token vocabulary has
ceiling ln(150000) ≈ 11.9; over 256 bytes it is ln(256) ≈ 5.55. `θ_H` does not
transfer between the two and must be a separate parameter.

---

### Stage 3 — Path A: Fast Passthrough

| | |
|---|---|
| **Purpose** | emit immediately when agents already agree |
| **Input** | the gate's own summaries |
| **Output** | consensus token |
| **Parameters** | none |

Path A **never touches the agents' full distributions**. The winning token is
read off the top-k comparison the gate already performed. In a distributed
deployment only a small digest crosses the network instead of a V-dimensional
vector.

Agreement is the common case for most tokens in most sequences, which is what
makes the whole design economical.

**Heterogeneous configuration:** Path A means "fuse with the plain mean; skip the
outlier screen and the robust estimator." It cannot skip constructing the shared
byte space, because without it there is nothing to read a token off.

---

### Stage 4 — Allgather ψ Vectors

| | |
|---|---|
| **Purpose** | make every agent's full distribution visible in one place |
| **Input** | `ψ⁽ⁱ⁾` held at each agent |
| **Output** | all `ψ⁽ⁱ⁾` co-located |
| **Parameters** | transport configuration only |
| **Cost** | O(N·V), paid only on contested tokens |

No combination of distributions is computable until they are mutually visible.
There is no trick that avoids this once the gate has decided fusion is required —
the only saving available is *not entering this stage*, which is Stage 2's job.

**Deployment note:** if agents are co-located, this stage is a local gather with
no communication cost, and the latency argument motivating Stage 2 weakens
accordingly. Stage 2's value in a single-process deployment is the skipped
*arithmetic*, not skipped bandwidth.

---

### Stage 5 — Fast Mean Fusion

**ψ\* = normalize( Σᵢ wᵢ ψ⁽ⁱ⁾ )**

| | |
|---|---|
| **Purpose** | cheapest real fusion, tried first |
| **Input** | `{ψ⁽ⁱ⁾}`, weights `wᵢ` |
| **Output** | fused amplitude `ψ*` |
| **Parameters** | `wᵢ` |
| **Cost** | O(V), closed form, no iteration |

A chordal-mean approximation to the spherical Fréchet mean: average the amplitude
vectors, then re-project onto the sphere by renormalising.

**Valid when** agents disagree honestly and moderately — the small-angle regime,
where the chord approximates the arc. It is **not** robust: a single adversarial
agent can drag the mean arbitrarily far. Its breakdown point is 0. Stage 6 exists
precisely to test whether that assumption held.

---

### Stage 6 — Outlier Check

**‖ψ⁽ⁱ⁾ − ψ\*‖ ≫ median spread ?**

| | |
|---|---|
| **Purpose** | test whether the mean can be trusted, before paying for robustness |
| **Input** | `{ψ⁽ⁱ⁾}`, `ψ*` |
| **Output** | outlier mask over agents, and the distance vector |
| **Parameters** | `λ` (MAD multiplier), MAD floor |
| **Cost** | O(N) |

**Rule:** agent `i` is flagged when `dᵢ > median(d) + λ · MAD(d)`, where
`dᵢ = ‖ψ⁽ⁱ⁾ − ψ*‖` and `MAD(d) = median(|d − median(d)|)`.

Median-absolute-deviation is used rather than mean/standard-deviation because the
outlier being tested for would itself skew a mean-based statistic. The MAD is
floored at a small constant so that a degenerate spread (all agents identical)
does not divide by zero and does not flag anyone.

**Definitional sensitivity.** For an even number of agents, "median" is ambiguous
— lower-of-two versus average-of-two. The two give different outlier masks. This
must be fixed consistently across every place the screen is computed, or the same
ensemble is judged differently in different parts of the system.

---

### Stage 7 — Escalation Tier: Geometric Median

**Weiszfeld: ψ ← Σᵢ wᵢψ⁽ⁱ⁾ / d(ψ, ψ⁽ⁱ⁾), reproject**

| | |
|---|---|
| **Purpose** | fuse robustly when the mean cannot be trusted |
| **Input** | `{ψ⁽ⁱ⁾}`, weights `wᵢ` |
| **Output** | robust fused amplitude `ψ*` |
| **Parameters** | `max_iter`, convergence `tol`, distance floor `eps`, initialisation |
| **Entered** | only on a Stage 6 flag |

Iteratively reweights each agent by the inverse of its distance to the current
estimate, re-averages, and re-projects onto the sphere — converging to the point
minimising total distance to all agents.

**Breakdown point 1/2.** Up to just under half the agents may be arbitrarily
corrupted and the estimate still converges to the honest answer. This is the
guarantee the whole robustness argument rests on, and **it requires an honest
majority** — hence N ≥ 3.

**Initialisation** determines both iteration count and, in degenerate
configurations, which optimum is reached. It is part of the specification, not an
implementation detail. Convergence is slowest exactly when it matters most: a
clearly separated adversarial agent.

---

### Stage 8 — Sheaf Reconciliation

**local sections (tokenizers) → glued global section (bytes)**

| | |
|---|---|
| **Purpose** | make distributions over *different* vocabularies comparable at all |
| **Input** | per-agent distribution `p⁽ⁱ⁾` over its own vocabulary; all vocabularies |
| **Output** | consensus byte string; per-agent distributions back in token space |
| **Parameters** | `k`, `ε` (mass floor), `min_support`, `max_depth`, decode beam and length, byte-space `θ_H` |
| **Required when** | tokenizers differ. Pure overhead when they do not |

Without a shared base space, "token *v*" does not denote the same string across
agents, and their amplitude vectors are not even the same length. Fusion is not
inaccurate in that setting — it is undefined.

#### 8.1 The base space

Every agent's vocabulary is expressed in **raw bytes**. A union prefix tree is
built over all of them. A node is a byte prefix `s`.

| Sheaf concept | Here |
|---|---|
| base space | the tree's nodes (byte prefixes) |
| stalk at `s` | a distribution over the next byte |
| local section | one agent's conditional at `s`, on the open set where it still has mass |
| restriction map | parent → child edge: conditioning one byte further |
| gluing | agreement on overlaps ⇒ one global section over the whole tree |

**The tree depends only on the vocabularies, never on the prompt.** It is
therefore a build-time artifact, constructed once per ensemble and reused for
every query and every decoding step. It must be rebuilt when the ensemble
changes, and a mismatch between artifact and ensemble is a correctness failure,
not a performance one.

#### 8.2 The two masses

For each agent, at each node `s`:

```
cover(s) = Σ p(v) over tokens whose bytes START WITH s
term(s)  = Σ p(v) over tokens whose bytes EQUAL s

cover(s) = term(s) + Σ_b cover(s·b)          (structural identity)
```

`cover` is the probability the next token begins with `s`. `term` is the
probability it *is* `s` — a statement about that agent's segmentation, not about
content.

#### 8.3 Local sections

The stalk at `s` for agent `i` is its distribution over the next byte,
conditioned on continuing:

```
c⁽ⁱ⁾(s)[b] = cover⁽ⁱ⁾(s·b) / ( cover⁽ⁱ⁾(s) − term⁽ⁱ⁾(s) )
```

Every agent's stalk lives on the **same 256-simplex**, which is what makes them
comparable. Stages 1, 5, 6 and 7 therefore apply unchanged — **per node**.

#### 8.4 Token boundaries are horizons, not content

An agent whose token ends exactly at `s` has no opinion about what follows: one
forward pass simply does not reach further. It **leaves the support** at deeper
nodes and the remaining agents carry the prediction.

It must **not** be fused as an additional "stop" symbol alongside the 256 bytes.
Consider two agents both predicting `" the"`, one holding it as a single token,
the other as `" th"` + `"e"`. At node `" th"` the first continues and the second
stops. Treating that as a vote produces a 50/50 split — a manufactured
disagreement about where a tokenizer draws boundaries, between two agents that
agree completely.

**Content is consensus; segmentation stays local.** Boundaries re-enter only at
projection (§8.7).

#### 8.5 Gluing

Fused conditionals are chained from the root:

```
glued_cover(s·b) = glued_cover(s) · fused_c(s)[b],      glued_cover(root) = 1
```

The chained product is automatically a valid measure over byte strings, so gluing
requires no renormalisation pass. Consistent local sections determine a unique
global section — the sheaf condition, made operational.

#### 8.6 Emission

Bytes are emitted while at least `min_support` agents can still see past the
current prefix. The chunk therefore ends at the **ensemble's joint knowledge
horizon**, not at any single tokenizer's boundary.

`min_support` trades length against precision:

| Value | Behaviour |
|---|---|
| 1 | any agent may extend; longest chunks, lowest precision |
| ⌈N/2⌉ | majority must agree to continue |
| N | unanimous; shortest chunks, highest precision |

Emitted bytes are committed to the context and re-encoded by every agent, so a
wrong byte compounds. This is the parameter to tighten when generation quality
matters more than step count.

#### 8.7 Projection back to token space

For the next generation step, each agent needs the consensus in *its own*
vocabulary:

```
q⁽ⁱ⁾(v) ∝ glued_cover(bytes(v)) · P⁽ⁱ⁾[token ends at bytes(v) | prefix]
```

Content comes from the consensus; the segmentation model is that agent's own
boundary rate. Two diagnostics fall out and should be surfaced:

- **unreachable mass** — glued mass flowing into subtrees where this agent has no
  token boundary at all: what its vocabulary genuinely cannot express;
- **reweighting drift** — how far the projection had to reweight this agent to
  reach the consensus.

#### 8.8 Termination

**Stop tokens have no byte image**, so they cannot exist in the base space at all.
Their mass must be captured *before* the distribution is renormalised onto bytes,
and carried alongside as a separate signal. Generation halts on a majority vote of
stop mass.

If this is not done, the ensemble renormalises the models' intention to stop out
of existence and can never terminate.

Note that a model's end-of-turn token is not always the one its tokenizer
nominates as end-of-sequence; the set of terminating tokens is part of the
ensemble specification.

---

## 5. Parameter reference

| Parameter | Stage | Governs | Notes |
|---|---|---|---|
| `N` | 0 | agent count | ≥ 3 for Stage 7's guarantee to hold |
| `wᵢ` | 0, 5, 7 | relative trust per agent | non-negative, normalised at use |
| `k` | 2, 8 | summary / candidate width | bounds gate cost at O(N·k); in Stage 8 also bounds tree work per step |
| `θ_H` | 2 | entropy threshold | scale depends on the space; does not transfer between token and byte space |
| `θ_D` | 2 | divergence threshold | on the amplitude (Hellinger) scale |
| `λ` | 6 | MAD multiplier | higher ⇒ fewer escalations |
| MAD floor | 6 | degenerate-spread guard | prevents false flags when agents coincide |
| `max_iter`, `tol`, `eps` | 7 | Weiszfeld convergence | slowest to converge when an adversary is present |
| initialisation | 7 | Weiszfeld start point | affects iteration count and degenerate outcomes |
| `ε` | 8 | node mass floor | nodes below it are pruned from fusion |
| `min_support` | 8 | bytes committed per step | length vs precision |
| `max_depth` | 8 | maximum prefix length | truncation preserves cover mass but not boundary mass |
| beam, max length | 8 | decode search | |
| tree artifact | 8 | the shared base space | build-time; must match the ensemble exactly |

**No threshold here is self-calibrating.** All are properties of the ensemble and
the domain, and are intended to be swept against a validation set rather than
fixed by reasoning.

---

## 6. Invariants

Properties that must hold for the architecture to be sound. Most fail silently.

1. `‖ψ⁽ⁱ⁾‖₂ = 1` after Stage 1; `p = ψ²`.
2. The gate decides using data cheaper than what it gates. Violating this makes
   Stage 2 a cost rather than a saving.
3. Both gate conditions are required; either alone is insufficient.
4. Stage 5 is entered only when the gate says contested; Stage 7 only when Stage 6
   flags. Escalation is never unconditional.
5. Stage 7's guarantee requires an honest majority.
6. Every agent's vocabulary maps to raw bytes correctly. A wrong mapping produces
   a well-formed tree that is silently misaligned.
7. Tokens with no byte image — control, chat and stop tokens — never enter the
   base space.
8. `cover(s) = term(s) + Σ_b cover(s·b)` at every node.
9. `cover` is monotone non-increasing with depth, so an active node's ancestors
   are always active.
10. Fused conditionals are normalised, and no byte supported by any agent receives
    zero fused mass.
11. Byte-space gate divergence is measured over all active nodes, not the root.
12. The tree artifact matches the ensemble it is used with.
13. Emitted bytes never split a multi-byte character in the context fed back to
    agents.
14. Tie-breaking in decode is deterministic and independent of how the tree
    enumerates children.

---

## 7. Provenance

Individual techniques are drawn from published work. **The overall framework is a
synthesis, not one paper.**

| Stage | Source |
|---|---|
| 1 | Hellinger / Bhattacharyya square-root transform; Knuth, *Why Square Roots of Probabilities?*, arXiv:1603.08427 |
| 2, 3 | SAFE, *When to Ensemble*, arXiv:2510.15346; arXiv:2606.00405 |
| 4, 5 | DeePEn — Huang et al., NeurIPS 2024, arXiv:2404.12715; MPI-style allgather |
| 6 | Median-absolute-deviation screening; Pillutla et al., Robust Federated Aggregation |
| 7 | DecentLLMs — Byzantine-robust LLM-agent consensus via Weiszfeld, arXiv:2507.14928 |
| 8 | *Exact Byte-Level Probabilities from Tokenized Language Models* — Phan et al., ICLR 2025, arXiv:2410.09303 |

**Scope note on Stage 8.** The cited work also corrects for the *prompt's* cover —
the set of token sequences encoding the same prompt bytes. This architecture
specifies exact cover-mass marginalisation for the next-token step and re-encodes
the full context each step; it does not enumerate prompt covers. That is the gap
to close for full exactness.
