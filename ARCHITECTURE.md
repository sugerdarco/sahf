# Architecture

This document describes what `sahf-lite` actually builds, how data flows through
it per decoding step, and — explicitly — what's been simplified away relative to
the full 8-stage design, and why each simplification is safe *for this config*
but wouldn't be in general.

## Data flow, one decoding step

```
prompt_ids (shared, since both agents use one tokenizer)
     │
     ▼
 ┌─────────────────────────────────────────────┐
 │  Agent 1.next_logits(input_ids)              │   two independent forward
 │  Agent 2.next_logits(input_ids)               │   passes, same prefix
 └─────────────────────────────────────────────┘
     │
     ▼
 Stage 1 — softmax_to_amplitude(logits) for each agent
     ψ₁, ψ₂  (points on the unit hypersphere)
     │
     ▼
 Stage 2 — divergence_gate(ψ₁, ψ₂)
     H  = entropy of the mean distribution
     D  = ‖ψ₁ − ψ₂‖₂   (∝ Hellinger distance)
     │
     ├── H < θ_H  AND  D < θ_D  ──────────────► Path A: argmax of the mean
     │                                            probs, done. No fusion object
     │                                            is even constructed.
     │
     └── otherwise ───────────────────────────► Path B:
              │
              ▼
        Stage 5 — fast_mean_fusion(ψ₁, ψ₂)  →  ψ*
              │
              ▼
        Stage 6 — detect_outliers(ψ₁, ψ₂, ψ*)
              │
              ├── no outlier ──────────────────► use ψ*, argmax, done
              │
              └── outlier flagged ─────────────► Stage 7:
                       weiszfeld_geometric_median(ψ₁, ψ₂)  →  ψ*
                       argmax(ψ*²), done
     │
     ▼
 chosen token_id, appended to input_ids, fed back to BOTH agents next step
 (safe: one tokenizer, so "token 4711" means the same thing to both)
```

Every step's `H`, `D`, path taken, outlier flags, and (if triggered) Weiszfeld
iteration count are written to `steps.jsonl` — see README.md for the log format.

## Module map

| File | Stage(s) | Responsibility |
|---|---|---|
| `sahf/amplitude.py` | 1 | logits → ψ = √softmax(z) |
| `sahf/gate.py` | 2 | entropy + pairwise spread → agree/disagree |
| `sahf/fusion.py` | 5 | weighted chordal mean on the sphere |
| `sahf/robust.py` | 6, 7 | MAD outlier screen; Weiszfeld geometric median |
| `sahf/agents.py` | — | `HFAgent` (real transformers model) / `MockAgent` (no GPU/internet) |
| `sahf/orchestrator.py` | — | wires 1→2→(3│5→6→7) into the per-token loop |
| `sahf/logger.py` | — | everything → `out/` |

## Explicit simplifications vs. the full 8-stage design

**Stage 8 (Sheaf Reconciliation) is dropped entirely.** The original stage
exists to reconcile *different* tokenizers via a shared byte-prefix tree. Both
default agents (Qwen2.5 family) use the same tokenizer, so "token v" already
means the same thing to both — there is nothing to reconcile. This is the
single biggest scope cut that makes a same-machine, build-it-today prototype
possible; see the top-level conversation history for why it was deferred first.
**This only holds as long as both configured models share a tokenizer** —
`run.py` checks `len(tokenizer)` for both agents at startup and refuses to run
if they differ, specifically so this assumption fails loudly instead of
silently producing garbage.

**Stage 4 (Allgather) collapses to a Python list comprehension.** The original
stage is a distributed-systems collective, needed when agents live on separate
machines. Here, both agents run in one process on one machine, so "make every
agent's ψ visible to the fusion step" is just `[agent.next_logits(...) for
agent in agents]` — there is no network hop to model. If agents are ever split
across machines, this is the piece to replace (with `torch.distributed`, gRPC,
or similar), and the latency properties of the whole gate design would need
re-measuring, since the entire point of Stage 2 is to avoid paying a
*communication* cost — which barely exists in-process.

**No KV-cache.** `HFAgent.next_logits` re-runs the full forward pass over the
whole sequence so far, every step, for every agent. This trades decode speed
for a much simpler and more obviously-correct v1 (no per-agent cache-alignment
bugs to chase down when a token from Path B doesn't match what an agent would
have generated itself). Direct consequence of the project's own
cascading-cost philosophy: get the cheap-vs-expensive *decision logic* right
first, optimize the *constant factors* second. Tracked as the top follow-up in
`HISTORY.md`.

**N=2 weakens the Stage 6/7 robustness story.** The geometric median's real
selling point — breakdown point 1/2, i.e. it survives up to just under half the
agents being adversarial — requires an honest *majority*. At N=2, if the two
agents disagree, "outlier detection" can only ever say "one of you is further
from the average than the other," which is not the same claim as "the honest
majority overrules the corrupted minority." The code runs correctly at N=2
(and is tested at N=2), but the interesting Byzantine-robustness test
(`test_three_agents_one_poisoned_escalates_and_recovers_majority_token` in
`tests/test_orchestrator_mock.py`) intentionally uses N=3 mock agents, because
that's the smallest N where the claim is actually true. Adding a real or mock
3rd agent to `config.yaml`-driven runs is a small change (the orchestrator
already accepts any N ≥ 2) — the code doesn't assume exactly two agents
anywhere except the tokenizer-match check in `run.py`.

## What would need to change for the full 8-stage version

1. **Heterogeneous tokenizers** — build the byte-prefix trie, token→byte and
   byte→token projections (Stage 8). This is the piece with no shortcut; budget
   the most engineering time here.
2. **Real distributed allgather** — replace the list comprehension with an
   actual collective if agents move to separate machines.
3. **KV-cache per agent**, plus a policy for what an agent does when the
   consensus token doesn't align with what it would have generated (re-tokenize
   and catch up over multiple of its own steps — only actually necessary once
   tokenizers differ; with a shared tokenizer this is a non-issue since the
   consensus token *is* one of that agent's own tokens by construction).
4. **N ≥ 3 agents** to make Stage 7's robustness guarantee meaningful in a real
   (not mock) run.
5. **Threshold calibration** — `θ_H`, `θ_D`, and the MAD multiplier are current
   best guesses (see HISTORY.md); a real version would sweep these against a
   validation set the way the SAFE paper did, rather than hand-picking them.
