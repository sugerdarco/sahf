"""Runnable demo -- no model download required.

    python -m sahf.sheaf.demo

Three scenarios:
  1. pure segmentation mismatch  -> must NOT register as disagreement
  2. one byzantine agent         -> Stage 6 flags it, Stage 7 absorbs it
  3. multi-step byte generation  -> consensus text across three vocabularies
"""

from __future__ import annotations

import numpy as np

from .pipeline import Stage8Pipeline
from .prefix_tree import BytePrefixTree
from .reconciler import SheafReconciler
from .adapters import StaticAgent
from .vocab import VocabSpec

RULE = "=" * 74


def vocab(name: str, toks: list[bytes]) -> VocabSpec:
    return VocabSpec.from_mapping({i: b for i, b in enumerate(toks)}, name=name)


def dist(size: int, w: dict[int, float]) -> np.ndarray:
    p = np.zeros(size)
    for i, v in w.items():
        p[i] = v
    return p / p.sum()


# --------------------------------------------------------------------------


def scenario_1_segmentation():
    print(RULE)
    print("1. SEGMENTATION MISMATCH IS NOT DISAGREEMENT")
    print(RULE)
    print("Three agents all predict ' Paris'. Each tokenizer splits it differently.\n")

    a = vocab("coarse", [b" Paris", b" Berlin"])  # one token
    b = vocab("medium", [b" Par", b" Ber", b"is", b"lin"])  # two tokens
    c = vocab("fine", [b" P", b" B", b"a", b"r", b"i", b"s"])  # many tokens

    tree = BytePrefixTree.from_vocabs([a, b, c])
    probs = [
        dist(2, {0: 0.95, 1: 0.05}),
        dist(4, {0: 0.95, 1: 0.05}),
        dist(6, {0: 0.95, 1: 0.05}),
    ]
    sec = SheafReconciler(fusion="auto").reconcile(tree, probs)

    print(f"  union tree: {tree.n_nodes} nodes over {tree.n_agents} vocabularies")
    print("\n  per-node view (support = agents that can still continue):")
    print(f"  {'prefix':<10} {'support':<12} {'horizon':<12} disagreement")
    for node in sec.nodes[:7]:
        r = sec.reports.get(int(node))
        if r is None:
            continue
        sup = ",".join(tree.vocabs[i].name[:4] for i in r.support) or "-"
        hor = ",".join(tree.vocabs[i].name[:4] for i in r.horizon) or "-"
        print(f"  {r.prefix!r:<10} {sup:<12} {hor:<12} {r.disagreement:.4f}")

    worst = max(r.disagreement for r in sec.reports.values())
    print(f"\n  max disagreement anywhere in the tree: {worst:.6f}")
    print("  -> agents drop out as their tokens end; nobody is scored as conflicting.")
    print(f"\n  consensus: {sec.decode(min_support=1)[0][0]!r}")


def scenario_2_byzantine():
    print("\n" + RULE)
    print("2. ONE BYZANTINE AGENT (Stages 6 + 7, applied per node)")
    print(RULE)
    print("Four agents. Three predict ' Paris'; one insists on ' Berlin'.\n")

    v = [vocab(f"agent{i}", [b" Paris", b" Berlin"]) for i in range(4)]
    tree = BytePrefixTree.from_vocabs(v)
    honest = dist(2, {0: 0.95, 1: 0.05})
    byz = dist(2, {0: 0.02, 1: 0.98})
    probs = [honest, honest, honest, byz]

    node = tree.find(b" ")
    for mode in ("mean", "auto"):
        sec = SheafReconciler(fusion=mode).reconcile(tree, probs)
        cond = sec.conditionals[node]
        esc = sum(1 for r in sec.reports.values() if r.escalated)
        label = "Stage 5 only" if mode == "mean" else "Stage 5 -> 7 on flag"
        print(
            f"  {label:<22} P(next byte='P')={cond[ord('P')]:.4f}  "
            f"P('B')={cond[ord('B')]:.4f}  escalated nodes={esc}"
        )
    print("\n  -> the geometric median pulls the consensus back toward the honest majority.")


def scenario_3_generation():
    print("\n" + RULE)
    print("3. MULTI-STEP BYTE-LEVEL GENERATION")
    print(RULE)

    target = b" Paris is the capital"

    def granular_vocab(name: str, max_len: int) -> VocabSpec:
        """A vocabulary that fully covers the target, capped at `max_len` bytes.

        Stands in for three tokenizers with different compression: the coarse
        agent sees several bytes ahead per forward pass, the fine one only one.
        """
        toks = sorted(
            {target[i : i + n] for i in range(len(target)) for n in range(1, max_len + 1)}
        )
        return vocab(name, list(toks))

    specs = [
        granular_vocab("coarse", 6),
        granular_vocab("medium", 3),
        granular_vocab("fine", 1),
    ]
    agents = [StaticAgent(v.name, v) for v in specs]

    def lead(v: VocabSpec, remaining: bytes) -> np.ndarray:
        """Agent predicts its own longest token matching the remaining target."""
        best, blen = None, 0
        for i, tok in enumerate(v.token_bytes):
            if tok and remaining.startswith(tok) and len(tok) > blen:
                best, blen = i, len(tok)
        p = np.full(v.size, 0.02 / max(v.size - 1, 1))
        if best is not None:
            p[best] = 0.98
        return p / p.sum()

    pipe = Stage8Pipeline(agents, mode="topk_union", k=32)
    sizes = ", ".join(f"{v.name}={v.size} tokens (<={m}B)" for v, m in zip(specs, (6, 3, 1), strict=True))
    print(f"  target: {target!r}")
    print(f"  vocabs: {sizes}")
    print(f"  min_support = {pipe.min_support} of 3 (majority)\n")
    print(f"  {'emitted':<10} {'support at stop':<18} running text")

    produced = b""
    last = None
    for _ in range(8):
        remaining = target[len(produced) :]
        if not remaining:
            break
        ctx = produced.decode("utf-8", errors="replace")
        for ag, v in zip(agents, specs, strict=True):
            ag.set(ctx, lead(v, remaining))
        res = pipe.step(ctx)
        emitted = res.consensus_bytes
        if not emitted:
            break
        produced += emitted
        node = res.section.tree.find(emitted)
        rep = res.section.reports.get(node)
        stopped = ",".join(specs[i].name for i in rep.support) if rep else "-"
        print(f"  {emitted!r:<10} {stopped:<18} {produced.decode()!r}")
        last = res

    print(f"\n  reconstructed: {produced.decode()!r}   exact match: {produced == target}")
    print("  -> each chunk ends where the ensemble's majority runs out of lookahead,")
    print("     not where any one tokenizer draws a boundary.")

    if last is not None:
        print(f"\n  final step diagnostics:\n  {last.summary()}")
        print("\n  same step, back in each agent's own token space:")
        for v in specs:
            q = last.token_probs[v.name]
            top = np.argsort(-q)[:3]
            pretty = ", ".join(f"{v.token_bytes[i]!r}:{q[i]:.3f}" for i in top if q[i] > 1e-6)
            print(f"    {v.name:<8} {pretty}")


def main():
    scenario_1_segmentation()
    scenario_2_byzantine()
    scenario_3_generation()
    print("\n" + RULE)


if __name__ == "__main__":
    main()
