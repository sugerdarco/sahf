"""Stage 8 (Sheaf Reconciliation) benchmark -- Stage 8 ONLY, realistic inputs.

Stages 1-7 are not built here. What Stage 8 actually receives in the full
architecture is reproduced instead:

  * three agents with real, independently trained tokenizers of different
    families and sizes (byte-level BPE 32k / byte-level BPE 27k / Unigram 32k);
  * per-agent next-token distributions over their own vocabularies, peaked on
    the same intended continuation but competing over tokens that share byte
    prefixes, on top of a dense softmax tail;
  * the top-k id summaries Stage 2 already computes -- the "quick token search"
    that Stage 8 rides on rather than re-deriving.

Reported per step: the end-to-end latency plus a phase breakdown
(top-k search / tree build / mass propagation / fusion+gluing / decode /
projection back to token space), and the quality metrics that decide whether a
given k is usable at all.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sahf.sheaf import BytePrefixTree, SheafReconciler, Stage8Pipeline, VocabSpec
from sahf.sheaf.adapters import StaticAgent, top_k_ids

HERE = Path(__file__).resolve().parent
TOK_DIR = HERE / "tokenizers"
CORPUS_PATH = HERE / "corpus.txt"
RESULTS_PATH = HERE / "results" / "bench_results.json"
NAMES = ["agent-A-bpe32k", "agent-B-bpe50k", "agent-C-uni32k"]

# entropy regimes, as observed in real decoding
REGIMES = {
    "confident": dict(p1=0.90, tail=0.02),  # mid-word, forced continuation
    "typical": dict(p1=0.55, tail=0.10),  # ordinary next-word
    "uncertain": dict(p1=0.18, tail=0.35),  # start of a clause, many options
}


class Shim:
    """Minimal HF-tokenizer surface over a raw `tokenizers.Tokenizer`."""

    def __init__(self, tok, name):
        self._tok = tok
        self.name_or_path = name
        self.all_special_ids = []
        self.added_tokens_decoder = {}

    def get_vocab(self):
        return self._tok.get_vocab()

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text).ids


# --------------------------------------------------------------------------
# agent construction
# --------------------------------------------------------------------------


def load_agents():
    specs, shims = [], []
    for n in NAMES:
        tok = Tokenizer.from_file(str(TOK_DIR / f"{n}.json"))
        shim = Shim(tok, n)
        vs = VocabSpec.from_hf(shim, name=n)
        specs.append(vs)
        shims.append(shim)
    return specs, shims


def build_prefix_index(vs: VocabSpec):
    """first 2 bytes -> token ids, for picking realistic competitor tokens."""
    idx: dict[bytes, list[int]] = {}
    for i, b in enumerate(vs.token_bytes):
        if b:
            idx.setdefault(b[:2], []).append(i)
    return idx


def longest_match(vs: VocabSpec, index, target: bytes) -> int | None:
    """The token this agent would most plausibly emit next for `target`."""
    best, blen = None, 0
    for tid in index.get(target[:2], []):
        b = vs.token_bytes[tid]
        if b and target.startswith(b) and len(b) > blen:
            best, blen = tid, len(b)
    if best is None:  # fall back to any single-byte token
        for tid in index.get(target[:1], []):
            b = vs.token_bytes[tid]
            if b and target.startswith(b):
                return tid
    return best


class DistBuilder:
    """Realistic next-token distributions over one agent's vocabulary."""

    def __init__(self, vs: VocabSpec, rng: np.random.Generator, n_competitors: int = 96):
        self.vs = vs
        self.rng = rng
        self.index = build_prefix_index(vs)
        self.n_comp = n_competitors
        self.size = vs.size
        # a fixed Zipf tail over a fixed permutation == a plausible dense softmax
        perm = rng.permutation(self.size)
        z = 1.0 / (1.0 + np.arange(self.size)) ** 1.1
        self.tail = np.zeros(self.size)
        self.tail[perm] = z / z.sum()

    def build(self, target: bytes, p1: float, tail: float) -> tuple[np.ndarray, int]:
        p = np.zeros(self.size)
        true_id = longest_match(self.vs, self.index, target)
        if true_id is None:
            true_id = int(self.rng.integers(self.size))
        head = 1.0 - tail
        p[true_id] = p1 * head

        # competitors: tokens sharing a byte prefix with the true token,
        # which is what actually crowds the top-k in real decoding
        pool = list(self.index.get(target[:2], []))
        pool += list(self.index.get(target[:1], []))
        pool = [t for t in dict.fromkeys(pool) if t != true_id][: self.n_comp]
        if pool:
            w = 1.0 / (1.0 + np.arange(len(pool))) ** 1.2
            w = w / w.sum() * (1.0 - p1) * head
            p[np.array(pool)] += w
        else:
            p += (1.0 - p1) * head / self.size

        p += self.tail * tail
        return p / p.sum(), true_id


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


def continuations(shims, n: int, rng) -> list[tuple[str, bytes]]:
    """(context, intended next bytes) drawn from held-out corpus text."""
    with open(CORPUS_PATH, errors="ignore") as fh:
        blob = fh.read(6_000_000)
    out = []
    while len(out) < n:
        i = int(rng.integers(1000, len(blob) - 400))
        seg = blob[i : i + 300]
        if "\x00" in seg:
            continue
        ctx, nxt = seg[:200], seg[200:240].encode("utf-8", errors="ignore")
        if len(nxt) < 8:
            continue
        out.append((ctx, nxt))
    return out


def phase_timed(specs, probs, k, fusion="auto"):
    """One Stage 8 step with a per-phase breakdown."""
    t = {}
    t0 = time.perf_counter()
    restrict = [top_k_ids(p, k) for p in probs]
    t1 = time.perf_counter()
    tree = BytePrefixTree.from_vocabs(specs, restrict=restrict)
    t2 = time.perf_counter()
    covers = [tree.cover_mass(a, probs[a]) for a in range(len(specs))]
    t3 = time.perf_counter()
    sec = SheafReconciler(fusion=fusion).reconcile(tree, probs)
    t4 = time.perf_counter()
    consensus = sec.decode(min_support=max(1, (len(specs) + 1) // 2))
    t5 = time.perf_counter()
    for a in range(len(specs)):
        sec.project_to_vocab(a)
    t6 = time.perf_counter()

    t["topk_search"] = (t1 - t0) * 1e3
    t["tree_build"] = (t2 - t1) * 1e3
    t["mass_prop"] = (t3 - t2) * 1e3
    t["fuse_glue"] = (t4 - t3) * 1e3
    t["decode"] = (t5 - t4) * 1e3
    t["project"] = (t6 - t5) * 1e3
    t["_total_phases"] = (t6 - t0) * 1e3 - (t3 - t2)  # mass_prop is measured, not extra
    return t, tree, sec, consensus, covers


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def run(specs, cases, builders, *, k, regime, trials, fusion="auto", min_support=None):
    cfg = REGIMES[regime]
    lat, phases, cov, nodes, esc = [], [], [], [], []
    first, good, chunk = [], [], []
    agents = [StaticAgent(v.name, v) for v in specs]
    pipe = Stage8Pipeline(agents, mode="topk_union", k=k, fusion=fusion, min_support=min_support)

    for ctx, nxt in cases[:trials]:
        probs = [b.build(nxt, cfg["p1"], cfg["tail"])[0] for b in builders]
        for ag, p in zip(agents, probs, strict=True):
            ag.set(ctx, p)

        t0 = time.perf_counter()
        res = pipe.step(ctx)
        lat.append((time.perf_counter() - t0) * 1e3)

        ph, tree, _sec, _cons, _ = phase_timed(specs, probs, k, fusion)
        phases.append(ph)
        cov.append(float(np.mean(res.coverage)))
        nodes.append(tree.n_nodes)
        esc.append(res.n_escalated)

        # Quality. "Is the whole chunk a correct prefix" is a BAD metric: it
        # conflates correctness with how many bytes were emitted, so it falls as
        # k rises purely because chunks lengthen. Measure the two separately.
        c = res.consensus_bytes
        chunk.append(len(c))
        first.append(bool(c) and c[:1] == nxt[:1])
        n = 0
        for i, byte in enumerate(c):
            if i < len(nxt) and byte == nxt[i]:
                n += 1
            else:
                break
        good.append(n)

    agg = {p: statistics.median(x[p] for x in phases) for p in phases[0] if not p.startswith("_")}
    return dict(
        k=k,
        regime=regime,
        n_agents=len(specs),
        median_ms=statistics.median(lat),
        p95_ms=pct(lat, 0.95),
        mean_ms=statistics.mean(lat),
        phases=agg,
        coverage=statistics.mean(cov),
        nodes=statistics.median(nodes),
        escalated=statistics.mean(esc),
        first_byte=statistics.mean(first),
        correct_bytes=statistics.mean(good),
        chunk_bytes=statistics.mean(chunk),
        min_support=min_support,
    )


def main():
    rng = np.random.default_rng(0)
    if not TOK_DIR.exists():
        raise SystemExit("run `python benchmarks/train_bench_tokenizers.py` first")
    print("loading tokenizers ...", flush=True)
    specs, shims = load_agents()
    for vs, sh in zip(specs, shims, strict=True):
        vs.verify(sh, ["def main():\n    return 1", "the quick brown fox"], verbose=False)
        print(f"  {vs.name:<16} {vs.size:>6} tokens   scheme={vs.scheme}")

    builders = [DistBuilder(v, rng) for v in specs]
    cases = continuations(shims, 260, rng)
    TRIALS = 200
    results = []

    print(f"\nwarmup + {TRIALS} trials per configuration\n")
    run(specs, cases, builders, k=32, regime="typical", trials=20)  # warmup

    print("=" * 78)
    print("SWEEP 1 -- top-k (Stage 2 summary width), typical entropy, 3 agents")
    print("=" * 78)
    print(
        f"{'k':>5} {'median ms':>10} {'p95 ms':>9} {'nodes':>7} {'coverage':>9} "
        f"{'1st byte':>9} {'correct B':>10} {'chunk B':>8}"
    )
    for k in (8, 16, 32, 64, 128, 256):
        r = run(specs, cases, builders, k=k, regime="typical", trials=TRIALS)
        results.append(r)
        print(
            f"{k:>5} {r['median_ms']:>10.2f} {r['p95_ms']:>9.2f} {r['nodes']:>7.0f} "
            f"{r['coverage']:>9.3f} {r['first_byte']:>9.1%} {r['correct_bytes']:>10.2f} "
            f"{r['chunk_bytes']:>8.2f}"
        )

    print("\n" + "=" * 78)
    print("SWEEP 2 -- entropy regime at k=64, 3 agents")
    print("=" * 78)
    print(
        f"{'regime':>10} {'median ms':>10} {'p95 ms':>9} {'coverage':>9} "
        f"{'escalated':>10} {'1st byte':>9} {'correct B':>10}"
    )
    for reg in REGIMES:
        r = run(specs, cases, builders, k=64, regime=reg, trials=TRIALS)
        results.append(r)
        print(
            f"{reg:>10} {r['median_ms']:>10.2f} {r['p95_ms']:>9.2f} "
            f"{r['coverage']:>9.3f} {r['escalated']:>10.2f} {r['first_byte']:>9.1%} "
            f"{r['correct_bytes']:>10.2f}"
        )

    print("\n" + "=" * 78)
    print("SWEEP 3 -- agent count at k=64, typical entropy")
    print("=" * 78)
    print(f"{'N':>4} {'median ms':>10} {'p95 ms':>9} {'nodes':>7} {'ms/agent':>9}")
    for n in (3, 5, 8):
        sp = [specs[i % 3] for i in range(n)]
        bd = [builders[i % 3] for i in range(n)]
        r = run(sp, cases, bd, k=64, regime="typical", trials=max(60, TRIALS // 2))
        r["n_agents"] = n
        results.append(r)
        print(
            f"{n:>4} {r['median_ms']:>10.2f} {r['p95_ms']:>9.2f} {r['nodes']:>7.0f} "
            f"{r['median_ms'] / n:>9.2f}"
        )

    print("\n" + "=" * 78)
    print("SWEEP 4 -- min_support: how many bytes to commit per step (k=64, typical)")
    print("=" * 78)
    print(
        f"{'min_support':>12} {'median ms':>10} {'chunk B':>9} {'correct B':>10} "
        f"{'wasted B':>9} {'precision':>10}"
    )
    for ms_ in (1, 2, 3):
        r = run(specs, cases, builders, k=64, regime="typical", trials=TRIALS, min_support=ms_)
        results.append(r)
        wasted = r["chunk_bytes"] - r["correct_bytes"]
        prec = r["correct_bytes"] / r["chunk_bytes"] if r["chunk_bytes"] else 0
        print(
            f"{ms_:>12} {r['median_ms']:>10.2f} {r['chunk_bytes']:>9.2f} "
            f"{r['correct_bytes']:>10.2f} {wasted:>9.2f} {prec:>10.1%}"
        )

    print("\n" + "=" * 78)
    print("PHASE BREAKDOWN -- k=64, typical entropy, 3 agents")
    print("=" * 78)
    base = next(
        r
        for r in results
        if r["k"] == 64
        and r["regime"] == "typical"
        and r["n_agents"] == 3
        and r["min_support"] is None
    )
    tot = sum(base["phases"].values())
    for name, ms in sorted(base["phases"].items(), key=lambda x: -x[1]):
        bar = "#" * round(ms / tot * 46)
        print(f"  {name:<13} {ms:>7.3f} ms  {ms / tot:>5.1%}  {bar}")
    print(f"  {'TOTAL':<13} {tot:>7.3f} ms")

    print("\n" + "=" * 78)
    print("REFERENCE -- full-vocabulary mode (no top-k truncation)")
    print("=" * 78)
    t0 = time.perf_counter()
    full = BytePrefixTree.from_vocabs(specs)
    build_s = time.perf_counter() - t0
    rec = SheafReconciler(fusion="auto")
    print(f"  static tree: {full.n_nodes:,} nodes, built once in {build_s:.2f} s")
    for reg in ("confident", "typical"):
        cfg = REGIMES[reg]
        ts = []
        for _ctx, nxt in cases[:12]:
            probs = [b.build(nxt, cfg["p1"], cfg["tail"])[0] for b in builders]
            t0 = time.perf_counter()
            rec.reconcile(full, probs)
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"  {reg:>10}: {statistics.median(ts):>8.1f} ms/step  (coverage 1.000)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
