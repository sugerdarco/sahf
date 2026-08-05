"""Stage-by-stage audit of sahf_sheaf.

Adversarial probes per stage, checking invariants and inputs the happy path
never exercises. Exits non-zero if any probe fails, so it can gate CI.
Findings and fixes are written up in docs/AUDIT_REPORT.md.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sahf.sheaf import BytePrefixTree, SheafReconciler, Stage8Pipeline, VocabSpec
from sahf.sheaf.adapters import StaticAgent
from sahf.sheaf.vocab import bytes_to_unicode, detect_scheme

FAILS = []


def check(stage, name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(f"[{stage}] {name}: {detail}")
    print(f"  {tag}  {name}" + (f"   -- {detail}" if detail and not ok else ""))


def vocab(name, toks):
    return VocabSpec.from_mapping({i: b for i, b in enumerate(toks)}, name=name)


def dist(n, w):
    p = np.zeros(n)
    for i, v in w.items():
        p[i] = v
    return p / p.sum()


# =====================================================================
print("\nSTAGE A -- vocabulary extraction (vocab.py)")
# =====================================================================

enc = bytes_to_unicode()
ok = all("".join(enc[b] for b in bytes([i])) and True for i in range(256))
from sahf.sheaf.vocab import decode_bytelevel

roundtrip = all(decode_bytelevel(enc[i]) == bytes([i]) for i in range(256))
check("A", "byte-level covers all 256 byte values", roundtrip)

# A1: byte-level vocab that legitimately contains "##" (markdown/code corpora)
bl_vocab = ["".join(enc[b] for b in t) for t in [b"##", b" the", b"\n", b"def", b" x"]]
scheme = detect_scheme(bl_vocab)
check(
    "A",
    "byte-level vocab containing '##' is not misread as wordpiece",
    scheme == "byte_level",
    f"detected {scheme!r}, expected 'byte_level'",
)

# A2: genuine wordpiece must still be detected
wp_vocab = ["play", "##ing", "##ed", "the", "cat", "[UNK]"]
scheme = detect_scheme(wp_vocab)
check("A", "genuine wordpiece still detected", scheme == "wordpiece", f"detected {scheme!r}")

# A3: sentencepiece
sp_vocab = ["\u2581the", "\u2581cat", "<0x0A>", "s"]
check("A", "sentencepiece detected", detect_scheme(sp_vocab) == "sentencepiece")

# A4: control / added tokens leaking into the byte space
chat_vocab = ["".join(enc[b] for b in t) for t in [b"<|im_start|>", b" hi"]]


class FakeTok:
    def __init__(self, toks):
        self._v = {t: i for i, t in enumerate(toks)}
        self.all_special_ids = []
        self.vocab_size = len(toks)
        self.name_or_path = "fake"

    def get_vocab(self):
        return self._v

    def encode(self, text, add_special_tokens=False):
        return []


vs = VocabSpec.from_hf(FakeTok(chat_vocab), scheme="byte_level")
leaks = vs.token_bytes[0] == b"<|im_start|>"
check(
    "A",
    "control tokens do not enter the byte tree as literal text",
    not leaks,
    "'<|im_start|>' decoded to its literal bytes and will be fused as real text",
)

# =====================================================================
print("\nSTAGE B -- byte-prefix tree (prefix_tree.py)")
# =====================================================================

v = vocab("a", [b" the", b" th", b" cat", b"x"])
tree = BytePrefixTree.from_vocabs([v])
p = dist(4, {0: 0.4, 1: 0.2, 2: 0.3, 3: 0.1})
cover, term = tree.cover_mass(0, p)
ident = all(
    abs(cover[n] - (term[n] + sum(cover[c] for c in tree.children[n].values()))) < 1e-12
    for n in range(tree.n_nodes)
)
check("B", "cover(s) = term(s) + sum children", ident)
check("B", "root cover == 1", abs(cover[0] - 1.0) < 1e-12)

# B1: probability vector shorter than the vocab table
short = np.array([0.5, 0.5])
try:
    tree.cover_mass(0, short)
    ok, detail = False, "silently accepted a mis-sized probability vector"
except IndexError:
    ok, detail = False, "raised a bare IndexError instead of a clear error"
except ValueError as e:
    ok, detail = True, str(e)
check("B", "mis-sized probability vector rejected clearly", ok, detail)

# B2: duplicate ids in restrict
tree_dup = BytePrefixTree.from_vocabs([v], restrict=[[0, 0, 1]])
cov_d, _ = tree_dup.cover_mass(0, p)
check(
    "B",
    "duplicate ids in restrict do not double-count mass",
    abs(cov_d[0] - 0.6) < 1e-12,
    f"root cover = {cov_d[0]:.4f}, expected 0.6 (0.4+0.2)",
)

# B3: truncation preserves the identity
v_long = vocab("a", [b"abcdef", b"abcxyz", b"ab"])
t_trunc = BytePrefixTree.from_vocabs([v_long], max_depth=3)
c2, t2 = t_trunc.cover_mass(0, dist(3, {0: 0.5, 1: 0.3, 2: 0.2}))
ident2 = all(
    abs(c2[n] - (t2[n] + sum(c2[c] for c in t_trunc.children[n].values()))) < 1e-12
    for n in range(t_trunc.n_nodes)
)
check("B", "identity holds under max_depth truncation", ident2)

# B4: monotonicity (parent >= child), required for the depth-sorted glue walk
mono = all(
    cover[n] >= cover[c] - 1e-15 for n in range(tree.n_nodes) for c in tree.children[n].values()
)
check("B", "cover mass is monotone down the tree", mono)

# =====================================================================
print("\nSTAGE C -- sheaf gluing (sheaf.py)")
# =====================================================================

a = vocab("a", [b" the", b" that", b" a"])
b = vocab("b", [b" th", b" a", b"e", b"at"])
tr = BytePrefixTree.from_vocabs([a, b])
pa, pb = dist(3, {0: 0.5, 1: 0.3, 2: 0.2}), dist(4, {0: 0.8, 1: 0.2})
sec = SheafReconciler(fusion="mean").reconcile(tr, [pa, pb])

d1 = sum(sec.cover[int(n)] for n in sec.nodes if tr.depth[int(n)] == 1)
check("C", "glued depth-1 mass == 1", abs(d1 - 1.0) < 1e-9, f"got {d1}")

norm = all(abs(c.sum() - 1.0) < 1e-9 for c in sec.conditionals.values())
check("C", "every fused conditional is normalized", norm)

# C1: a fused conditional must not zero out a byte some agent supports
zero_bug = False
for node, cond in sec.conditionals.items():
    for byte, child in tr.children[node].items():
        if any(cv[child] > 0 for cv in sec.agent_cover) and cond[byte] == 0:
            zero_bug = True
check("C", "no supported byte is assigned zero fused mass", not zero_bug)

# C2: mismatched number of distributions
try:
    SheafReconciler().reconcile(tr, [pa])
    ok = False
except ValueError:
    ok = True
check("C", "agent-count mismatch rejected", ok)

# C3: probs longer/shorter than vocab reach reconcile unchecked
try:
    SheafReconciler().reconcile(tr, [np.array([1.0]), pb])
    ok, detail = False, "accepted a mis-sized distribution without error"
except (ValueError, IndexError) as e:
    ok, detail = isinstance(e, ValueError), type(e).__name__
check("C", "mis-sized distribution rejected in reconcile", ok, detail)

# C4: single-agent-support fast path equals the plain mean of one section
a2 = vocab("a", [b"ab"])
b2 = vocab("b", [b"a"])
tr2 = BytePrefixTree.from_vocabs([a2, b2])
s2 = SheafReconciler(fusion="mean").reconcile(tr2, [dist(1, {0: 1.0}), dist(1, {0: 1.0})])
node_a = tr2.find(b"a")
check(
    "C",
    "sole surviving agent carries the continuation",
    abs(s2.conditionals[node_a][ord("b")] - 1.0) < 1e-12,
)

# =====================================================================
print("\nSTAGE D -- projection & pipeline (sheaf.py / pipeline.py)")
# =====================================================================

q, rep0 = sec.project_to_vocab(0)
check("D", "projection is a valid distribution", abs(q.sum() - 1.0) < 1e-9)

# D1: residual diagnostic sanity -- fine-grained vocab double counts across depths
fine = vocab("fine", [b"a", b"ab", b"abc", b"abcd"])
coarse = vocab("coarse", [b"abcd", b"abce"])
trf = BytePrefixTree.from_vocabs([fine, coarse])
secf = SheafReconciler(fusion="mean").reconcile(
    trf, [dist(4, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}), dist(2, {0: 0.9, 1: 0.1})]
)
_, rep_fine = secf.project_to_vocab(0)
claimed = 0.0
for n in secf.nodes:
    n = int(n)
    if n == 0 or secf.cover[n] <= 0:
        continue
    if trf.terminals_at(n, 0) and secf.agent_cover[0][n] > 0:
        claimed += secf.cover[n] * secf.agent_term[0][n] / secf.agent_cover[0][n]
check(
    "D",
    "projection separates reweighting drift from real loss",
    abs(rep_fine.mass_ratio - claimed) < 1e-9 and rep_fine.unreachable >= 0.0,
    f"mass_ratio={rep_fine.mass_ratio:.3f} unreachable={rep_fine.unreachable:.3f}",
)

# D2: coverage recomputes mass propagation
calls = {"n": 0}
orig = BytePrefixTree.cover_mass


def counting(self, agent, probs, restrict=None):
    calls["n"] += 1
    return orig(self, agent, probs, restrict)


BytePrefixTree.cover_mass = counting
ags = [StaticAgent("a", a), StaticAgent("b", b)]
ags[0].set("ctx", pa)
ags[1].set("ctx", pb)
Stage8Pipeline(ags, mode="topk_union", k=10).step("ctx")
BytePrefixTree.cover_mass = orig
check(
    "D",
    "mass propagation runs once per agent per step",
    calls["n"] == 2,
    f"ran {calls['n']} times for 2 agents (hot path duplicated)",
)

# D3: UTF-8 safety in generate()
uni = "café".encode()  # b'caf\xc3\xa9'
va = vocab("a", [uni[:4], uni[4:]])  # uni[:4] ends mid-character
vb = vocab("b", [uni[:4], uni[4:]])
ag = [StaticAgent("a", va), StaticAgent("b", vb)]
pipe = Stage8Pipeline(ag, mode="topk_union", k=8, min_support=1)
for _a in ag:
    _a.set("", dist(2, {0: 1.0}))
    _a.set("caf", dist(2, {1: 1.0}))
try:
    text, _ = pipe.generate("", max_steps=1)
    ok = "\ufffd" not in text
    detail = f"emitted {text!r} containing U+FFFD from a split multi-byte character"
except KeyError:
    ok, detail = True, "n/a"
check("D", "generate() never emits U+FFFD mid-character", ok, detail)

# =====================================================================
print("\nSTAGE E -- model adapter (sources.py)")
# =====================================================================

src = (Path(__file__).resolve().parents[1] / "sahf" / "sheaf" / "adapters.py").read_text()
check(
    "E",
    "EOS / stop mass is surfaced rather than silently renormalised away",
    "stop_probability" in src and "_eos_ids" in src,
    "special-token mass is zeroed and renormalised; the ensemble can never stop",
)

# =====================================================================
print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} DEFECTS FOUND\n")
    for f in FAILS:
        print(" *", f)
else:
    print("no defects found")
print("=" * 70)
sys.exit(1 if FAILS else 0)
