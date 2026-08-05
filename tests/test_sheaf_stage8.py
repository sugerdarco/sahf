"""Invariant and regression tests for Stage 8. Run: pytest -q"""

from __future__ import annotations

import numpy as np

from sahf.sheaf.prefix_tree import BytePrefixTree
from sahf.sheaf.reconciler import (
    SheafReconciler,
    geometric_median,
    mean_fuse,
    to_amplitude,
    to_probability,
)
from sahf.sheaf.adapters import StaticAgent
from sahf.sheaf.vocab import VocabSpec, bytes_to_unicode, decode_bytelevel


def make_vocab(name: str, toks: list[bytes]) -> VocabSpec:
    return VocabSpec.from_mapping({i: b for i, b in enumerate(toks)}, name=name)


def dist(size: int, weights: dict[int, float]) -> np.ndarray:
    p = np.zeros(size)
    for i, w in weights.items():
        p[i] = w
    return p / p.sum()


# --------------------------------------------------------------------------
# vocab extraction
# --------------------------------------------------------------------------


def test_bytelevel_roundtrip():
    enc = bytes_to_unicode()
    for raw in [b" the", b"\n\t", bytes(range(256)), "café".encode()]:
        display = "".join(enc[b] for b in raw)
        assert decode_bytelevel(display) == raw


def test_scheme_detection():
    from sahf.sheaf.vocab import detect_scheme

    assert detect_scheme(["\u2581the", "\u2581cat", "<0x0A>"]) == "sentencepiece"
    assert detect_scheme(["##ing", "play", "##ed"]) == "wordpiece"
    assert detect_scheme(["Ġthe", "Ġcat", "hello"]) == "byte_level"


# --------------------------------------------------------------------------
# trie invariants
# --------------------------------------------------------------------------


def test_cover_equals_term_plus_children():
    v = make_vocab("a", [b" the", b" th", b" cat", b" c", b"x"])
    tree = BytePrefixTree.from_vocabs([v])
    p = dist(5, {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.1, 4: 0.1})
    cover, term = tree.cover_mass(0, p)

    for node in range(tree.n_nodes):
        child_sum = sum(cover[c] for c in tree.children[node].values())
        assert abs(cover[node] - (term[node] + child_sum)) < 1e-12, tree.node_bytes(node)

    assert abs(cover[0] - 1.0) < 1e-12
    assert abs(cover[tree.find(b" th")] - 0.6) < 1e-12  # " the" + " th"


def test_node_bytes_roundtrip():
    v = make_vocab("a", [b"hello", b"help", b"he", b"\xf0\x9f\x98\x80"])
    tree = BytePrefixTree.from_vocabs([v])
    for raw in [b"hello", b"help", b"he", b"\xf0\x9f\x98\x80", b"h"]:
        node = tree.find(raw)
        assert node >= 0
        assert tree.node_bytes(node) == raw


def test_union_tree_shares_nodes():
    a = make_vocab("a", [b" the"])
    b = make_vocab("b", [b" th", b"e"])
    tree = BytePrefixTree.from_vocabs([a, b])
    assert tree.find(b" th") >= 0
    assert tree.terminals_at(tree.find(b" the"), 0) == [0]
    assert tree.terminals_at(tree.find(b" th"), 1) == [0]
    assert tree.terminals_at(tree.find(b" the"), 1) == []  # b has no such token


# --------------------------------------------------------------------------
# the boundary-vs-content distinction (the core design decision)
# --------------------------------------------------------------------------


def test_boundary_mismatch_is_not_disagreement():
    """A and B predict the same text with different segmentations => no conflict."""
    a = make_vocab("a", [b" the", b" a"])
    b = make_vocab("b", [b" th", b" a"])
    tree = BytePrefixTree.from_vocabs([a, b])
    pa = dist(2, {0: 1.0})
    pb = dist(2, {0: 1.0})

    sec = SheafReconciler(fusion="mean").reconcile(tree, [pa, pb])

    # every node up to " th" is unanimous
    for prefix in [b"", b" ", b" t"]:
        node = tree.find(prefix)
        assert sec.reports[node].disagreement < 1e-12, prefix

    # at " th" agent b has reached its horizon; a carries the continuation
    node = tree.find(b" th")
    assert sec.reports[node].support == [0]
    assert sec.reports[node].horizon == [1]
    assert abs(sec.conditionals[node][ord("e")] - 1.0) < 1e-12

    # and the glued measure puts full mass on the real string
    assert abs(sec.cover[tree.find(b" the")] - 1.0) < 1e-9


def test_real_content_disagreement_is_reported():
    a = make_vocab("a", [b" cat", b" dog"])
    b = make_vocab("b", [b" cat", b" dog"])
    tree = BytePrefixTree.from_vocabs([a, b])
    sec = SheafReconciler(fusion="mean").reconcile(
        tree, [dist(2, {0: 1.0}), dist(2, {1: 1.0})]
    )
    node = tree.find(b" ")
    assert sec.reports[node].disagreement > 0.9  # orthogonal amplitudes


# --------------------------------------------------------------------------
# gluing
# --------------------------------------------------------------------------


def test_glued_section_is_a_measure():
    """Glued cover mass at each depth sums to 1 minus what was pruned."""
    a = make_vocab("a", [b" the", b" that", b" this", b" a"])
    b = make_vocab("b", [b" th", b" thi", b" a", b"e"])
    c = make_vocab("c", [b" t", b" a", b"h"])
    tree = BytePrefixTree.from_vocabs([a, b, c])
    probs = [
        dist(4, {0: 0.5, 1: 0.3, 2: 0.1, 3: 0.1}),
        dist(4, {0: 0.6, 1: 0.3, 2: 0.1, 3: 0.0}),
        dist(3, {0: 0.8, 1: 0.2, 2: 0.0}),
    ]
    sec = SheafReconciler(fusion="mean").reconcile(tree, probs)

    depth1 = sum(sec.cover[int(n)] for n in sec.nodes if tree.depth[int(n)] == 1)
    assert abs(depth1 - 1.0) < 1e-9
    for cond in sec.conditionals.values():
        assert abs(cond.sum() - 1.0) < 1e-9


def test_conditionals_are_normalized_per_node():
    v1 = make_vocab("a", [b"ab", b"ac", b"b"])
    v2 = make_vocab("b", [b"a", b"ab", b"b"])
    tree = BytePrefixTree.from_vocabs([v1, v2])
    sec = SheafReconciler(fusion="mean").reconcile(
        tree, [dist(3, {0: 0.5, 1: 0.3, 2: 0.2}), dist(3, {0: 0.4, 1: 0.4, 2: 0.2})]
    )
    for node, cond in sec.conditionals.items():
        assert abs(cond.sum() - 1.0) < 1e-9, node
        assert (cond >= 0).all()


def test_decode_recovers_agreed_string():
    a = make_vocab("a", [b" Paris", b" London"])
    b = make_vocab("b", [b" Par", b" Lon", b"is"])
    c = make_vocab("c", [b" Pa", b" Lo"])
    tree = BytePrefixTree.from_vocabs([a, b, c])
    probs = [dist(2, {0: 0.9, 1: 0.1}), dist(3, {0: 0.9, 1: 0.1}), dist(2, {0: 0.9, 1: 0.1})]
    sec = SheafReconciler(fusion="mean").reconcile(tree, probs)
    best = sec.decode(min_support=1)[0][0]
    assert best == b" Paris"
    assert sec.greedy_bytes(min_support=1) == b" Paris"


# --------------------------------------------------------------------------
# projection back to token space
# --------------------------------------------------------------------------


def test_projection_returns_valid_distribution():
    a = make_vocab("a", [b" the", b" cat"])
    b = make_vocab("b", [b" th", b" ca", b"e", b"t"])
    tree = BytePrefixTree.from_vocabs([a, b])
    probs = [dist(2, {0: 0.7, 1: 0.3}), dist(4, {0: 0.7, 1: 0.3})]
    sec = SheafReconciler(fusion="mean").reconcile(tree, probs)

    for agent in (0, 1):
        q, rep = sec.project_to_vocab(agent)
        assert abs(q.sum() - 1.0) < 1e-9
        assert (q >= 0).all()
        assert rep.unreachable >= 0.0
        assert rep.mass_ratio > 0.0

    q0, _ = sec.project_to_vocab(0)
    assert q0[0] > q0[1]  # " the" was the consensus favourite


# --------------------------------------------------------------------------
# robustness (Stages 6/7 acting per node)
# --------------------------------------------------------------------------


def test_geometric_median_resists_one_adversary():
    honest = to_amplitude(dist(4, {0: 0.9, 1: 0.1}))
    adversary = to_amplitude(dist(4, {3: 1.0}))
    psis = np.stack([honest, honest, honest, adversary])
    w = np.full(4, 0.25)

    mean_p = to_probability(mean_fuse(psis, w))
    gm_p = to_probability(geometric_median(psis, w))

    assert gm_p[3] < mean_p[3]
    assert gm_p[0] > mean_p[0]
    assert np.argmax(gm_p) == 0


def test_auto_escalation_triggers_on_outlier():
    a = make_vocab("a", [b"ab", b"zz"])
    b = make_vocab("b", [b"ab", b"zz"])
    c = make_vocab("c", [b"ab", b"zz"])
    d = make_vocab("d", [b"ab", b"zz"])
    tree = BytePrefixTree.from_vocabs([a, b, c, d])
    good = dist(2, {0: 0.99, 1: 0.01})
    bad = dist(2, {0: 0.01, 1: 0.99})
    sec = SheafReconciler(fusion="auto").reconcile(tree, [good, good, good, bad])
    assert any(r.escalated for r in sec.reports.values())

    root = sec.conditionals[0]
    mean_sec = SheafReconciler(fusion="mean").reconcile(tree, [good, good, good, bad])
    assert root[ord("a")] > mean_sec.conditionals[0][ord("a")]


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def test_pipeline_topk_union_end_to_end():
    from sahf.sheaf.pipeline import Stage8Pipeline

    a = make_vocab("a", [b" Paris", b" Berlin"])
    b = make_vocab("b", [b" Par", b" Ber", b"is"])
    c = make_vocab("c", [b" Pa", b" Be"])
    agents = [StaticAgent("a", a), StaticAgent("b", b), StaticAgent("c", c)]
    ctx = "The capital of France is"
    agents[0].set(ctx, dist(2, {0: 0.95, 1: 0.05}))
    agents[1].set(ctx, dist(3, {0: 0.9, 1: 0.1}))
    agents[2].set(ctx, dist(2, {0: 0.92, 1: 0.08}))

    pipe = Stage8Pipeline(agents, mode="topk_union", k=8)
    res = pipe.step(ctx)

    assert res.consensus_bytes.startswith(b" Pa")
    assert res.n_fused_nodes > 0
    assert all(abs(v.sum() - 1.0) < 1e-9 for v in res.token_probs.values())
    assert np.argmax(res.token_probs["a"]) == 0


def test_full_mode_matches_topk_when_k_covers_vocab():
    from sahf.sheaf.pipeline import Stage8Pipeline

    a = make_vocab("a", [b" the", b" a"])
    b = make_vocab("b", [b" th", b" a", b"e"])
    agents = [StaticAgent("a", a), StaticAgent("b", b)]
    ctx = "x"
    agents[0].set(ctx, dist(2, {0: 0.8, 1: 0.2}))
    agents[1].set(ctx, dist(3, {0: 0.8, 1: 0.2, 2: 0.0}))

    r1 = Stage8Pipeline(agents, mode="topk_union", k=100).step(ctx)
    r2 = Stage8Pipeline(agents, mode="full").step(ctx)
    assert r1.consensus_bytes == r2.consensus_bytes


def test_rejects_single_agent():
    from sahf.sheaf.pipeline import Stage8Pipeline

    try:
        Stage8Pipeline([StaticAgent("a", make_vocab("a", [b"x"]))])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a single agent")


# --------------------------------------------------------------------------
# regression tests -- one per defect found in the Stage 8 audit
# --------------------------------------------------------------------------


def test_bytelevel_vocab_with_hash_tokens_not_read_as_wordpiece():
    """AUDIT-1: '##' is an ordinary byte-level token in markdown/code corpora."""
    from sahf.sheaf.vocab import bytes_to_unicode, detect_scheme

    enc = bytes_to_unicode()
    vocab = ["".join(enc[b] for b in t) for t in [b"##", b" the", b"\n", b"def", b" x"]]
    assert detect_scheme(vocab) == "byte_level"
    # and genuine wordpiece must still be recognised
    assert detect_scheme(["play", "##ing", "##ed", "the", "cat"]) == "wordpiece"


def test_control_tokens_excluded_from_byte_space():
    """AUDIT-2: '<|im_start|>' must not be fused as if the model predicted that text."""
    from sahf.sheaf.vocab import bytes_to_unicode

    enc = bytes_to_unicode()
    toks = ["".join(enc[b] for b in t) for t in [b"<|im_start|>", b" hi"]]

    class Tok:
        def __init__(self):
            self._v = {t: i for i, t in enumerate(toks)}
            self.all_special_ids = []
            self.vocab_size = len(toks)
            self.name_or_path = "fake"

        def get_vocab(self):
            return self._v

    vs = VocabSpec.from_hf(Tok(), scheme="byte_level")
    assert vs.token_bytes[0] is None
    assert vs.token_bytes[1] == b" hi"


def test_mis_sized_distribution_rejected():
    """AUDIT-3: a wrong-length vector mis-indexes tokens instead of failing."""
    v = make_vocab("a", [b"ab", b"cd", b"ef"])
    tree = BytePrefixTree.from_vocabs([v])
    try:
        tree.cover_mass(0, np.array([0.5, 0.5]))
    except ValueError as e:
        assert "length 3" in str(e)
        return
    raise AssertionError("expected ValueError for a mis-sized distribution")


def test_duplicate_restrict_ids_do_not_double_count():
    """AUDIT-4: top-k lists merged from several sources contain repeats."""
    v = make_vocab("a", [b" the", b" th", b" cat", b"x"])
    p = dist(4, {0: 0.4, 1: 0.2, 2: 0.3, 3: 0.1})
    tree = BytePrefixTree.from_vocabs([v], restrict=[[0, 0, 0, 1]])
    cover, _ = tree.cover_mass(0, p)
    assert abs(cover[0] - 0.6) < 1e-12


def test_projection_report_separates_drift_from_loss():
    """AUDIT-5: mass_ratio may exceed 1; unreachable is the real loss figure."""
    a = make_vocab("fine", [b"a", b"ab", b"abc", b"abcd"])
    b = make_vocab("coarse", [b"abcd", b"abce"])
    tree = BytePrefixTree.from_vocabs([a, b])
    sec = SheafReconciler(fusion="mean").reconcile(
        tree, [dist(4, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}), dist(2, {0: 0.9, 1: 0.1})]
    )
    q, rep = sec.project_to_vocab(0)
    assert abs(q.sum() - 1.0) < 1e-9
    assert rep.mass_ratio > 1.0  # boundaries at four depths reweight above 1
    assert rep.unreachable >= 0.0

    # a vocabulary that can never end a token in a live subtree reports the loss
    x = make_vocab("x", [b"zz"])
    y = make_vocab("y", [b"zz", b"qq"])
    t2 = BytePrefixTree.from_vocabs([x, y])
    s2 = SheafReconciler(fusion="mean").reconcile(
        t2, [dist(1, {0: 1.0}), dist(2, {0: 0.5, 1: 0.5})]
    )
    _, rep_x = s2.project_to_vocab(0)
    assert rep_x.unreachable > 0.0  # the 'qq' branch is unrepresentable for x


def test_mass_propagation_runs_once_per_agent():
    """AUDIT-6: coverage must not re-run the per-step hot path."""
    from sahf.sheaf.pipeline import Stage8Pipeline

    calls = {"n": 0}
    original = BytePrefixTree.cover_mass

    def counting(self, agent, probs, restrict=None):
        calls["n"] += 1
        return original(self, agent, probs, restrict)

    a = make_vocab("a", [b" the", b" a"])
    b = make_vocab("b", [b" th", b" a", b"e"])
    agents = [StaticAgent("a", a), StaticAgent("b", b)]
    agents[0].set("ctx", dist(2, {0: 0.8, 1: 0.2}))
    agents[1].set("ctx", dist(3, {0: 0.8, 1: 0.2, 2: 0.0}))

    BytePrefixTree.cover_mass = counting
    try:
        Stage8Pipeline(agents, mode="topk_union", k=10).step("ctx")
    finally:
        BytePrefixTree.cover_mass = original
    assert calls["n"] == 2, f"cover_mass ran {calls['n']} times for 2 agents"


def test_generate_never_splits_a_multibyte_character():
    """AUDIT-7: a chunk ending mid-character must not corrupt the context."""
    from sahf.sheaf.pipeline import Stage8Pipeline

    u = "café".encode()  # b'caf\xc3\xa9'
    va = make_vocab("a", [u[:4], u[4:]])  # u[:4] ends mid-character
    vb = make_vocab("b", [u[:4], u[4:]])
    agents = [StaticAgent("a", va), StaticAgent("b", vb)]
    for ag in agents:
        ag.set("", dist(2, {0: 1.0}))
        ag.set("caf", dist(2, {1: 1.0}))
    pipe = Stage8Pipeline(agents, mode="topk_union", k=8, min_support=1)
    text, _ = pipe.generate("", max_steps=1)
    assert "\ufffd" not in text, f"got {text!r}"


def test_stop_mass_is_surfaced_not_renormalised_away():
    """AUDIT-8: stop tokens have no byte image, so they must be reported."""
    from sahf.sheaf.pipeline import Stage8Pipeline

    a = make_vocab("a", [b" x", b" y"])
    b = make_vocab("b", [b" x", b" y"])
    agents = [StaticAgent("a", a), StaticAgent("b", b)]
    for ag in agents:
        ag.set("ctx", dist(2, {0: 0.9, 1: 0.1}), stop=0.95)
    res = Stage8Pipeline(agents, mode="topk_union", k=8).step("ctx")
    assert res.should_stop
    assert res.stop_mass["a"] == 0.95

    for ag in agents:
        ag.set("ctx2", dist(2, {0: 0.9, 1: 0.1}), stop=0.01)
    assert not Stage8Pipeline(agents, mode="topk_union", k=8).step("ctx2").should_stop


def test_generation_halts_on_consensus_stop():
    """AUDIT-8b: generate() must terminate when the ensemble votes to stop."""
    from sahf.sheaf.pipeline import Stage8Pipeline

    a = make_vocab("a", [b"ab", b"cd"])
    b = make_vocab("b", [b"ab", b"cd"])
    agents = [StaticAgent("a", a), StaticAgent("b", b)]
    for ag in agents:
        ag.set("", dist(2, {0: 1.0}), stop=0.0)
        ag.set("ab", dist(2, {1: 1.0}), stop=0.99)
    text, trace = Stage8Pipeline(agents, mode="topk_union", k=8, min_support=1).generate(
        "", max_steps=5
    )
    assert text == "ab"
    assert trace[-1].should_stop
