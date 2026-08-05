"""The prefix tree is built ONCE by a separate command and reused by every run.

It is a property of the ensemble's vocabularies, not of any prompt, so it must
not be rebuilt per query or per decoding step. These tests cover the artifact
round-trip and the `prebuilt` decode path that consumes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from sahf.gate import GateThresholds
from sahf.sheaf import (
    BytePrefixTree,
    SheafOrchestrator,
    Stage8Pipeline,
    StaticAgent,
    VocabSpec,
)

TARGET = b" Paris is the capital of France"


def covering_vocab(name: str, max_len: int) -> VocabSpec:
    toks = sorted({TARGET[i : i + n] for i in range(len(TARGET)) for n in range(1, max_len + 1)})
    return VocabSpec.from_mapping({i: b for i, b in enumerate(toks)}, name=name)


@pytest.fixture
def specs():
    return [covering_vocab("coarse", 6), covering_vocab("medium", 3), covering_vocab("fine", 1)]


@pytest.fixture
def probs(specs):
    rng = np.random.default_rng(0)
    out = []
    for vs in specs:
        p = rng.random(vs.size) ** 4
        p[3] += 5.0
        out.append(p / p.sum())
    return out


# --------------------------------------------------------------------------
# artifact round-trip
# --------------------------------------------------------------------------


def test_saved_tree_reloads_identically(tmp_path, specs, probs):
    tree = BytePrefixTree.from_vocabs(specs)
    path = tree.save(tmp_path / "prefix_tree.npz")
    assert path.exists()
    loaded = BytePrefixTree.load(path)

    assert loaded.n_nodes == tree.n_nodes
    assert loaded.n_agents == tree.n_agents
    assert loaded.children == tree.children
    assert loaded.parent == tree.parent
    assert loaded.depth == tree.depth
    assert loaded.max_depth_seen == tree.max_depth_seen

    for a in range(tree.n_agents):
        c1, t1 = tree.cover_mass(a, probs[a])
        c2, t2 = loaded.cover_mass(a, probs[a])
        assert np.allclose(c1, c2)
        assert np.allclose(t1, t2)


def test_saved_tree_carries_the_vocabularies(tmp_path, specs):
    """A run must not need to load tokenizers just to reconcile bytes."""
    loaded = BytePrefixTree.load(BytePrefixTree.from_vocabs(specs).save(tmp_path / "t.npz"))
    for original, restored in zip(specs, loaded.vocabs):
        assert restored.name == original.name
        assert restored.scheme == original.scheme
        assert restored.token_bytes == original.token_bytes


def test_terminals_survive_the_round_trip(tmp_path, specs):
    tree = BytePrefixTree.from_vocabs(specs)
    loaded = BytePrefixTree.load(tree.save(tmp_path / "t.npz"))
    node = tree.find(b" Paris")
    assert node >= 0
    for a in range(tree.n_agents):
        assert loaded.terminals_at(node, a) == tree.terminals_at(node, a)


def test_format_version_mismatch_is_refused(tmp_path, specs):
    import json

    path = BytePrefixTree.from_vocabs(specs).save(tmp_path / "t.npz")
    with np.load(path, allow_pickle=False) as z:
        arrays = dict(z)
    meta = json.loads(bytes(arrays["meta"]).decode())
    meta["format_version"] = 999
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="format"):
        BytePrefixTree.load(path)


# --------------------------------------------------------------------------
# sparse propagation -- what makes a prebuilt tree usable per step
# --------------------------------------------------------------------------


def test_restricted_propagation_matches_full_propagation(specs, probs):
    tree = BytePrefixTree.from_vocabs(specs)
    for a in range(tree.n_agents):
        full_c, full_t = tree.cover_mass(a, probs[a])
        all_ids = np.arange(specs[a].size)
        sparse_c, sparse_t = tree.cover_mass(a, probs[a], restrict=all_ids)
        assert np.allclose(full_c, sparse_c)
        assert np.allclose(full_t, sparse_t)


def test_restricted_propagation_touches_only_the_restricted_paths(specs, probs):
    tree = BytePrefixTree.from_vocabs(specs)
    cover, _ = tree.cover_mass(0, probs[0], restrict=np.array([3]))
    touched = tree.touched_nodes(0)
    assert touched is not None and touched.size <= tree.max_depth_seen + 1
    assert np.count_nonzero(cover) == touched.size


def test_scratch_buffers_are_cleared_between_steps(specs, probs):
    """Reused buffers must not leak the previous step's mass."""
    tree = BytePrefixTree.from_vocabs(specs)
    tree.cover_mass(0, probs[0], restrict=np.arange(specs[0].size))
    first_nonzero = np.count_nonzero(tree.cover_mass(0, probs[0], restrict=np.array([3]))[0])
    direct = BytePrefixTree.from_vocabs(specs).cover_mass(0, probs[0], restrict=np.array([3]))[0]
    assert first_nonzero == np.count_nonzero(direct)


# --------------------------------------------------------------------------
# the decode path
# --------------------------------------------------------------------------


def test_prebuilt_matches_rebuild_per_step(specs, probs):
    """Same consensus whether the tree was prebuilt or rebuilt for this step."""
    agents = [StaticAgent(v.name, v) for v in specs]
    tree = BytePrefixTree.from_vocabs(specs)

    pre = Stage8Pipeline(agents, mode="prebuilt", tree=tree, k=16)
    tk = Stage8Pipeline(agents, mode="topk_union", k=16)

    a = pre.step_from_probs(probs)
    b = tk.step_from_probs(probs)
    assert a.consensus_bytes == b.consensus_bytes


def test_prebuilt_mode_does_not_rebuild_the_tree(specs, probs):
    agents = [StaticAgent(v.name, v) for v in specs]
    tree = BytePrefixTree.from_vocabs(specs)
    pipe = Stage8Pipeline(agents, mode="prebuilt", tree=tree, k=16)

    calls = {"n": 0}
    original = BytePrefixTree.from_vocabs.__func__

    def counting(cls, *a, **k):
        calls["n"] += 1
        return original(cls, *a, **k)

    BytePrefixTree.from_vocabs = classmethod(counting)
    try:
        for _ in range(3):
            pipe.step_from_probs(probs)
    finally:
        BytePrefixTree.from_vocabs = classmethod(original)
    assert calls["n"] == 0, "prebuilt mode rebuilt the tree"


def test_prebuilt_mode_requires_a_tree(specs):
    agents = [StaticAgent(v.name, v) for v in specs]
    with pytest.raises(ValueError, match="prebuilt"):
        Stage8Pipeline(agents, mode="prebuilt")


def test_tree_agent_count_mismatch_is_refused(specs):
    agents = [StaticAgent(v.name, v) for v in specs[:2]]
    tree = BytePrefixTree.from_vocabs(specs)  # 3 agents
    with pytest.raises(ValueError, match="agents"):
        Stage8Pipeline(agents, mode="prebuilt", tree=tree)


def test_orchestrator_accepts_a_prebuilt_tree(tmp_path, specs, probs):
    agents = [StaticAgent(v.name, v) for v in specs]
    for ag, p in zip(agents, probs):
        ag.set("ctx", p)
    loaded = BytePrefixTree.load(BytePrefixTree.from_vocabs(specs).save(tmp_path / "t.npz"))

    orch = SheafOrchestrator(agents, GateThresholds(), k=16, tree=loaded)
    chunk, record, res = orch.step("ctx", 0)

    assert chunk
    assert record["tree_nodes"] == loaded.n_nodes
    plain = SheafOrchestrator(agents, GateThresholds(), k=16).step("ctx", 0)[0]
    assert chunk == plain, "prebuilt and per-step trees disagreed"


def test_decode_ties_break_deterministically(specs):
    """Exact ties must not depend on a tree's child iteration order."""
    v = [VocabSpec.from_mapping({0: b"ab", 1: b"ac"}, name=f"a{i}") for i in range(2)]
    agents = [StaticAgent(s.name, s) for s in v]
    tied = np.array([0.5, 0.5])
    tree = BytePrefixTree.from_vocabs(v)

    pre = Stage8Pipeline(agents, mode="prebuilt", tree=tree, k=8, min_support=1)
    tk = Stage8Pipeline(agents, mode="topk_union", k=8, min_support=1)
    assert (
        pre.step_from_probs([tied, tied]).consensus_bytes
        == tk.step_from_probs([tied, tied]).consensus_bytes
    )
