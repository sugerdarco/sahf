"""One regression test per issue found in the Stage 8 integration audit.

See docs/STAGE8_INTEGRATION_AUDIT.md. `audit/audit_integration.py` is the
harness that found them; these pin the fixes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sahf.gate import GateThresholds
from sahf.sheaf import SheafOrchestrator, StaticAgent, UpstreamAgent, VocabSpec
from sahf.sheaf.prefix_tree import BytePrefixTree


def vocab(name, toks):
    return VocabSpec.from_mapping({i: b for i, b in enumerate(toks)}, name=name)


def dist(n, w):
    p = np.zeros(n)
    for i, v in w.items():
        p[i] = v
    return p / p.sum()


def simple_agents(n=3):
    specs = [vocab(f"a{i}", [b" Paris", b" Berlin", b" Rome"]) for i in range(n)]
    ags = [StaticAgent(s.name, s) for s in specs]
    for ag in ags:
        ag.set("ctx", dist(3, {0: 0.8, 1: 0.15, 2: 0.05}))
    return ags


class Tok:
    def __init__(self, mapping, eos=None, added=None, bos=None):
        self._v = mapping
        self.all_special_ids = []
        self.added_tokens_decoder = added or {}
        self.eos_token_id = eos
        self.bos_token_id = bos
        self.vocab_size = len(mapping)
        self.name_or_path = "fake"

    def get_vocab(self):
        return self._v

    def encode(self, text, add_special_tokens=False):
        return [0]

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        class E:
            input_ids = torch.tensor([[0, 1]] if text else [[]], dtype=torch.long)

        return E()


class Agent:
    def __init__(self, tok, size, bias_id=0, bias=5.0):
        self.name = "fake"
        self.tokenizer = tok
        self.vocab_size = size
        self._size, self._bias_id, self._bias = size, bias_id, bias

    def next_logits(self, input_ids):
        if input_ids.numel() == 0:
            raise RuntimeError("empty input_ids reached the model")
        lg = torch.zeros(1, self._size)
        lg[0, self._bias_id] = self._bias
        return lg


VOC = {"Ġthe": 0, "Ġcat": 1, "<|eot_id|>": 2, "<|end_of_text|>": 3}


# --------------------------------------------------------------------------


def test_tree_and_mass_are_computed_once_per_step():
    """INT-1/2: the gate and reconcile each built their own tree and mass."""
    calls = {"tree": 0, "mass": 0}
    orig_build, orig_mass = BytePrefixTree.from_vocabs.__func__, BytePrefixTree.cover_mass

    def cb(cls, *a, **k):
        calls["tree"] += 1
        return orig_build(cls, *a, **k)

    def cm(self, agent, probs, restrict=None):
        calls["mass"] += 1
        return orig_mass(self, agent, probs, restrict)

    BytePrefixTree.from_vocabs = classmethod(cb)
    BytePrefixTree.cover_mass = cm
    try:
        SheafOrchestrator(simple_agents(), GateThresholds(), k=16).step("ctx", 0)
    finally:
        BytePrefixTree.from_vocabs = classmethod(orig_build)
        BytePrefixTree.cover_mass = orig_mass

    assert calls["tree"] == 1, f"tree built {calls['tree']} times"
    assert calls["mass"] == 3, f"cover_mass ran {calls['mass']} times for 3 agents"


def test_reused_mass_gives_the_same_result_as_recomputing():
    """The reuse path must be a pure optimisation, not a behaviour change."""
    from sahf.sheaf import SheafReconciler, Stage8Pipeline

    ags = simple_agents()
    probs = [a.next_token_probs("ctx") for a in ags]
    pipe = Stage8Pipeline(ags, mode="topk_union", k=16)
    tree = pipe.build_tree(probs)
    mass = ([], [])
    for a in range(3):
        c, t = tree.cover_mass(a, probs[a])
        mass[0].append(c)
        mass[1].append(t)

    rec = SheafReconciler(fusion="auto")
    fresh = rec.reconcile(tree, probs)
    reused = rec.reconcile(tree, probs, mass=mass)

    assert np.allclose(fresh.cover, reused.cover)
    assert set(fresh.conditionals) == set(reused.conditionals)
    for node in fresh.conditionals:
        assert np.allclose(fresh.conditionals[node], reused.conditionals[node])


def test_byte_entropy_threshold_is_independent_of_token_space_theta_h():
    """INT-3: theta_H is a token-space number and must not be inherited silently."""
    o_default = SheafOrchestrator(simple_agents(), GateThresholds(entropy=2.0), k=16)
    assert o_default.byte_entropy_threshold == 2.0
    o_custom = SheafOrchestrator(
        simple_agents(), GateThresholds(entropy=2.0), k=16, byte_entropy_threshold=0.4
    )
    assert o_custom.byte_entropy_threshold == 0.4
    _, record, _ = o_default.step("ctx", 0)
    assert "entropy" in record, "entropy must be logged so theta_H can be calibrated"


def test_step_record_is_json_serializable():
    """RunLogger writes steps.jsonl with json.dumps; numpy scalars would break it."""
    _, record, _ = SheafOrchestrator(simple_agents(), GateThresholds(), k=16).step("ctx", 0)
    json.loads(json.dumps(record))


def test_empty_prompt_does_not_crash_the_forward_pass():
    """INT-4: generate(prompt="") encoded to zero tokens and the model raised."""
    agent = UpstreamAgent(Agent(Tok(VOC), 4))
    p = agent.next_token_probs("")
    assert abs(p.sum() - 1.0) < 1e-9


def test_stop_mass_covers_all_end_of_turn_tokens():
    """INT-5: Llama-3 generates <|eot_id|> but eos_token_id is <|end_of_text|>."""
    tok = Tok(VOC, eos=3, added={2: "<|eot_id|>", 3: "<|end_of_text|>"})
    agent = UpstreamAgent(Agent(tok, 4, bias_id=2, bias=20.0))

    assert 2 in agent.stop_token_ids and 3 in agent.stop_token_ids
    assert agent.stop_probability("hello") > 0.5

    # and an explicit override is honoured
    only3 = UpstreamAgent(Agent(tok, 4, bias_id=2, bias=20.0), stop_token_ids={3})
    assert only3.stop_probability("hello") < 0.5


def test_context_cache_is_bounded():
    """INT-6: one vocab-sized array per generated step, retained forever."""
    agent = UpstreamAgent(Agent(Tok(VOC), 4))
    for i in range(1, 60):
        agent.next_token_probs("x" * i)
    assert len(agent._cache) <= UpstreamAgent.CACHE_SIZE
    assert len(agent._stop) <= UpstreamAgent.CACHE_SIZE


def test_no_dead_branch_in_direct_hf_agent():
    """INT-7: a loop over pad_token_id that discarded its own value."""
    import inspect

    from sahf.sheaf import adapters

    assert 'for extra in ("pad_token_id", "eot_token_id")' not in inspect.getsource(adapters)


def test_demo_docstring_uses_the_right_module_path():
    """INT-8: the standalone package path survived the move into this repo."""
    import inspect

    from sahf.sheaf import demo

    assert "sahf_sheaf.demo" not in inspect.getdoc(demo)


def test_shared_stalks_give_identical_results():
    """INT-9: the gate and reconcile each computed the per-node sections.

    The cache must be a pure optimisation. If this ever diverges, the fused
    conditionals differ depending on whether the gate ran — a silent behaviour
    change dependent on a performance path.
    """
    from sahf.sheaf import SheafReconciler, Stage8Pipeline
    from sahf.sheaf.reconciler import byte_level_sections

    ags = simple_agents()
    probs = [a.next_token_probs("ctx") for a in ags]
    pipe = Stage8Pipeline(ags, mode="topk_union", k=32)
    tree = pipe.build_tree(probs)
    covers, terms = [], []
    for a in range(3):
        c, t = tree.cover_mass(a, probs[a])
        covers.append(c)
        terms.append(t)
    _root, _spread, stalks = byte_level_sections(tree, covers, terms)

    rec = SheafReconciler(fusion="auto")
    fresh = rec.reconcile(tree, probs)
    shared = rec.reconcile(tree, probs, mass=(covers, terms), stalks=stalks)

    assert set(fresh.conditionals) == set(shared.conditionals)
    for node in fresh.conditionals:
        assert np.allclose(fresh.conditionals[node], shared.conditionals[node])
        assert fresh.reports[node].support == shared.reports[node].support
        assert fresh.reports[node].horizon == shared.reports[node].horizon
    assert np.allclose(fresh.cover, shared.cover)


def test_gate_does_not_change_the_consensus_it_routes():
    """Whichever path the gate picks, the tree it built must be the one used."""
    ags = simple_agents()
    orch = SheafOrchestrator(ags, GateThresholds(), k=32)
    chunk, record, res = orch.step("ctx", 0)
    assert chunk == res.consensus_bytes
    assert record["tree_nodes"] == res.tree_stats.n_nodes
