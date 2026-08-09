"""Adversarial probes of the Stage 8 INTEGRATION layer.

audit/audit_stage8.py covers the library internals (vocab / tree / sheaf /
pipeline). Nothing covered the code written to bolt it onto this repo:
UpstreamAgent, SheafOrchestrator, the byte-level gate, and the run/logging path.
That is what this probes.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sahf.agents import MockAgent, PoisonedAgentWrapper
from sahf.gate import GateThresholds
from sahf.sheaf import SheafOrchestrator, StaticAgent, UpstreamAgent, VocabSpec
from sahf.sheaf.prefix_tree import BytePrefixTree

FAILS = []


def check(area, name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          -> {detail}" if not ok else ""))
    if not ok:
        FAILS.append(f"[{area}] {name}: {detail}")


def vocab(name, toks):
    return VocabSpec.from_mapping({i: b for i, b in enumerate(toks)}, name=name)


def dist(n, w):
    p = np.zeros(n)
    for i, v in w.items():
        p[i] = v
    return p / p.sum()


def simple_agents(n=3):
    v = [vocab(f"a{i}", [b" Paris", b" Berlin", b" Rome"]) for i in range(n)]
    ags = [StaticAgent(s.name, s) for s in v]
    for ag in ags:
        ag.set("ctx", dist(3, {0: 0.8, 1: 0.15, 2: 0.05}))
    return ags


# =====================================================================
print("\nORCHESTRATOR — redundant work")
# =====================================================================

ags = simple_agents()
calls = {"tree": 0, "mass": 0}
orig_build = BytePrefixTree.from_vocabs.__func__
orig_mass = BytePrefixTree.cover_mass


def counting_build(cls, *a, **k):
    calls["tree"] += 1
    return orig_build(cls, *a, **k)


def counting_mass(self, agent, probs, restrict=None):
    calls["mass"] += 1
    return orig_mass(self, agent, probs, restrict)


BytePrefixTree.from_vocabs = classmethod(counting_build)
BytePrefixTree.cover_mass = counting_mass
SheafOrchestrator(ags, GateThresholds(), k=16).step("ctx", 0)
BytePrefixTree.from_vocabs = classmethod(orig_build)
BytePrefixTree.cover_mass = orig_mass

check(
    "orchestrator",
    "tree is built once per decoding step",
    calls["tree"] == 1,
    f"built {calls['tree']} times: the gate builds one and reconcile builds another",
)
check(
    "orchestrator",
    "mass propagation runs once per agent per step",
    calls["mass"] == 3,
    f"ran {calls['mass']} times for 3 agents (expected 3)",
)

# =====================================================================
print("\nORCHESTRATOR — gate behaviour")
# =====================================================================

# Is the entropy half of the two-factor gate ever the deciding factor in byte
# space? Root sections over 256 bytes are far peakier than a 150k-token vocab.
ents = []
for spread in (0.0, 0.5, 1.0):
    v = [vocab(f"a{i}", [b" Paris", b" Berlin"]) for i in range(3)]
    a2 = [StaticAgent(s.name, s) for s in v]
    a2[0].set("c", dist(2, {0: 1 - spread / 2, 1: spread / 2}))
    a2[1].set("c", dist(2, {0: 0.9, 1: 0.1}))
    a2[2].set("c", dist(2, {0: 0.9, 1: 0.1}))
    _, rec, _ = SheafOrchestrator(a2, GateThresholds(), k=8).step("c", 0)
    ents.append(rec["entropy"])
# The entropy condition cannot discriminate in byte space: it is ~0 when the first
# byte is near-deterministic (very common) and 2.3-2.7 nats when it is not, against
# a token-space theta_H of 2.0 and a ceiling of ln(256)=5.55. That is a property of
# the space, not a fixable defect, so what is asserted is the mitigation: the
# threshold must not silently inherit config_sheaf.yaml's value, and the observed entropy
# must be logged so it can be calibrated against real models.
orch_probe = SheafOrchestrator(simple_agents(), GateThresholds(entropy=2.0), k=16)
check(
    "orchestrator",
    "byte entropy threshold is separately settable, not inherited silently",
    hasattr(orch_probe, "byte_entropy_threshold")
    and SheafOrchestrator(
        simple_agents(), GateThresholds(entropy=2.0), k=16, byte_entropy_threshold=0.5
    ).byte_entropy_threshold
    == 0.5,
    "no way to set a byte-space theta_H independent of the token-space one",
)
_, _rec_ent, _ = orch_probe.step("ctx", 0)
check(
    "orchestrator",
    "observed byte entropy is logged for calibration",
    "entropy" in _rec_ent and "divergence" in _rec_ent,
    "steps.jsonl carries no entropy figure to calibrate theta_H from",
)

# =====================================================================
print("\nLOGGING — records must survive json.dumps")
# =====================================================================

_, record, _ = SheafOrchestrator(simple_agents(), GateThresholds(), k=16).step("ctx", 0)
try:
    json.dumps(record)
    ok, detail = True, ""
except TypeError as e:
    ok, detail = False, f"RunLogger writes steps.jsonl with json.dumps: {e}"
check("logging", "step record is JSON-serializable", ok, detail)

# =====================================================================
print("\nUPSTREAM AGENT — interface compatibility")
# =====================================================================


class FakeTok:
    def __init__(self, mapping, eos=None):
        self._v = mapping
        self.all_special_ids = []
        self.added_tokens_decoder = {}
        self.eos_token_id = eos
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


class FakeAgent:
    def __init__(self, name, tok, size, stop_bias=0.0, stop_id=None):
        self.name = name
        self.tokenizer = tok
        self.vocab_size = size
        self._size = size
        self._stop_bias = stop_bias
        self._stop_id = stop_id

    def next_logits(self, input_ids):
        if input_ids.numel() == 0:
            raise RuntimeError("empty input_ids reached the model")
        lg = torch.zeros(1, self._size)
        lg[0, 0] = 5.0
        if self._stop_bias and self._stop_id is not None:
            lg[0, self._stop_id] = self._stop_bias
        return lg


VOC = {"Ġthe": 0, "Ġcat": 1, "<|eot_id|>": 2, "<|end_of_text|>": 3}

# P1: PoisonedAgentWrapper must pass through
tok = FakeTok(VOC)
try:
    wrapped = PoisonedAgentWrapper(FakeAgent("x", tok, 4), mode="invert")
    UpstreamAgent(wrapped)
    ok, detail = True, ""
except Exception as e:
    ok, detail = False, f"{type(e).__name__}: {e}"
check("upstream_agent", "PoisonedAgentWrapper can be wrapped", ok, detail)

# P2: MockAgent must be rejected (it has no tokenizer)
try:
    UpstreamAgent(MockAgent("m", vocab_size=8))
    ok, detail = False, "accepted a tokenizer-less agent"
except ValueError:
    ok, detail = True, ""
check("upstream_agent", "tokenizer-less agent rejected clearly", ok, detail)

# P3: empty prompt
agent = UpstreamAgent(FakeAgent("e", FakeTok(VOC), 4))
try:
    agent.next_token_probs("")
    ok, detail = True, ""
except Exception as e:
    ok, detail = False, f"generate(prompt='') hits this: {type(e).__name__}: {e}"
check("upstream_agent", "empty context does not crash the forward pass", ok, detail)

# P4: multi-EOT models (Llama-3 has <|eot_id|> AND <|end_of_text|>; tokenizer
# .eos_token_id exposes only one of them)
tok_multi = FakeTok(VOC, eos=3)  # <|end_of_text|>
tok_multi.added_tokens_decoder = {2: "<|eot_id|>", 3: "<|end_of_text|>"}
# the model overwhelmingly predicts <|eot_id|> (id 2), which is NOT eos_token_id
ag_multi = UpstreamAgent(FakeAgent("llama-ish", tok_multi, 4, stop_bias=20.0, stop_id=2))
stop = ag_multi.stop_probability("hello")
check(
    "upstream_agent",
    "stop mass covers every end-of-turn token, not just tokenizer.eos_token_id",
    stop > 0.5,
    f"stop_probability={stop:.4f} while the model put ~1.0 on <|eot_id|>; "
    "config_sheaf.yaml lists Llama-3.2, which stops on <|eot_id|>, so generation "
    "would never terminate",
)

# P5: context cache growth
ag_cache = UpstreamAgent(FakeAgent("c", FakeTok(VOC), 4))
for i in range(1, 51):
    ag_cache.next_token_probs("x" * i)
n_cached = len(ag_cache._cache)
check(
    "upstream_agent",
    "per-context cache is bounded",
    n_cached <= 16,
    f"{n_cached} contexts retained; every step of every generation is kept for the "
    "process lifetime, each a full vocab-sized float64 array",
)

# =====================================================================
print("\nDEAD CODE / DOC DRIFT")
# =====================================================================

src = (Path(__file__).resolve().parents[1] / "sahf" / "sheaf" / "adapters.py").read_text()
check(
    "hygiene",
    "no no-op branch in DirectHFAgent stop-id collection",
    'for extra in ("pad_token_id", "eot_token_id")' not in src,
    "loops over pad_token_id then discards it via `and extra == 'eot_token_id'`",
)

demo_src = (Path(__file__).resolve().parents[1] / "sahf" / "sheaf" / "demo.py").read_text()
check(
    "hygiene",
    "demo docstring points at the right module path",
    "sahf_sheaf.demo" not in demo_src,
    "says `python -m sahf_sheaf.demo`; in this repo it is `python -m sahf.sheaf.demo`",
)

# =====================================================================
print("\nNUMERICS — inherited dtype")
# =====================================================================

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lg = torch.randn(1, 4096)
    from sahf.amplitude import softmax_to_amplitude

    p32 = (softmax_to_amplitude(lg) ** 2)[0].double().numpy()
    p16 = (softmax_to_amplitude(lg.bfloat16()) ** 2)[0].double().numpy()
rel = float(np.abs(p32 - p16).max() / p32.max())
check(
    "numerics",
    "bfloat16 logits preserve top-k ordering adequately",
    rel < 0.05,
    f"max relative deviation {rel:.3f} between float32 and bfloat16 probabilities "
    "(config_sheaf.yaml defaults to bfloat16)",
)

# =====================================================================
print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} ISSUES FOUND\n")
    for f in FAILS:
        print(" *", f)
else:
    print("no issues found")
print("=" * 70)
sys.exit(1 if FAILS else 0)
