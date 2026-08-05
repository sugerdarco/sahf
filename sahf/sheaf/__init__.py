"""
Stage 8 — Sheaf Reconciliation.

The one stage the fast version deliberately drops (see ARCHITECTURE.md): with a
shared tokenizer there is no vocabulary mismatch to resolve, so Stages 1-7 can
fuse amplitude vectors token-for-token. The moment the agents' tokenizers differ,
"token v" stops meaning the same thing across agents and those vectors are not
even the same length, let alone comparable.

Stage 8 builds the shared byte-level base space that makes them comparable at
all: every agent's vocabulary is expressed in raw bytes, a union byte-prefix tree
is built over them, and each agent's distribution is pushed onto it. At every
node the agents' local sections live on the SAME simplex, so Stages 1/5/6/7 apply
per node, unchanged. Gluing those conditionals down from the root yields one
byte-level consensus, which is then translated back into each agent's own token
space for the next generation step.

Relationship to Stages 1-7
--------------------------
This subpackage does not replace any of them. `SheafOrchestrator` reuses
`sahf.amplitude` (Stage 1) and `sahf.gate` (Stage 2) directly. Stages 5/6/7 are
transcribed into numpy inside `reconciler.py` because they run per tree node --
hundreds of times per token on small vectors, where torch's per-call overhead
would dominate. `tests/test_sheaf_parity.py` asserts the transcriptions match
`sahf.fusion` and `sahf.robust` numerically, so the repo's Stage 5/6/7 remain the
single source of truth for what those stages mean.

Quickstart
----------
    from sahf.agents import HFAgent
    from sahf.gate import GateThresholds
    from sahf.sheaf import SheafOrchestrator, UpstreamAgent

    agents = [UpstreamAgent(HFAgent(n)) for n in ("Qwen/...", "google/gemma-...", "...")]
    orch = SheafOrchestrator(agents, GateThresholds())
    text, history = orch.generate("The capital of France is")

Offline demo, no models required:

    python -m sahf.sheaf.demo
"""

from .adapters import (
    DirectHFAgent,
    StaticAgent,
    UpstreamAgent,
    assert_distinct_tokenizers,
    top_k_ids,
)
from .orchestrator import SheafOrchestrator
from .pipeline import Stage8Pipeline, Stage8Result
from .prefix_tree import BytePrefixTree, TreeStats
from .reconciler import (
    GlobalSection,
    NodeReport,
    ProjectionReport,
    SheafReconciler,
    detect_outliers,
    geometric_median,
    hellinger,
    mean_fuse,
    to_amplitude,
    to_probability,
)
from .vocab import VocabSpec, bytes_to_unicode, detect_scheme

__all__ = [
    "SheafOrchestrator",
    "Stage8Pipeline",
    "Stage8Result",
    "BytePrefixTree",
    "TreeStats",
    "SheafReconciler",
    "GlobalSection",
    "NodeReport",
    "ProjectionReport",
    "VocabSpec",
    "UpstreamAgent",
    "assert_distinct_tokenizers",
    "DirectHFAgent",
    "StaticAgent",
    "top_k_ids",
    "to_amplitude",
    "to_probability",
    "mean_fuse",
    "geometric_median",
    "detect_outliers",
    "hellinger",
    "bytes_to_unicode",
    "detect_scheme",
]
