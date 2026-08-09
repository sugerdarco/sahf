"""
SAHF Distribution Fusion — cross-tokenizer ensembling.

Per-token fusion of N language models whose tokenizers DO NOT agree. Because
"token v" denotes different strings in different vocabularies, agents cannot be
combined in token space at all: Stage 8 first reconciles them in a shared
byte-prefix space, and Stages 1/5/6/7 then run per node inside it.

  sahf.amplitude   Stage 1 — psi = sqrt(softmax(z))
  sahf.gate        Stage 2 — divergence gate
  sahf.fusion      Stage 5 — chordal mean on the sphere
  sahf.robust      Stages 6, 7 — MAD screen, Weiszfeld geometric median
  sahf.sheaf       Stage 8 — byte-prefix reconciliation and the decode loop

Entry points: run_sheaf.py (generate), build_prefix_tree.py (optional prebuilt
tree), run_batch_prompts.py (many prompts). Benchmarking against DeePEn lives in
deepen/ and nothing here depends on it.

The single-tokenizer variant (FusionOrchestrator, run.py, config.yaml) has been
removed — with a shared tokenizer there is no vocabulary mismatch to resolve, and
this project now targets only the mismatched case. ARCHITECTURE.md and HISTORY.md
are kept as the historical record of that earlier design.
"""

from .amplitude import softmax_to_amplitude
from .gate import GateThresholds, GateDecision, divergence_gate
from .fusion import fast_mean_fusion
from .robust import detect_outliers, weiszfeld_geometric_median
from .agents import HFAgent, MockAgent, PoisonedAgentWrapper
from .logger import RunLogger

__all__ = [
    "softmax_to_amplitude",
    "GateThresholds",
    "GateDecision",
    "divergence_gate",
    "fast_mean_fusion",
    "detect_outliers",
    "weiszfeld_geometric_median",
    "HFAgent",
    "MockAgent",
    "PoisonedAgentWrapper",
    "RunLogger",
]

__version__ = "0.1.0"
