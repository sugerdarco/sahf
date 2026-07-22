"""
sahf_lite — fast, single-tokenizer variant of the SAHF distribution-fusion pipeline.

This package intentionally implements only Stages 1-7 of the full 8-stage design
(see ARCHITECTURE.md). Stage 8 (Sheaf Reconciliation) is dropped because both
default agents share one tokenizer, so there is no vocabulary mismatch to resolve.
"""

from .amplitude import softmax_to_amplitude
from .gate import GateThresholds, GateDecision, divergence_gate
from .fusion import fast_mean_fusion
from .robust import detect_outliers, weiszfeld_geometric_median
from .agents import HFAgent, MockAgent, PoisonedAgentWrapper, assert_shared_vocab_size
from .orchestrator import FusionOrchestrator
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
    "assert_shared_vocab_size",
    "FusionOrchestrator",
    "RunLogger",
]

__version__ = "0.1.0"
