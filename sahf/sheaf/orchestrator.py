"""
SheafOrchestrator — the per-token decode loop for agents with DIFFERENT tokenizers.

The project previously carried a shared-tokenizer orchestrator, which compared
amplitude vectors token-for-token and fed one `input_ids` tensor back to every
agent. That variant has been removed; this one drops the shared-tokenizer
assumption, so both of those steps work differently:

  comparison   agents' vectors are over different vocabularies and different
               lengths, so they are pushed onto a shared byte-prefix tree first
               and compared per node (Stage 8).
  feedback     there is no shared token id to append. The consensus is a byte
               chunk; it is appended as TEXT and every agent re-encodes the
               extended context with its own tokenizer.

Everything else is deliberately the same, and where it is the same it reuses the
repo's code rather than reimplementing it:

  Stage 1  `sahf.amplitude.softmax_to_amplitude`, called inside `UpstreamAgent`.
  Stage 2  `sahf.gate.mean_entropy` and `pairwise_amplitude_spread`, applied to
           the agents' byte-level root sections. Same GateThresholds, same
           two-factor Path A / Path B routing.
  Stage 5  chordal mean, per tree node.
  Stage 6  MAD outlier screen, per tree node, same `mad_multiplier` as config_sheaf.yaml.
  Stage 7  Weiszfeld geometric median, per node, only where Stage 6 fires.

Stages 5/6/7 run as numpy transcriptions inside `reconciler.py` because they
execute per node -- hundreds of times per token -- where torch's per-call
overhead would dominate. `tests/test_sheaf_parity.py` pins them numerically to
`sahf.fusion` and `sahf.robust`.

Path A here means "fuse with the plain mean and skip the robust machinery
entirely", exactly as Path A in the fast version skips fusion; it does not skip
building the tree, because without the tree there is no shared space in which to
read off a next token at all.

The per-step record keeps the fast version's field names (`path`, `escalated`,
`entropy`, `divergence`) so existing log tooling and `out/runs/*/steps.jsonl`
readers keep working, and adds Stage 8 fields alongside them.
"""

from __future__ import annotations

import codecs
import time
from typing import List, Optional

import numpy as np

from ..gate import GateDecision, GateThresholds, mean_entropy, pairwise_amplitude_spread  # noqa: F401
from .pipeline import Stage8Pipeline
from .reconciler import byte_level_sections, to_amplitude


class SheafOrchestrator:
    """Per-token decode loop across agents with mismatched tokenizers.

    Parameters
    ----------
    agents        : `UpstreamAgent` (or anything exposing `next_token_probs(text)`,
                    `vocab_spec()`, optionally `stop_probability(text)`)
    thresholds    : the same `GateThresholds` Stages 1-7 use
    weights       : per-agent fusion weights, defaults to uniform
    max_new_bytes : byte budget for `generate`, the analogue of max_new_tokens
    mad_multiplier: Stage 6 threshold, mirrors config_sheaf.yaml gate.mad_multiplier
    k             : top-k width per agent. Default 16 on benchmark evidence --
                    quality is flat from k=8 to k=256 while cost grows 23x
                    (docs/STAGE8_BENCHMARK_RESULTS.md).
    min_support   : how many agents must still see ahead for a byte to be emitted.
                    Defaults to a simple majority.
    """

    def __init__(
        self,
        agents: List,
        thresholds: GateThresholds,
        weights: Optional[List[float]] = None,
        max_new_bytes: int = 256,
        mad_multiplier: float = 3.0,
        k: int = 16,
        min_support: Optional[int] = None,
        byte_entropy_threshold: Optional[float] = None,
        tree=None,
        logger=None,
    ):
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents to fuse.")
        self.agents = agents
        self.thresholds = thresholds
        # theta_H is a token-space number and does not transfer. Entropy over 256
        # bytes tops out at ln(256)=5.55, against ln(151000)=11.9 for a modern
        # vocabulary, and measured byte-space root entropy sits in a narrow band:
        # near 0 when the first byte is near-deterministic (very common, since
        # byte-level tokenizers put a leading space on most word-initial tokens)
        # and 2.3-2.7 nats otherwise. In neither regime does theta_H=2.0
        # discriminate -- it behaves as a constant. Divergence carries the
        # routing decision in practice.
        #
        # It therefore gets its own key rather than silently inheriting
        # config_sheaf.yaml's value, and the observed entropy is logged on every step so
        # it can be calibrated from steps.jsonl against real models.
        self.byte_entropy_threshold = (
            thresholds.entropy if byte_entropy_threshold is None else byte_entropy_threshold
        )
        self.weights = weights or [1.0 / len(agents)] * len(agents)
        self.max_new_bytes = max_new_bytes
        self.logger = logger

        # Two pipelines over the same agents: Path A never escalates, Path B may.
        # Building both up front keeps `step` free of branching on construction.
        # A prebuilt tree turns this into `prebuilt` mode: the trie is loaded from
        # disk once and every step propagates only its top-k tokens through the
        # precomputed paths. Without one, the trie is rebuilt per step.
        common = dict(
            mode="prebuilt" if tree is not None else "topk_union",
            tree=tree,
            k=k,
            weights=np.asarray(self.weights, dtype=np.float64),
            min_support=min_support,
        )
        self._path_a = Stage8Pipeline(agents, fusion="mean", **common)
        self._path_b = Stage8Pipeline(agents, fusion="auto", **common)
        self._path_b.reconciler.mad_multiplier = mad_multiplier
        self._eps = self._path_b.reconciler.eps

    # ------------------------------------------------------------------

    def _byte_level_gate(self, probs: List[np.ndarray]):
        """Stage 2, applied in the shared byte space rather than in token space.

        Same two-factor criterion and same `GateThresholds` as the fast version:

          H  entropy of the mean consensus distribution over the next byte,
             computed by `sahf.gate.mean_entropy` on the agents' root sections;
          D  max pairwise amplitude spread, but taken over EVERY active node
             rather than the root only.

        The per-node maximum is not a refinement, it is required for correctness.
        Byte-level tokenizers put a leading space on most word-initial tokens, so
        " Paris" and " Berlin" share their whole root section: two agents in total
        disagreement look unanimous at depth 0. Gating on the root would send
        nearly every step down Path A and Stage 6/7 would never fire.
        """
        import torch

        tree = self._path_a.build_tree(probs)
        (covers, terms), candidates = self._path_a.propagate(tree, probs)

        root_sections, D, stalks = byte_level_sections(
            tree, covers, terms, eps=self._eps, candidates=candidates
        )
        H = mean_entropy([torch.from_numpy(to_amplitude(r)) for r in root_sections])

        agree = (H < self.byte_entropy_threshold) and (D < self.thresholds.divergence)
        # The tree, the propagated mass and the per-node stalks are handed back so
        # `step` can reuse them. Recomputing all three cost ~20-25% of every
        # decoding step, the bulk of it the per-node section pass.
        return (
            GateDecision(agree=agree, entropy=H, divergence=D),
            tree,
            (covers, terms),
            stalks,
            candidates,
        )

    # ------------------------------------------------------------------

    def step(self, context: str, step_idx: int) -> tuple:
        """One reconciled decoding step. Returns `(consensus_bytes, record)`."""
        t0 = time.perf_counter()
        probs = [agent.next_token_probs(context) for agent in self.agents]

        decision, tree, mass, stalks, candidates = self._byte_level_gate(probs)
        record = {
            "step": step_idx,
            "entropy": decision.entropy,
            "divergence": decision.divergence,
        }

        pipe = self._path_a if decision.agree else self._path_b
        res = pipe.step_from_probs(
            probs, context=context, tree=tree, mass=mass, stalks=stalks, candidates=candidates
        )

        record.update(
            {
                "path": "A_fast_passthrough" if decision.agree else "B_fusion_pipeline",
                "escalated": res.n_escalated > 0,
                "consensus_bytes": res.consensus_bytes.decode("utf-8", errors="replace"),
                "n_bytes": len(res.consensus_bytes),
                # Stage 8 specifics
                "tree_nodes": res.tree_stats.n_nodes,
                "fused_nodes": res.n_fused_nodes,
                "escalated_nodes": res.n_escalated,
                "coverage": res.coverage,
                "unreachable": res.unreachable,
                "stop_mass": res.stop_mass,
                "pruned_mass": res.pruned_mass,
                "elapsed_ms": (time.perf_counter() - t0) * 1e3,
            }
        )
        if self.logger:
            self.logger.log_step(record)
        return res.consensus_bytes, record, res

    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_bytes: Optional[int] = None) -> tuple:
        """Byte-level consensus generation. Returns `(text, history)`.

        Returns text, not `(output_ids, history)`: with mismatched tokenizers there
        is no shared id sequence to return.

        The running text is assembled with an INCREMENTAL utf-8 decoder. Consensus
        chunks are byte strings and routinely end part-way through a multi-byte
        character; decoding the buffer outright each step would substitute U+FFFD
        and feed corrupted context back to the agents, breaking every non-ASCII
        language.
        """
        budget = self.max_new_bytes if max_new_bytes is None else max_new_bytes
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = decoder.decode(prompt.encode("utf-8"))
        history = []
        emitted = 0

        step_idx = 0
        while emitted < budget:
            chunk, record, res = self.step(text, step_idx)
            history.append(record)
            if res.should_stop:
                record["stopped"] = "eos_consensus"
                break
            if not chunk:
                record["stopped"] = "no_consensus"
                break
            emitted += len(chunk)
            text += decoder.decode(chunk)
            step_idx += 1

        return text, history
