"""
Stage 8 / part 5 -- end-to-end pipeline.

    agents (own tokenizers)
        -> next-token distributions
        -> union byte-prefix tree            [prefix_tree]
        -> per-node local sections in sqrt-space
        -> Stage 5 mean, Stage 6 screen, Stage 7 escalation -- per node
        -> glued global section over bytes   [sheaf]
        -> consensus bytes  +  per-agent token distributions for the next step

Two modes, matching the deck's cascading-cost principle:

  "topk_union" (default)
      Build the tree per step over the union of each agent's top-k ids. Those
      ids are what Stage 2 already computes for its cheap summaries, so Stage 8
      rides on data the pipeline pays for anyway.

      `k` defaults to 16 on benchmark evidence (docs/BENCHMARK_RESULTS.md): with
      three real tokenizers, output quality is FLAT from k=8 to k=256 while cost
      grows 23x. The ceiling on a single step is the mean token length (~3.3
      bytes) -- past its own token boundary an agent has no information, so a
      wider k adds competitor tokens, not lookahead.

  "full"
      One static tree over the complete vocabularies, built once and reused for
      every step. No tail truncation.

      OFFLINE EVALUATION ONLY. On three real 32k tokenizers the union tree is
      207k nodes and a step costs 1.5-3.5 SECONDS. It cannot be used in a
      decoding loop.
"""

from __future__ import annotations

import codecs
import time
from dataclasses import dataclass, field

import numpy as np

from .prefix_tree import BytePrefixTree, TreeStats
from .reconciler import GlobalSection, SheafReconciler
from .adapters import Agent, top_k_ids


@dataclass
class Stage8Result:
    context: str
    section: GlobalSection
    consensus: list[tuple[bytes, float]]
    coverage: list[float]  # per agent: probability mass that reached the tree
    unreachable: dict[str, float]  # per agent: glued mass its vocab cannot express
    mass_ratio: dict[str, float]  # per agent: projection reweighting drift (1.0 = none)
    tree_stats: TreeStats
    n_fused_nodes: int
    n_escalated: int
    pruned_mass: float
    elapsed_ms: float
    token_probs: dict[str, np.ndarray] = field(default_factory=dict)
    stop_mass: dict[str, float] = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        """True when a majority of reporting agents put >50% on an end-of-text token.

        Stop tokens carry no bytes, so they cannot be represented in the shared
        base space at all. They are excluded from fusion and surfaced here
        instead -- otherwise the ensemble renormalises the stop away and can
        never terminate.
        """
        if not self.stop_mass:
            return False
        votes = sum(1 for v in self.stop_mass.values() if v > 0.5)
        return votes * 2 > len(self.stop_mass)

    @property
    def consensus_bytes(self) -> bytes:
        return self.consensus[0][0] if self.consensus else b""

    def summary(self) -> str:
        top = ", ".join(f"{b!r}:{p:.3f}" for b, p in self.consensus[:5])
        res = ", ".join(f"{k}={v:.2e}" for k, v in self.unreachable.items())
        return (
            f"nodes={self.tree_stats.n_nodes} fused={self.n_fused_nodes} "
            f"escalated={self.n_escalated} pruned={self.pruned_mass:.2e} "
            f"{self.elapsed_ms:.1f}ms\n  consensus: {top}\n  unreachable: {res}"
        )


class Stage8Pipeline:
    """Sheaf reconciliation across N agents with mismatched vocabularies."""

    def __init__(
        self,
        agents: list[Agent],
        *,
        mode: str = "topk_union",
        tree: BytePrefixTree | None = None,
        k: int = 16,
        max_depth: int | None = None,
        fusion: str = "auto",
        eps: float = 1e-6,
        weights: np.ndarray | None = None,
        project_back: bool = True,
        min_support: int | None = None,
    ) -> None:
        if mode not in ("topk_union", "full", "prebuilt"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "prebuilt" and tree is None:
            raise ValueError(
                'mode="prebuilt" needs a tree; build one once with '
                "`python build_prefix_tree.py` and load it with BytePrefixTree.load()"
            )
        if len(agents) < 2:
            raise ValueError("sheaf reconciliation needs at least 2 agents")
        self.agents = agents
        self.mode = mode
        self.k = k
        self.max_depth = max_depth
        self.weights = weights
        self.project_back = project_back
        # how many agents must still be able to continue for a byte to be emitted.
        # Default is a simple majority: the consensus chunk ends at the ensemble's
        # joint knowledge horizon, not at whichever tokenizer happens to stop first.
        self.min_support = (
            min_support if min_support is not None else max(1, (len(agents) + 1) // 2)
        )
        self.vocabs = [a.vocab_spec() for a in agents]
        self.reconciler = SheafReconciler(eps=eps, fusion=fusion, max_depth=max_depth)
        self._static_tree: BytePrefixTree | None = None
        if mode == "full":
            self._static_tree = BytePrefixTree.from_vocabs(self.vocabs, max_depth=max_depth)
        elif mode == "prebuilt":
            if tree.n_agents != len(agents):
                raise ValueError(
                    f"prefix tree was built for {tree.n_agents} agents but {len(agents)} "
                    "were supplied; rebuild it for this ensemble"
                )
            self._static_tree = tree
            self.vocabs = tree.vocabs  # the artifact is the source of truth

    # ------------------------------------------------------------------

    @property
    def static_tree(self) -> BytePrefixTree | None:
        """The prebuilt full-vocabulary tree (``mode="full"`` only)."""
        return self._static_tree

    def build_tree(self, probs: list[np.ndarray]) -> BytePrefixTree:
        """The tree for this step. Prebuilt/full modes return the one static tree."""
        if self._static_tree is not None:
            return self._static_tree
        restrict = [top_k_ids(p, self.k) for p in probs]
        return BytePrefixTree.from_vocabs(self.vocabs, restrict=restrict, max_depth=self.max_depth)

    # ------------------------------------------------------------------

    def step(
        self, context: str, *, beam: int = 8, max_len: int = 16, min_support: int | None = None
    ) -> Stage8Result:
        """Reconcile one decoding step across all agents."""
        t0 = time.perf_counter()
        probs = [a.next_token_probs(context) for a in self.agents]
        return self._reconcile(
            context, probs, t0, beam=beam, max_len=max_len, min_support=min_support
        )

    def step_from_probs(
        self,
        probs: list[np.ndarray],
        *,
        context: str = "",
        beam: int = 8,
        max_len: int = 16,
        min_support: int | None = None,
        tree: BytePrefixTree | None = None,
        mass: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
        stalks: dict | None = None,
        candidates: np.ndarray | None = None,
    ) -> Stage8Result:
        """Same, when the distributions were obtained elsewhere (Stage 4's allgather).

        `tree` / `mass` let a caller that already built the tree and propagated the
        mass hand them straight in, instead of paying for both a second time.
        """
        return self._reconcile(
            context,
            probs,
            time.perf_counter(),
            beam=beam,
            max_len=max_len,
            min_support=min_support,
            tree=tree,
            mass=mass,
            stalks=stalks,
            candidates=candidates,
        )

    def propagate(
        self, tree: BytePrefixTree, probs: list[np.ndarray]
    ) -> tuple[tuple[list[np.ndarray], list[np.ndarray]], np.ndarray | None]:
        """Push each agent's distribution onto `tree`; return `(mass, candidates)`.

        In `prebuilt` mode only the top-k tokens' precomputed paths are walked, so
        the per-step cost is set by k and not by the size of the tree. The touched
        nodes double as the candidate set, keeping node selection O(touched) as
        well -- otherwise every step would scan a million-node array.
        """
        covers, terms, touched = [], [], []
        use_restrict = self.mode == "prebuilt"
        for a, p in enumerate(probs):
            restrict = top_k_ids(p, self.k) if use_restrict else None
            c, t = tree.cover_mass(a, p, restrict=restrict)
            covers.append(c)
            terms.append(t)
            if use_restrict:
                touched.append(tree.touched_nodes(a))
        candidates = np.concatenate([t for t in touched if t is not None]) if touched else None
        return (covers, terms), candidates

    def _reconcile(
        self,
        context: str,
        probs: list[np.ndarray],
        t0: float,
        *,
        beam: int,
        max_len: int,
        min_support: int | None = None,
        tree: BytePrefixTree | None = None,
        mass: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
        stalks: dict | None = None,
        candidates: np.ndarray | None = None,
    ) -> Stage8Result:
        ms = self.min_support if min_support is None else min_support
        if tree is None:
            tree = self.build_tree(probs)
            mass = None  # mass, stalks and candidates belong to a tree; never mix
            stalks = None
            candidates = None
        if mass is None:
            mass, candidates = self.propagate(tree, probs)
        section = self.reconciler.reconcile(
            tree, probs, weights=self.weights, mass=mass, stalks=stalks, candidates=candidates
        )
        consensus = section.decode(beam=beam, max_len=max_len, min_support=ms)

        # root cover, read off the propagation reconcile already did. Calling
        # cover_mass again here doubled the per-step cost of the hot path.
        coverage = [float(section.agent_cover[a][0]) for a in range(len(self.agents))]

        unreachable: dict[str, float] = {}
        mass_ratio: dict[str, float] = {}
        token_probs: dict[str, np.ndarray] = {}
        if self.project_back:
            for a, agent in enumerate(self.agents):
                q, rep = section.project_to_vocab(a)
                token_probs[agent.name] = q
                unreachable[agent.name] = rep.unreachable
                mass_ratio[agent.name] = rep.mass_ratio

        stop_mass = {
            ag.name: float(ag.stop_probability(context))
            for ag in self.agents
            if hasattr(ag, "stop_probability")
        }

        return Stage8Result(
            context=context,
            section=section,
            consensus=consensus,
            coverage=coverage,
            unreachable=unreachable,
            mass_ratio=mass_ratio,
            stop_mass=stop_mass,
            tree_stats=tree.stats(),
            n_fused_nodes=len(section.conditionals),
            n_escalated=sum(1 for r in section.reports.values() if r.escalated),
            pruned_mass=section.pruned_mass,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
            token_probs=token_probs,
        )

    # ------------------------------------------------------------------

    def generate(
        self, prompt: str, *, max_steps: int = 16, stop: bytes | None = None
    ) -> tuple[str, list[Stage8Result]]:
        """Byte-level consensus generation.

        Each step emits the consensus byte chunk and re-tokenizes the extended
        context with every agent's own tokenizer. Re-encoding the full context
        each step is what keeps agents consistent when their token boundaries
        fall in different places; it is also the approximation noted in the
        README under "exactness".

        Chunks are byte strings and routinely end part-way through a multi-byte
        UTF-8 character, so the running text is built with an INCREMENTAL
        decoder. It buffers an incomplete trailing sequence until the following
        chunk completes it. Decoding the buffer outright each step instead
        substitutes U+FFFD for that partial tail and feeds the corrupted text
        back to the agents as context -- which breaks every non-ASCII language.
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = decoder.decode(prompt.encode("utf-8"))
        trace: list[Stage8Result] = []
        emitted_bytes = bytearray()

        for _ in range(max_steps):
            res = self.step(text)
            trace.append(res)
            if res.should_stop:
                break
            chunk = res.consensus_bytes
            if not chunk:
                break
            emitted_bytes.extend(chunk)
            text += decoder.decode(chunk)
            if stop and stop in bytes(emitted_bytes):
                break

        # No final flush. If the run ended part-way through a multi-byte
        # character, those bytes stay buffered rather than surfacing as U+FFFD;
        # the raw chunk is still on the last trace entry's `consensus_bytes` for
        # a caller that wants to resume.
        return text, trace
