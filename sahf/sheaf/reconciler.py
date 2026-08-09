"""
Stage 8 / part 3 -- local sections, gluing, and the trip back to token space.

The sheaf picture the deck describes, made concrete:

  base space      the byte-prefix tree's nodes (byte strings s)
  stalk at s      a distribution over the next BYTE, given the prefix s
  local section   one agent's conditional at s, defined only on the open set
                  where that agent still has mass to continue with
  restriction     parent -> child edge; restricting is conditioning one byte on
  gluing          agents agree on overlaps => one global section over the tree,
                  recovered by chaining conditionals down from the root

Token distributions over different vocabularies cannot be averaged -- they live
in different spaces. Conditionals at a shared byte node live in the SAME
256-simplex for every agent, so Stages 5/6/7 apply unchanged, per node.

WHY THE TOKEN BOUNDARY IS MARGINALIZED OUT
------------------------------------------
A node splits an agent's cover mass two ways: mass that continues to a further
byte, and mass that stops because one of its tokens ends exactly here. It is
tempting to fuse "stop" as a 257th symbol alongside the 256 bytes. That is
wrong, and wrong in a way that quietly corrupts the consensus.

Take agents predicting the same text " the". Agent A holds " the" as one token;
agent B holds " th" then "e". At node " th", A continues with 'e' while B stops.
Fusing "stop" as a symbol reports a 50/50 split -- a manufactured disagreement
about where a tokenizer draws its boundaries, not about what comes next. The two
agents in fact agree completely.

So a stop is read as a HORIZON, not a vote: B has simply run out of what one
forward pass can tell us. B leaves the support at deeper nodes and A carries the
prediction. Boundaries are tokenizer artifacts, and keeping them out of the
fused result is the entire point of Stage 8.

Boundaries return only where they belong -- in `project_to_vocab`, where the
consensus content is re-segmented by each agent's own tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .prefix_tree import BytePrefixTree

BYTE_DIM = 256

# --------------------------------------------------------------------------
# Amplitude-space primitives (mirror Stages 1 / 5 / 6 / 7)
# --------------------------------------------------------------------------


def to_amplitude(p: np.ndarray) -> np.ndarray:
    """Stage 1: psi = sqrt(p). Maps the simplex onto the unit sphere."""
    return np.sqrt(np.clip(p, 0.0, None))


def to_probability(psi: np.ndarray) -> np.ndarray:
    """Inverse of Stage 1 (Born rule): p = |psi|^2, renormalized."""
    p = np.square(psi)
    s = p.sum()
    return p / s if s > 0 else p


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def mean_fuse(psis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Stage 5: chordal mean -- weighted average re-projected onto the sphere."""
    return _normalize(np.tensordot(weights, psis, axes=(0, 0)))


def geometric_median(
    psis: np.ndarray,
    weights: np.ndarray,
    *,
    iters: int = 50,
    tol: float = 1e-6,
    eps: float = 1e-8,
) -> np.ndarray:
    """Stage 7: Weiszfeld on the sphere. Breakdown point 1/2.

    A numpy transcription of `sahf.robust.weiszfeld_geometric_median`, matching it
    step for step: the same UNWEIGHTED normalised mean as the starting point, the
    same inverse-distance reweighting, the same max_iter=50 / tol=1e-6 / eps=1e-8.

    It is transcribed rather than called because Stage 8 runs this per tree node,
    hundreds of times per token, on 256-dim vectors -- torch's per-call overhead
    would dominate. `tests/test_sheaf_parity.py` asserts the two agree to 1e-6 on
    random inputs, so any drift in the repo's Stage 7 fails the suite rather than
    silently giving Stage 8 different semantics.
    """
    psi = _normalize(psis.mean(axis=0))
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(psi[None, :] - psis, axis=1), eps)
        inv = weights / d
        nxt = _normalize(np.tensordot(inv, psis, axes=(0, 0)) / max(inv.sum(), eps))
        shift = float(np.linalg.norm(nxt - psi))
        psi = nxt
        if shift < tol:
            break
    return psi


def hellinger(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Hellinger distance, read off amplitudes as plain Euclidean distance."""
    return float(np.linalg.norm(psi_a - psi_b) / np.sqrt(2.0))


def _torch_median(x: np.ndarray) -> float:
    """`torch.median`'s definition, which is NOT `np.median`'s.

    For an even number of elements torch returns the LOWER of the two central
    values; numpy averages them. `sahf.robust.detect_outliers` calls
    `torch.median`, so a numpy transcription that used `np.median` would flag
    different agents as outliers whenever the agent count is even.

    The repo's default config runs N=3, where the two agree, so this would have
    stayed invisible until someone added a fourth agent. Caught by
    tests/test_sheaf_parity.py.
    """
    return float(np.sort(x)[(x.size - 1) // 2])


def detect_outliers(
    psis: np.ndarray, psi_star: np.ndarray, mad_multiplier: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 6: median-absolute-deviation screen on distance-to-centre.

    A numpy transcription of `sahf.robust.detect_outliers`, including its MAD floor
    of 1e-8 and its threshold `median + mad_multiplier * MAD`.

    An earlier standalone version of Stage 8 used a modified z-score
    (0.6745*(d-med)/MAD > 3.5), which is a materially different rule -- roughly
    `median + 5.19*MAD`. Stage 8 must escalate on exactly the same condition as
    the rest of the pipeline, so the repo's rule wins and the parity test pins it.

    Returns ``(outlier_mask, distances)``.
    """
    d = np.linalg.norm(psis - psi_star[None, :], axis=1)
    med = _torch_median(d)
    mad = max(_torch_median(np.abs(d - med)), 1e-8)
    return d > (med + mad_multiplier * mad), d


def mad_outlier(psis: np.ndarray, psi_star: np.ndarray, mad_multiplier: float = 3.0) -> bool:
    """True when Stage 6 flags any agent -- the escalation trigger for Stage 7."""
    if psis.shape[0] < 3:
        return False
    mask, _ = detect_outliers(psis, psi_star, mad_multiplier)
    return bool(mask.any())


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class NodeReport:
    node: int
    prefix: bytes
    support: list[int]  # agents that can still continue past this prefix
    disagreement: float  # max pairwise Hellinger between local sections
    escalated: bool  # Stage 6 flagged an outlier -> Stage 7 used
    horizon: list[int] = field(default_factory=list)  # agents whose tokens end here


@dataclass
class ProjectionReport:
    """Diagnostics for one projection of the glued section back to token space.

    unreachable : glued probability mass flowing into subtrees where this agent
                  has no token boundary at all -- the honest "this vocabulary
                  cannot express that" figure.
    mass_ratio  : pre-normalisation total. 1.0 means the glued section agreed
                  with this agent's own cover mass; further from 1 means the
                  projection had to reweight this agent harder to reach the
                  consensus. Not a loss measure, and legitimately >1.
    """

    unreachable: float
    mass_ratio: float


@dataclass
class GlobalSection:
    """The glued global section: one measure over byte strings."""

    tree: BytePrefixTree
    nodes: np.ndarray  # active node indices, depth-sorted
    cover: np.ndarray  # glued cover mass per node
    conditionals: dict[int, np.ndarray] = field(default_factory=dict)  # node -> 256 dims
    reports: dict[int, NodeReport] = field(default_factory=dict)
    agent_cover: list[np.ndarray] = field(default_factory=list)
    agent_term: list[np.ndarray] = field(default_factory=list)
    pruned_mass: float = 0.0

    # ---- reading the result --------------------------------------------

    def decode(
        self, *, beam: int = 8, max_len: int = 16, min_support: int = 2
    ) -> list[tuple[bytes, float]]:
        """Beam search over the glued byte measure.

        A path stops when fewer than `min_support` agents can still continue --
        the joint knowledge horizon of the ensemble, not any single tokenizer's
        boundary. Returns ``(bytes, probability)``, best first.
        """
        finished: list[tuple[bytes, float]] = []
        frontier: list[tuple[int, float, bytes]] = [(0, 1.0, b"")]

        for _ in range(max_len):
            nxt: list[tuple[int, float, bytes]] = []
            for node, acc, path in frontier:
                cond = self.conditionals.get(node)
                rep = self.reports.get(node)
                if cond is None or rep is None or len(rep.support) < min_support:
                    if path:
                        finished.append((path, acc))
                    continue
                for b, child in self.tree.children[node].items():
                    p = float(cond[b])
                    if p > 0:
                        nxt.append((child, acc * p, path + bytes([b])))
            if not nxt:
                break
            nxt.sort(key=lambda t: -t[1])
            frontier = nxt[:beam]

        for _node, acc, path in frontier:
            if path:
                finished.append((path, acc))

        merged: dict[bytes, float] = {}
        for path, p in finished:
            merged[path] = max(merged.get(path, 0.0), p)
        # Ties are broken by byte string, not by insertion order. Exact ties do
        # occur, and without this the winner depends on the order the tree happens
        # to iterate a node's children -- so a prebuilt tree and a per-step tree
        # could disagree on the same input despite identical probabilities.
        return sorted(merged.items(), key=lambda t: (-t[1], t[0]))

    def greedy_bytes(self, *, max_len: int = 16, min_support: int = 2) -> bytes:
        """Cheapest read-off: follow the argmax byte while support holds."""
        node, out = 0, bytearray()
        for _ in range(max_len):
            cond = self.conditionals.get(node)
            rep = self.reports.get(node)
            if cond is None or rep is None or len(rep.support) < min_support:
                break
            b = int(np.argmax(cond))
            child = self.tree.children[node].get(b)
            if child is None:
                break
            out.append(b)
            node = child
        return bytes(out)

    def project_to_vocab(self, agent: int) -> tuple[np.ndarray, ProjectionReport]:
        """Translate the glued section back into agent `agent`'s token space.

        The deck's "translating back for the next generation step". Content comes
        from the consensus; SEGMENTATION comes from this agent's own tokenizer,
        via its local boundary rate term(s)/cover(s):

            q(v)  ~  glued_cover(bytes(v)) * P_agent[token ends at bytes(v) | prefix]

        Returns ``(probs_over_vocab, report)``.

        Note the pre-normalisation total is NOT a loss measure. It equals
        ``sum_s term_a(s) * glued_cover(s)/cover_a(s)`` -- an importance
        reweighting of the agent's own distribution, which sums to exactly 1 only
        when the consensus happens to match that agent. For a vocabulary with
        boundaries at many depths it routinely exceeds 1, so
        ``report.mass_ratio`` records it as a drift diagnostic, and the genuine
        "this vocabulary cannot express it" quantity is computed separately as
        ``report.unreachable``.
        """
        tree = self.tree
        cov_a, term_a = self.agent_cover[agent], self.agent_term[agent]
        size = len(tree.vocabs[agent].token_bytes)
        q = np.zeros(size, dtype=np.float64)
        claimed = 0.0

        for node in self.nodes:
            node = int(node)
            g = float(self.cover[node])
            if g <= 0 or node == 0:
                continue
            ids = tree.terminals_at(node, agent)
            if not ids or cov_a[node] <= 0:
                continue
            mass = g * float(term_a[node] / cov_a[node])
            if mass <= 0:
                continue
            claimed += mass
            for tid in ids:
                q[tid] += mass / len(ids)

        s = q.sum()
        if s > 0:
            q /= s

        # ---- unreachable mass -------------------------------------------
        # Bottom-up: can a token of `agent` ever end at or below this node?
        active = {int(n) for n in self.nodes}
        has_below: dict[int, bool] = {}
        for node in reversed(self.nodes):
            node = int(node)
            has_below[node] = bool(tree.terminals_at(node, agent)) or any(
                has_below.get(c, False) for c in tree.children[node].values() if c in active
            )
        # Sum the MAXIMAL subtrees with no boundary: glued mass entering them can
        # never be assigned to any token of this vocabulary.
        unreachable = 0.0
        for node in self.nodes:
            node = int(node)
            if node == 0 or has_below.get(node, False):
                continue
            parent = tree.parent[node]
            if parent < 0 or has_below.get(parent, False):
                unreachable += float(self.cover[node])

        return q, ProjectionReport(unreachable=unreachable, mass_ratio=claimed)

    def disagreement_profile(self, k: int = 10) -> list[NodeReport]:
        """Nodes where agents diverge most -- where the vocab mismatch bites."""
        return sorted(self.reports.values(), key=lambda r: -r.disagreement)[:k]


# --------------------------------------------------------------------------
# The reconciler
# --------------------------------------------------------------------------


class SheafReconciler:
    """Glues N per-agent token distributions into one byte-level global section.

    Parameters
    ----------
    eps           : glued-cover floor for a node to be fused at all
    support_eps   : continuation-mass floor for an agent to have an opinion
    fusion        : "mean" (Stage 5 only), "geometric_median" (Stage 7 always),
                    or "auto" (Stage 5, escalating per node on a Stage 6 flag)
    mad_multiplier: Stage 6 MAD multiplier; mirrors config_sheaf.yaml gate.mad_multiplier
    max_depth     : cap on prefix length to fuse
    agree_tol     : Hellinger distance below which sections count as unanimous,
                    skipping the robust machinery (mean == median when all
                    sections coincide, so this changes no result)
    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        support_eps: float = 1e-9,
        fusion: str = "auto",
        mad_multiplier: float = 3.0,
        max_depth: int | None = None,
        agree_tol: float = 1e-9,
    ) -> None:
        if fusion not in ("mean", "geometric_median", "auto"):
            raise ValueError(f"unknown fusion mode {fusion!r}")
        self.eps = eps
        self.support_eps = support_eps
        self.fusion = fusion
        self.mad_multiplier = mad_multiplier
        self.max_depth = max_depth
        self.agree_tol = agree_tol

    def _stalks(
        self,
        tree: BytePrefixTree,
        node: int,
        covers: list[np.ndarray],
        terms: list[np.ndarray],
        child_arr: np.ndarray,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        # thin wrapper so the gate and reconcile share one implementation
        return node_stalks(covers, terms, node, child_arr, self.support_eps)

    def _stalks_impl(
        self,
        tree: BytePrefixTree,
        node: int,
        covers: list[np.ndarray],
        terms: list[np.ndarray],
        child_arr: np.ndarray,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """Every agent's local section at `node`, restricted to the node's children.

        A node has a handful of children, not 256, and the conditional is
        structurally zero everywhere else -- so all the fusion algebra runs on
        `len(children)` dimensions instead of 256. That is exact, and it is what
        makes the per-node Weiszfeld iteration affordable.

        Returns ``(sections[n_support, n_children], support, horizon)``.
        """
        rows: list[np.ndarray] = []
        support: list[int] = []
        horizon: list[int] = []
        for a in range(len(covers)):
            cont = float(covers[a][node] - terms[a][node])
            if cont > self.support_eps:
                row = covers[a][child_arr]
                s = row.sum()
                if s > 0:
                    rows.append(row / s)
                    support.append(a)
            if terms[a][node] > self.support_eps:
                horizon.append(a)
        sections = np.stack(rows) if rows else np.zeros((0, child_arr.size))
        return sections, support, horizon

    def reconcile(
        self,
        tree: BytePrefixTree,
        probs: list[np.ndarray],
        *,
        weights: np.ndarray | None = None,
        mass: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
        stalks: dict | None = None,
        candidates: np.ndarray | None = None,
    ) -> GlobalSection:
        """`mass` accepts `(covers, terms)` already computed for this tree.

        Callers that inspect the mass before deciding whether to reconcile --
        `SheafOrchestrator`'s Stage 2 gate does exactly that -- would otherwise
        pay for the whole propagation twice per decoding step.
        """
        n_agents = len(probs)
        if n_agents != tree.n_agents:
            raise ValueError(f"tree built for {tree.n_agents} agents, got {n_agents} distributions")
        w_global = (
            np.full(n_agents, 1.0 / n_agents)
            if weights is None
            else np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
        )

        if mass is not None:
            covers, terms = mass
        else:
            covers, terms = [], []
            for a in range(n_agents):
                c, t = tree.cover_mass(a, probs[a])
                covers.append(c)
                terms.append(t)

        active = tree.active_nodes(
            covers, eps=self.eps, max_depth=self.max_depth, candidates=candidates
        )
        cand = {int(n) for n in active}
        cand.add(0)
        nodes = np.array(sorted(cand, key=lambda n: (tree.depth[n], n)), dtype=np.int64)

        conditionals: dict[int, np.ndarray] = {}
        reports: dict[int, NodeReport] = {}
        g_cover, in_cand = tree.glue_scratch(nodes)
        g_cover[0] = 1.0
        pruned = 0.0
        in_cand[nodes] = True

        # depth-sorted single pass: fuse the stalk, then push glued mass to children
        for node in nodes:
            node = int(node)
            if node != 0 and g_cover[node] <= 0:
                continue
            kids = tree.children[node]
            if not kids:
                continue

            cached = stalks.get(node) if stalks is not None else None
            if cached is not None:
                sections, support, horizon, byte_arr, child_arr = cached
            else:
                byte_arr, child_arr = live_children(covers, *tree.child_arrays(node))
                sections, support, horizon = self._stalks(
                    tree, node, covers, terms, child_arr
                )
            if child_arr.size == 0 or not support:
                continue

            psis = to_amplitude(sections)
            escalated = False

            if len(support) == 1:
                # Only one agent can still see past this prefix. Nothing to fuse:
                # it is not consensus, it is the sole surviving section. Very
                # common deep in the tree, where shorter tokenizers have stopped.
                fused = sections[0]
                disagreement = 0.0
            else:
                # pairwise Hellinger in one shot (Euclidean distance on amplitudes)
                diff = psis[:, None, :] - psis[None, :, :]
                dmat = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)) / np.sqrt(2.0)
                disagreement = float(dmat.max())

                w = w_global[support]
                w = w / w.sum()

                if disagreement <= self.agree_tol:
                    # unanimous: mean, median and any robust estimator coincide
                    psi_star = mean_fuse(psis, w)
                else:
                    psi_star = mean_fuse(psis, w)
                    if self.fusion == "geometric_median" or (self.fusion == "auto" and mad_outlier(
                        psis, psi_star, self.mad_multiplier
                    )):
                        psi_star = geometric_median(psis, w)
                        escalated = True
                fused = to_probability(psi_star)  # over this node's children only
            cond = np.zeros(BYTE_DIM, dtype=np.float64)
            cond[byte_arr] = fused
            conditionals[node] = cond
            reports[node] = NodeReport(
                node=node,
                prefix=tree.node_bytes(node),
                support=support,
                disagreement=disagreement,
                escalated=escalated,
                horizon=horizon,
            )

            here = g_cover[node]
            flow = here * fused
            keep = in_cand[child_arr]
            if keep.any():
                g_cover[child_arr[keep]] += flow[keep]
            if not keep.all():
                pruned += float(flow[~keep].sum())

        return GlobalSection(
            tree=tree,
            nodes=nodes,
            cover=g_cover,
            conditionals=conditionals,
            reports=reports,
            agent_cover=covers,
            agent_term=terms,
            pruned_mass=float(pruned),
        )


# --------------------------------------------------------------------------
# Stage 2 support: divergence in the byte base space
# --------------------------------------------------------------------------


def live_children(
    covers: list[np.ndarray], byte_arr: np.ndarray, child_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop children that carry no mass for any agent.

    In a per-step `topk_union` tree this is nearly a no-op: the trie only contains
    the top-k tokens, so almost every child is live. In a PREBUILT full-vocabulary
    tree it is essential. There a node can have hundreds of children -- every
    continuation any of the three vocabularies ever allows -- while a single step
    touches a handful of them. Without this filter every stalk vector is padded
    out with structural zeros and the fusion algebra runs on ~40x more dimensions
    than carry information, which made a prebuilt tree SLOWER per step than
    rebuilding a small one from scratch.

    Semantically free: a child with zero cover for every agent receives zero fused
    probability either way.
    """
    total = covers[0][child_arr].copy()
    for c in covers[1:]:
        total += c[child_arr]
    keep = total > 0.0
    if keep.all():
        return byte_arr, child_arr
    return byte_arr[keep], child_arr[keep]


def node_stalks(
    covers: list[np.ndarray],
    terms: list[np.ndarray],
    node: int,
    child_arr: np.ndarray,
    support_eps: float = 1e-9,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Every agent's local section at one node, restricted to that node's children.

    The single implementation used by both `SheafReconciler` and the Stage 2 gate.
    They previously each had their own copy, so every decoding step computed the
    per-node sections twice -- about 20% of the step.

    Returns ``(sections[n_support, n_children], support, horizon)``.
    """
    rows: list[np.ndarray] = []
    support: list[int] = []
    horizon: list[int] = []
    for a in range(len(covers)):
        if float(covers[a][node] - terms[a][node]) > support_eps:
            row = covers[a][child_arr]
            total = row.sum()
            if total > 0:
                rows.append(row / total)
                support.append(a)
        if terms[a][node] > support_eps:
            horizon.append(a)
    sections = np.stack(rows) if rows else np.zeros((0, child_arr.size))
    return sections, support, horizon


def byte_level_sections(
    tree: BytePrefixTree,
    covers: list[np.ndarray],
    terms: list[np.ndarray],
    *,
    eps: float = 1e-6,
    support_eps: float = 1e-9,
    candidates: np.ndarray | None = None,
) -> tuple[list[np.ndarray], float, dict]:
    """Inputs for a byte-space Stage 2 gate.

    Returns ``(root_sections, max_pairwise_spread, stalk_cache)``. The cache maps
    node -> the stalks computed here, so `reconcile` can consume them via its
    ``stalks=`` argument instead of computing the same sections a second time.

    root_sections  each agent's distribution over the FIRST byte, a 256-vector on
                   a shared simplex -- suitable for `sahf.gate.mean_entropy`.
    max spread     the largest pairwise amplitude distance found at ANY node,
                   i.e. `sahf.gate.pairwise_amplitude_spread` evaluated per node
                   and maximised.

    Why the spread cannot be read off the root alone: byte-level tokenizers put a
    leading space on most word-initial tokens, so " Paris" and " Berlin" share
    their entire root section. Two agents in total disagreement look unanimous at
    depth 0. Gating on the root would route almost every step to Path A and
    Stage 6/7 would effectively never run. The disagreement lives one node down,
    so every active node has to be checked.

    Cost is one pass over the active nodes with no fusion algebra, which is a
    small fraction of a full reconcile -- the point of a gate.
    """
    n_agents = len(covers)
    root = []
    for a in range(n_agents):
        row = np.zeros(BYTE_DIM, dtype=np.float64)
        for byte, child in tree.children[0].items():
            row[byte] = covers[a][child]
        total = row.sum()
        root.append(row / total if total > 0 else row)

    max_spread = 0.0
    cache: dict[int, tuple[np.ndarray, list[int], list[int]]] = {}
    for node in tree.active_nodes(covers, eps=eps, candidates=candidates):
        node = int(node)
        kids = tree.children[node]
        if not kids:
            continue
        byte_arr, child_arr = live_children(covers, *tree.child_arrays(node))
        if child_arr.size == 0:
            continue
        stalk = node_stalks(covers, terms, node, child_arr, support_eps)
        # The cache carries the geometry it was computed against. Storing only the
        # sections would let reconcile pair them with a child list it re-derived
        # itself, and any divergence surfaces as a raw shape error deep in the
        # fusion loop rather than as anything diagnosable.
        cache[node] = (*stalk, byte_arr, child_arr)
        sections = stalk[0]
        if sections.shape[0] < 2:
            continue
        psis = to_amplitude(sections)
        diff = psis[:, None, :] - psis[None, :, :]
        spread = float(np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).max())
        max_spread = max(max_spread, spread)

    return root, max_spread, cache
