"""
Stage 8 / part 2 -- the byte-prefix tree (shared base space).

One trie is built over the UNION of every agent's vocabulary, expressed in raw
bytes. A node is a byte prefix s. Two facts make it the right object:

  * agent A's token "  the" and agent B's tokens " th" + "e" land on the same
    path, so distributions that were incomparable at token level become
    comparable at every node;
  * the node set is the site the sheaf is defined over -- local sections live on
    nodes, restriction maps are the parent->child edges.

Two masses per node, per agent:

    cover(s) = sum of p(v) over tokens whose bytes START WITH s     (P[next token begins with s])
    term(s)  = sum of p(v) over tokens whose bytes EQUAL s          (P[next token IS exactly s])

with the identity that makes the conditionals proper distributions:

    cover(s) = term(s) + sum_b cover(s + b)

The local section at s is a distribution over the 256 next-byte values, taken
over the continuing mass ``cover(s) - term(s)``. See `sheaf.py` for why the
boundary is marginalized out rather than fused as an extra symbol.

Cost model (matters for the cascading-cost principle in the deck): the trie is
built ONCE. Per decoding step only the mass propagation runs, and that is two
`np.bincount` calls over a precomputed flat path index -- no Python loop over
the vocabulary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .vocab import VocabSpec

BYTE_DIM = 256  # the base alphabet: one symbol per byte value


def _pack_token_bytes(token_bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten `list[bytes | None]` into (blob, offsets, present) for npz storage."""
    present = np.zeros(len(token_bytes), dtype=bool)
    offsets = np.zeros(len(token_bytes) + 1, dtype=np.int64)
    chunks = []
    total = 0
    for i, b in enumerate(token_bytes):
        if b is not None:
            present[i] = True
            chunks.append(b)
            total += len(b)
        offsets[i + 1] = total
    blob = np.frombuffer(b"".join(chunks), dtype=np.uint8) if chunks else np.empty(0, np.uint8)
    return blob, offsets, present


def _unpack_token_bytes(blob, offsets, present) -> list:
    raw = blob.tobytes()
    return [
        raw[offsets[i] : offsets[i + 1]] if present[i] else None for i in range(len(present))
    ]


def _concat_ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Concatenation of ``range(s, s+l)`` for each (s, l), without a Python loop."""
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = starts[0]
    if starts.size > 1:
        breaks = np.cumsum(lens)[:-1]
        out[breaks] = starts[1:] - (starts[:-1] + lens[:-1]) + 1
    return np.cumsum(out)


@dataclass
class TreeStats:
    n_nodes: int
    max_depth: int
    n_agents: int
    truncated: bool


class BytePrefixTree:
    """Union byte-prefix tree over N agent vocabularies."""

    def __init__(self) -> None:
        # node arrays
        self.children: list[dict[int, int]] = [{}]
        self.parent: list[int] = [-1]
        self.byte_in: list[int] = [-1]  # byte on the edge from parent
        self.depth: list[int] = [0]

        # per-agent path index (built at construction, reused every step)
        self._flat_nodes: list[np.ndarray] = []
        self._flat_tok: list[np.ndarray] = []
        self._end_node: list[np.ndarray] = []
        self._end_tok: list[np.ndarray] = []
        self._terminals: list[dict[int, list[int]]] = []  # agent -> node -> token ids
        self._tok_index: list[np.ndarray] = []  # agent -> token id -> row in slice table
        self._tok_start: list[np.ndarray] = []  # agent -> row -> offset into flat arrays
        self._tok_len: list[np.ndarray] = []  # agent -> row -> path length
        self._scratch: dict[int, tuple] = {}  # agent -> reusable (cover, term, touched)
        self._touched: dict[int, np.ndarray | None] = {}
        self._glue_cover: np.ndarray | None = None
        self._glue_mask: np.ndarray | None = None
        self._glue_dirty: np.ndarray | None = None
        self._csr_offsets: np.ndarray | None = None
        self._csr_bytes: np.ndarray | None = None
        self._csr_nodes: np.ndarray | None = None

        self.vocabs: list[VocabSpec] = []
        self.max_depth_seen = 0
        self.truncated = False

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_vocabs(
        cls,
        vocabs: Sequence[VocabSpec],
        *,
        restrict: Sequence[Iterable[int]] | None = None,
        max_depth: int | None = None,
        max_nodes: int = 4_000_000,
    ) -> BytePrefixTree:
        """Build the union trie.

        Parameters
        ----------
        vocabs   : one VocabSpec per agent
        restrict : optional per-agent iterable of token ids to include. This is
                   the top-k-union mode -- pass the ids Stage 2 already gathered
                   and the tree costs microseconds instead of megabytes.
        max_depth: truncate tokens longer than this many bytes. Truncation keeps
                   cover mass correct but makes boundary mass unreliable below
                   the cut, so pair it with a decode depth <= max_depth.
        """
        self = cls()
        self.vocabs = list(vocabs)

        for a, vocab in enumerate(vocabs):
            # dict.fromkeys de-duplicates while preserving order. A repeated id
            # would otherwise be walked twice and have its probability counted
            # twice, silently breaking normalisation -- and callers assembling
            # `restrict` from several top-k lists produce repeats routinely.
            if restrict is not None:
                ids = list(dict.fromkeys(int(i) for i in restrict[a]))
            else:
                ids = vocab.byte_ids()

            flat_nodes: list[int] = []
            flat_tok: list[int] = []
            end_node: list[int] = []
            end_tok: list[int] = []
            terminals: dict[int, list[int]] = {}
            # Slice table: where each token's root-to-leaf path lives inside the
            # flat arrays. This is what lets a PREBUILT full-vocabulary tree
            # propagate only the top-k tokens per step instead of all of them --
            # without it, a persisted tree costs seconds per step and is useless
            # in a decode loop.
            tok_start: list[int] = []
            tok_len: list[int] = []

            for tid in ids:
                if tid >= len(vocab.token_bytes):
                    continue
                bs = vocab.token_bytes[tid]
                if not bs:
                    continue
                if max_depth is not None and len(bs) > max_depth:
                    bs = bs[:max_depth]
                    self.truncated = True

                node = 0
                tok_start.append(len(flat_nodes))
                flat_nodes.append(0)
                flat_tok.append(tid)
                for b in bs:
                    nxt = self.children[node].get(b)
                    if nxt is None:
                        if len(self.children) >= max_nodes:
                            raise MemoryError(
                                f"byte-prefix tree exceeded max_nodes={max_nodes}; "
                                "use restrict=<top-k ids> or lower max_depth"
                            )
                        nxt = len(self.children)
                        self.children.append({})
                        self.parent.append(node)
                        self.byte_in.append(b)
                        self.depth.append(self.depth[node] + 1)
                        self.children[node][b] = nxt
                    node = nxt
                    flat_nodes.append(node)
                    flat_tok.append(tid)

                self.max_depth_seen = max(self.max_depth_seen, self.depth[node])
                tok_len.append(len(flat_nodes) - tok_start[-1])
                end_node.append(node)
                end_tok.append(tid)
                terminals.setdefault(node, []).append(tid)

            self._flat_nodes.append(np.asarray(flat_nodes, dtype=np.int64))
            self._flat_tok.append(np.asarray(flat_tok, dtype=np.int64))
            self._end_node.append(np.asarray(end_node, dtype=np.int64))
            self._end_tok.append(np.asarray(end_tok, dtype=np.int64))
            self._terminals.append(terminals)

            et = np.asarray(end_tok, dtype=np.int64)
            index = np.full(max(len(vocab.token_bytes), 1), -1, dtype=np.int64)
            if et.size:
                index[et] = np.arange(et.size)
            self._tok_index.append(index)
            self._tok_start.append(np.asarray(tok_start, dtype=np.int64))
            self._tok_len.append(np.asarray(tok_len, dtype=np.int64))

        return self

    # ------------------------------------------------------------------
    # basic accessors
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.children)

    @property
    def n_agents(self) -> int:
        return len(self.vocabs)

    def stats(self) -> TreeStats:
        return TreeStats(self.n_nodes, self.max_depth_seen, self.n_agents, self.truncated)

    def node_bytes(self, node: int) -> bytes:
        """Reconstruct the byte prefix a node stands for."""
        out = bytearray()
        while node > 0:
            out.append(self.byte_in[node])
            node = self.parent[node]
        out.reverse()
        return bytes(out)

    def find(self, prefix: bytes) -> int:
        """Node index for a byte prefix, or -1 if absent."""
        node = 0
        for b in prefix:
            node = self.children[node].get(b, -1)
            if node < 0:
                return -1
        return node

    def terminals_at(self, node: int, agent: int) -> list[int]:
        """Token ids of `agent` whose bytes end exactly at `node`."""
        return self._terminals[agent].get(node, [])

    # ------------------------------------------------------------------
    # mass propagation -- the per-step hot path
    # ------------------------------------------------------------------

    def cover_mass(
        self, agent: int, probs: np.ndarray, restrict: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Push one agent's token distribution onto the tree.

        Returns ``(cover, term)``, both float64 arrays of length ``n_nodes``.

        ``restrict`` gives the token ids to propagate -- the top-k summary Stage 2
        already computes. This is the mode a PREBUILT tree runs in: only those
        tokens' precomputed root-to-leaf paths are touched, so the cost is
        O(k x depth) and does not grow with the size of the tree. Propagating a
        full 150k-token vocabulary through a million-node tree every step instead
        costs seconds.

        .. warning::
           With ``restrict``, the returned arrays are REUSED SCRATCH BUFFERS owned
           by the tree, so that a step does not allocate one array per agent per
           call. They stay valid until the next ``cover_mass`` call for the same
           agent. Anything that must outlive the current decoding step has to copy
           them. `GlobalSection` only reads them inside the step that produced it.
        """
        probs = np.asarray(probs, dtype=np.float64)
        expected = len(self.vocabs[agent].token_bytes)
        if probs.ndim != 1 or probs.shape[0] != expected:
            raise ValueError(
                f"agent {agent} ({self.vocabs[agent].name}): expected a 1-D distribution of "
                f"length {expected} to match its vocabulary, got shape {probs.shape}. "
                "A mis-sized vector silently mis-indexes tokens rather than failing."
            )
        n = self.n_nodes
        fn, ft = self._flat_nodes[agent], self._flat_tok[agent]
        en, et = self._end_node[agent], self._end_tok[agent]

        if restrict is None:
            cover = np.bincount(fn, weights=probs[ft], minlength=n) if fn.size else np.zeros(n)
            term = np.bincount(en, weights=probs[et], minlength=n) if en.size else np.zeros(n)
            self._touched[agent] = None
            return cover, term

        cover, term, previous = self._scratch_for(agent)
        if previous is not None and previous.size:
            cover[previous] = 0.0
            term[previous] = 0.0

        index = self._tok_index[agent]
        ids = np.unique(np.asarray(restrict, dtype=np.int64))
        ids = ids[(ids >= 0) & (ids < index.size)]
        rows = index[ids]
        rows = rows[rows >= 0]
        if rows.size == 0:
            self._touched[agent] = np.empty(0, dtype=np.int64)
            return cover, term

        starts, lens = self._tok_start[agent][rows], self._tok_len[agent][rows]
        gather = _concat_ranges(starts, lens)
        nodes, toks = fn[gather], ft[gather]

        uniq, inv = np.unique(nodes, return_inverse=True)
        cover[uniq] = np.bincount(inv, weights=probs[toks])

        end_nodes, end_toks = en[rows], et[rows]
        u2, i2 = np.unique(end_nodes, return_inverse=True)
        term[u2] = np.bincount(i2, weights=probs[end_toks])

        self._touched[agent] = uniq
        return cover, term

    def _scratch_for(self, agent: int):
        prev = self._scratch.get(agent)
        if prev is None:
            n = self.n_nodes
            self._scratch[agent] = (np.zeros(n), np.zeros(n), None)
            return self._scratch[agent][0], self._scratch[agent][1], None
        cover, term, _ = prev
        touched = self._touched.get(agent)
        self._scratch[agent] = (cover, term, touched)
        return cover, term, touched

    def glue_scratch(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reusable ``(glued_cover, in_candidate)`` buffers, cleared for `nodes`.

        `reconcile` needs two arrays indexed by global node id. Allocating them
        per step is free for a 200-node per-step tree and is the single largest
        per-step cost for a prebuilt 200k-node one -- 1.6 MB zeroed twice per
        token. Only the candidate entries are ever written, so only those need
        clearing.

        Same lifetime rule as `cover_mass(restrict=...)`: valid until the next
        step for this tree.
        """
        if self._glue_cover is None:
            self._glue_cover = np.zeros(self.n_nodes, dtype=np.float64)
            self._glue_mask = np.zeros(self.n_nodes, dtype=bool)
        else:
            prev = self._glue_dirty
            if prev is not None and prev.size:
                self._glue_cover[prev] = 0.0
                self._glue_mask[prev] = False
        self._glue_dirty = nodes
        return self._glue_cover, self._glue_mask

    def child_arrays(self, node: int) -> tuple[np.ndarray, np.ndarray]:
        """``(bytes, child_nodes)`` for one node, as O(1) array views.

        Building these per node with `np.fromiter` over the children dict is fine
        for a small per-step tree, but a prebuilt full-vocabulary tree has nodes
        with hundreds of children and thousands of them are visited per step --
        materialising all of that in Python dominated the step. The CSR arrays are
        built once per tree and sliced.
        """
        if self._csr_offsets is None:
            self._build_csr()
        lo, hi = self._csr_offsets[node], self._csr_offsets[node + 1]
        return self._csr_bytes[lo:hi], self._csr_nodes[lo:hi]

    def _build_csr(self) -> None:
        offsets = np.zeros(self.n_nodes + 1, dtype=np.int64)
        bytes_, nodes_ = [], []
        for i, kids in enumerate(self.children):
            for b, c in kids.items():
                bytes_.append(b)
                nodes_.append(c)
            offsets[i + 1] = len(bytes_)
        self._csr_offsets = offsets
        self._csr_bytes = np.asarray(bytes_, dtype=np.int64)
        self._csr_nodes = np.asarray(nodes_, dtype=np.int64)

    def touched_nodes(self, agent: int) -> np.ndarray | None:
        """Nodes the last ``cover_mass(restrict=...)`` call reached, or None.

        `SheafReconciler` uses the union of these as its candidate set, so node
        selection also stays O(touched) instead of scanning the whole tree.
        """
        return self._touched.get(agent)

    def boundary_rate(self, node: int, cover: np.ndarray, term: np.ndarray) -> float:
        """P[this agent's token ends exactly here | it begins with this prefix].

        Purely a property of the agent's segmentation, not of its content
        prediction. `sheaf.SheafReconciler` deliberately keeps this OUT of the
        fused conditional and uses it only when projecting back to token space.
        """
        total = cover[node]
        return float(term[node] / total) if total > 0 else 0.0

    def active_nodes(
        self,
        covers: Sequence[np.ndarray],
        *,
        eps: float = 1e-6,
        max_depth: int | None = None,
        candidates: np.ndarray | None = None,
    ) -> np.ndarray:
        """Nodes worth fusing: any agent puts >= eps cover mass there.

        This is the pruning that keeps Stage 8 cheap -- the full trie may have
        millions of nodes but only a few hundred are live at any decoding step.
        """
        if candidates is not None:
            cand = np.unique(np.asarray(candidates, dtype=np.int64))
            if cand.size == 0:
                return cand
            vals = np.maximum.reduce([np.asarray(c)[cand] for c in covers])
            sel = cand[vals >= eps]
            if max_depth is not None:
                sel = sel[np.asarray(self.depth)[sel] <= max_depth]
            return sel

        stacked = np.maximum.reduce([np.asarray(c) for c in covers])
        mask = stacked >= eps
        if max_depth is not None:
            mask &= np.asarray(self.depth) <= max_depth
        return np.flatnonzero(mask)

    # ------------------------------------------------------------------
    # persistence -- build once, reuse for every run
    # ------------------------------------------------------------------

    FORMAT_VERSION = 1

    def save(self, path: str | Path) -> Path:
        """Serialize the whole prepared tree to a single ``.npz`` artifact.

        This exists because building the tree is a one-off job, not per-query
        work: for a real three-model ensemble it means loading three tokenizers,
        extracting and verifying ~500k token byte images, and walking every one
        of them into the trie. Doing that at the start of every run is wasted
        time, and doing it per decoding step is not viable at all.

        The artifact carries the vocabularies too, so a run never needs to load
        `transformers` or the tokenizers themselves just to reconcile bytes.

        `children` is stored in CSR form (offsets/bytes/nodes) rather than as
        dicts; `_terminals` is not stored at all since it is recoverable from the
        end-node arrays.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        offsets, edge_bytes, edge_nodes = [0], [], []
        for kids in self.children:
            for b, c in sorted(kids.items()):
                edge_bytes.append(b)
                edge_nodes.append(c)
            offsets.append(len(edge_bytes))

        arrays: dict[str, np.ndarray] = {
            "parent": np.asarray(self.parent, dtype=np.int64),
            "byte_in": np.asarray(self.byte_in, dtype=np.int16),
            "depth": np.asarray(self.depth, dtype=np.int32),
            "child_offsets": np.asarray(offsets, dtype=np.int64),
            "child_bytes": np.asarray(edge_bytes, dtype=np.int16),
            "child_nodes": np.asarray(edge_nodes, dtype=np.int64),
        }

        meta: dict = {
            "format_version": self.FORMAT_VERSION,
            "n_nodes": self.n_nodes,
            "max_depth_seen": self.max_depth_seen,
            "truncated": self.truncated,
            "agents": [],
        }
        for a, vocab in enumerate(self.vocabs):
            for key in ("flat_nodes", "flat_tok", "end_node", "end_tok",
                        "tok_index", "tok_start", "tok_len"):
                arrays[f"a{a}_{key}"] = getattr(self, f"_{key}")[a]
            blob, offs, present = _pack_token_bytes(vocab.token_bytes)
            arrays[f"a{a}_vocab_blob"] = blob
            arrays[f"a{a}_vocab_offsets"] = offs
            arrays[f"a{a}_vocab_present"] = present
            meta["agents"].append({
                "name": vocab.name,
                "scheme": vocab.scheme,
                "size": vocab.size,
                "special_ids": sorted(int(i) for i in vocab.special_ids),
                "prefix_space": bool(getattr(vocab, "prefix_space", False)),
            })

        arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BytePrefixTree:
        """Rebuild a tree saved by `save`. No tokenizers required."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            if meta.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError(
                    f"{path}: prefix tree format v{meta.get('format_version')} but this "
                    f"build expects v{cls.FORMAT_VERSION}. Rebuild it with "
                    "`python build_prefix_tree.py`."
                )
            self = cls()
            self.parent = z["parent"].tolist()
            self.byte_in = z["byte_in"].tolist()
            self.depth = z["depth"].tolist()
            self.max_depth_seen = int(meta["max_depth_seen"])
            self.truncated = bool(meta["truncated"])

            offsets = z["child_offsets"]
            cbytes, cnodes = z["child_bytes"], z["child_nodes"]
            self.children = [
                dict(
                    zip(
                        cbytes[offsets[i] : offsets[i + 1]].tolist(),
                        cnodes[offsets[i] : offsets[i + 1]].tolist(),
                    )
                )
                for i in range(len(offsets) - 1)
            ]

            for a, spec in enumerate(meta["agents"]):
                token_bytes = _unpack_token_bytes(
                    z[f"a{a}_vocab_blob"], z[f"a{a}_vocab_offsets"], z[f"a{a}_vocab_present"]
                )
                self.vocabs.append(
                    VocabSpec(
                        name=spec["name"],
                        token_bytes=token_bytes,
                        special_ids=set(spec["special_ids"]),
                        scheme=spec["scheme"],
                        prefix_space=spec["prefix_space"],
                    )
                )
                for key, target in (
                    ("flat_nodes", self._flat_nodes),
                    ("flat_tok", self._flat_tok),
                    ("end_node", self._end_node),
                    ("end_tok", self._end_tok),
                    ("tok_index", self._tok_index),
                    ("tok_start", self._tok_start),
                    ("tok_len", self._tok_len),
                ):
                    target.append(z[f"a{a}_{key}"])

                # terminals are derivable, so they are not stored
                terminals: dict[int, list[int]] = {}
                for node, tok in zip(
                    self._end_node[a].tolist(), self._end_tok[a].tolist()
                ):
                    terminals.setdefault(node, []).append(tok)
                self._terminals.append(terminals)
        return self
