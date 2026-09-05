"""Entity-graph propagation (Lemma 1, Proposition 1).

Two edge sets over the candidate pool of a whole batch:

``E_=``  *identity* edges join candidates whose type-normalised surface forms
         are equal -- case, separator and whitespace folding, plus
         checksum-class agreement for verifiable types.
``E_~``  *similarity* edges join candidates whose documents collide under the
         MinHash/LSH index already maintained for deduplication and whose
         +/-64-character context windows exceed a cosine threshold.

Propagation is two-channel and deliberately asymmetric:

    s~=_v  = max(s_v, beta1 * max_{u in C(v)} s_u)          damped maximum
    s~~    = (1 - beta2) (I - beta2 A~)^{-1} s              damped diffusion
    s~_v   = max(s~=_v, s~~_v)

Exact value identity is near-conclusive evidence and deserves a max; context
similarity is weak evidence and deserves a dilution.

**Why this is certificate-safe.** Both channels are functions of the
*unordered multiset* of candidates: connected components ignore enumeration
order, and relabelling documents permutes the rows and columns of ``A~``
simultaneously. The composite map is therefore permutation-equivariant, so it
maps an exchangeable candidate sequence to an exchangeable score sequence
(Lemma 1) and the downstream conformal guarantee applies to propagated scores
unchanged. ``propagate`` must consequently be run *transductively* over the
pooled calibration + deployment batch -- never fitted on calibration and then
applied to deployment.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from ..taxonomy import CHECKSUMMED_TYPES
from ..types import Document, Span
from .checks import checksum_class  # noqa: F401  (re-exported for tests)
from .minhash import MinHashIndex

CONTEXT_WIDTH = 64
_SEPARATORS = re.compile(r"[\s\-_./()]+")
_TOKEN = re.compile(r"[a-z0-9]+")

#: Role words and honorifics a weak PERSON heuristic can drag into a span.
#: Stripping them before keying identity edges is entity resolution, not a
#: hack: "Cosigner Yuki Tanaka" and "Yuki Tanaka" are one entity, and merging
#: them is what gives Prop. 1 the multiplicity it needs.
_PERSON_ROLE_TOKENS = frozenset(
    """mr mrs ms dr prof borrower applicant client customer beneficiary guarantor
    holder contact cosigner co signer signer payee payer remitter patient
    physician account compliance officer manager representative""".split()
)


@dataclass(slots=True)
class PropagationConfig:
    """Layer I hyperparameters. Defaults are the paper's."""

    beta1: float = 0.95          # identity-channel damping
    beta2: float = 0.60          # similarity-channel damping
    power_iterations: int = 3    # truncation of the resolvent series
    context_width: int = CONTEXT_WIDTH
    context_cosine: float = 0.35
    num_perm: int = 128
    bands: int = 32
    rows: int = 4
    lsh_threshold: float = 0.60  # for *propagation* edges, looser than dedup
    max_similarity_degree: int = 32

    def validate(self) -> None:
        if not 0.0 <= self.beta1 <= 1.0:
            raise ValueError("beta1 must lie in [0, 1]")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must lie in [0, 1) for the resolvent to converge")
        if self.power_iterations < 1:
            raise ValueError("power_iterations must be >= 1")


def normalise_value(text: str, typ: str) -> str:
    """Type-normalised surface form used as the identity-edge key.

    Folding case, separators and whitespace makes ``DE28 7682 6991`` and
    ``de2876826991`` the same entity; for checksummed types the checksum class
    is appended so two different valid cards never merge on shape alone.
    """
    folded = _SEPARATORS.sub("", text.strip().lower())
    if typ in CHECKSUMMED_TYPES:
        return f"{folded}|{checksum_class(text, typ)}"
    if typ == "PERSON":
        # order-insensitive token set with role words stripped, so
        # "Riley, Melanie", "Melanie Riley" and "Borrower Melanie Riley" are
        # one entity
        tokens = sorted(
            t for t in _TOKEN.findall(text.lower())
            if len(t) > 1 and t not in _PERSON_ROLE_TOKENS
        )
        return "|".join(tokens) if tokens else folded
    return folded


def _context_vector(text: str, span: Span, width: int) -> dict[str, float]:
    """L2-normalised bag of context tokens around a span."""
    left = text[max(0, span.start - width) : span.start]
    right = text[span.end : min(len(text), span.end + width)]
    counts: dict[str, float] = defaultdict(float)
    for token in _TOKEN.findall(f"{left} {right}".lower()):
        if len(token) > 1:
            counts[token] += 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


@dataclass(slots=True)
class EntityGraph:
    """The propagation graph over a batch's candidate pool.

    ``nodes`` is the flat candidate list; ``components`` maps a normalised
    value to the node indices carrying it (the ``E_=`` components); ``affinity``
    is the sparse row-normalised similarity matrix ``A~`` as adjacency lists.
    """

    nodes: list[Span]
    doc_of: list[int]
    components: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    affinity: list[list[tuple[int, float]]] = field(default_factory=list)
    n_identity_edges: int = 0
    n_similarity_edges: int = 0

    def __len__(self) -> int:
        return len(self.nodes)

    def component_of(self, index: int) -> list[int]:
        key = (self.nodes[index].type, normalise_value(self.nodes[index].text, self.nodes[index].type))
        return self.components.get(key, [index])

    def stats(self) -> dict[str, float]:
        sizes = [len(v) for v in self.components.values()]
        return {
            "n_nodes": float(len(self.nodes)),
            "n_components": float(len(self.components)),
            "n_identity_edges": float(self.n_identity_edges),
            "n_similarity_edges": float(self.n_similarity_edges),
            "max_component": float(max(sizes) if sizes else 0),
            "mean_component": float(sum(sizes) / len(sizes)) if sizes else 0.0,
            "epsilon_G": self.max_affinity_mass(),
        }

    def max_affinity_mass(self) -> float:
        """``max_v sum_u A~_vu`` -- the ``epsilon_G`` of Prop. 1's dual bound."""
        if not self.affinity:
            return 0.0
        return max((sum(w for _, w in row) for row in self.affinity), default=0.0)


def build_entity_graph(
    docs: Sequence[Document],
    candidates: Sequence[Sequence[Span]] | None = None,
    config: PropagationConfig | None = None,
    index: MinHashIndex | None = None,
) -> EntityGraph:
    """Build ``E_=`` and ``E_~`` over the pooled candidate multiset.

    ``docs`` must be the *pooled* calibration + deployment batch for Lemma 1 to
    apply. ``candidates[i]`` are the proposals for ``docs[i]``; when omitted,
    ``docs[i].candidates`` is used.
    """
    cfg = config or PropagationConfig()
    cfg.validate()

    pools = candidates if candidates is not None else [d.candidates for d in docs]
    nodes: list[Span] = []
    doc_of: list[int] = []
    for doc_i, pool in enumerate(pools):
        for span in pool:
            nodes.append(span)
            doc_of.append(doc_i)

    graph = EntityGraph(nodes=nodes, doc_of=doc_of, affinity=[[] for _ in nodes])
    if not nodes:
        return graph

    # ---- E_= : identity components ---------------------------------------
    components: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, span in enumerate(nodes):
        components[(span.type, normalise_value(span.text, span.type))].append(i)
    graph.components = dict(components)
    graph.n_identity_edges = sum(len(v) - 1 for v in components.values() if len(v) > 1)

    # ---- E_~ : LSH document collisions, gated by context cosine ----------
    lsh = index
    if lsh is None:
        lsh = MinHashIndex(
            num_perm=cfg.num_perm, bands=cfg.bands, rows=cfg.rows,
            threshold=cfg.lsh_threshold,
        ).build([(str(i), d.text) for i, d in enumerate(docs)])

    nodes_by_doc: dict[int, list[int]] = defaultdict(list)
    for i, doc_i in enumerate(doc_of):
        nodes_by_doc[doc_i].append(i)

    ctx_cache: dict[int, dict[str, float]] = {}

    def ctx(node_i: int) -> dict[str, float]:
        if node_i not in ctx_cache:
            ctx_cache[node_i] = _context_vector(
                docs[doc_of[node_i]].text, nodes[node_i], cfg.context_width
            )
        return ctx_cache[node_i]

    raw: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for doc_i in range(len(docs)):
        for other_key, _sim in lsh.neighbours(str(doc_i), cfg.lsh_threshold):
            doc_j = int(other_key)
            if doc_j <= doc_i:
                continue
            for i in nodes_by_doc.get(doc_i, ()):
                for j in nodes_by_doc.get(doc_j, ()):
                    if nodes[i].type != nodes[j].type:
                        continue
                    weight = _cosine(ctx(i), ctx(j))
                    if weight >= cfg.context_cosine:
                        raw[i].append((j, weight))
                        raw[j].append((i, weight))

    # cap degree (keeps epsilon_G bounded) then row-normalise -> A~
    for i, row in enumerate(raw):
        row.sort(key=lambda t: -t[1])
        row = row[: cfg.max_similarity_degree]
        total = sum(w for _, w in row)
        graph.affinity[i] = [(j, w / total) for j, w in row] if total > 0 else []
    graph.n_similarity_edges = sum(len(r) for r in graph.affinity) // 2
    return graph


def propagate(
    graph: EntityGraph,
    scores: Sequence[float] | None = None,
    config: PropagationConfig | None = None,
) -> list[float]:
    """Run both channels and return the propagated scores ``s~``.

    Permutation-equivariant by construction (Lemma 1): the identity channel is
    a max over an unordered component and the similarity channel is a
    symmetric-support linear operator whose rows are indexed by the same
    permutation as its inputs.
    """
    cfg = config or PropagationConfig()
    cfg.validate()
    s = list(scores) if scores is not None else [n.score for n in graph.nodes]
    if not s:
        return []

    # ---- identity channel: damped maximum over the component -------------
    identity = list(s)
    for indices in graph.components.values():
        if len(indices) < 2:
            continue
        best = max(s[i] for i in indices)
        damped = cfg.beta1 * best
        for i in indices:
            if damped > identity[i]:
                identity[i] = damped

    # ---- similarity channel: truncated resolvent (personalised PageRank) --
    # (1 - b2) sum_{t>=0} b2^t A~^t s, truncated at power_iterations terms.
    diffusion = [(1.0 - cfg.beta2) * v for v in s]
    current = list(s)
    coeff = 1.0
    for _ in range(cfg.power_iterations):
        nxt = [0.0] * len(s)
        for i, row in enumerate(graph.affinity):
            if row:
                nxt[i] = sum(w * current[j] for j, w in row)
        coeff *= cfg.beta2
        for i, v in enumerate(nxt):
            diffusion[i] += (1.0 - cfg.beta2) * coeff * v
        current = nxt

    return [min(1.0, max(identity[i], diffusion[i])) for i in range(len(s))]


def apply_propagated_scores(
    docs: Sequence[Document],
    graph: EntityGraph,
    propagated: Sequence[float],
) -> None:
    """Write ``s~`` back onto the candidate spans in place."""
    for node, score in zip(graph.nodes, propagated):
        node.score = score
    _ = docs  # candidates are shared objects; kept for call-site symmetry


def recall_amplification(q: float, m: int) -> float:
    """Prop. 1's envelope ``1 - (1 - q)**m`` for an entity of multiplicity m."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    return 1.0 - (1.0 - q) ** max(1, m)


def cross_entity_leak_bound(beta2: float, epsilon_g: float, max_score: float = 1.0) -> float:
    """Prop. 1's dual bound ``beta2*eps_G/(1-beta2) * max_u s_u``."""
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must lie in [0, 1)")
    return beta2 * epsilon_g / (1.0 - beta2) * max_score
