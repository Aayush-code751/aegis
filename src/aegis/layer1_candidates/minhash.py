"""MinHash with LSH banding -- the index that does double duty.

The stream needs deduplication anyway; deduplication needs a near-duplicate
index; and that same index is the edge set of the propagation graph in
:mod:`aegis.layer1_candidates.propagation`. No second index is built, which is
the observation that turns stream redundancy from a storage cost into a recall
asset.

Implementation is dependency-free: ``k`` independent 64-bit hash permutations
over character shingles, banded into ``b`` bands of ``r`` rows with
``k = b * r``. Two documents of Jaccard similarity ``J`` collide in at least
one band with probability ``1 - (1 - J**r)**b`` -- the curve Prop. 1 uses to
bound the cross-entity edge mass.
"""
from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field

_WS = re.compile(r"\s+")
_MERSENNE = (1 << 61) - 1
_MAX64 = (1 << 64) - 1


def canonicalise(text: str) -> str:
    """Whitespace-folded, lowercased text -- the unit both dedup and LSH see."""
    return _WS.sub(" ", text.strip().lower())


def shingles(text: str, k: int = 5) -> set[bytes]:
    """Character k-shingles of the canonical form."""
    canon = canonicalise(text)
    if len(canon) < k:
        return {canon.encode()}
    data = canon.encode()
    return {data[i : i + k] for i in range(len(data) - k + 1)}


def content_hash(text: str) -> str:
    """SHA-256 of the canonical form: the exact-duplicate key and ledger key."""
    return hashlib.sha256(canonicalise(text).encode()).hexdigest()


def _permutations(num_perm: int, seed: int = 0) -> list[tuple[int, int]]:
    """Deterministic (a, b) coefficient pairs for ``h(x) = a*x + b mod p``."""
    out: list[tuple[int, int]] = []
    state = hashlib.sha256(f"aegis-minhash-{seed}".encode()).digest()
    while len(out) < num_perm:
        state = hashlib.sha256(state).digest()
        a = struct.unpack("<Q", state[:8])[0] % (_MERSENNE - 1) + 1
        b = struct.unpack("<Q", state[8:16])[0] % _MERSENNE
        out.append((a, b))
    return out


def signature(text: str, perms: list[tuple[int, int]], shingle_k: int = 5) -> tuple[int, ...]:
    """MinHash signature: one minimum per permutation."""
    grams = shingles(text, shingle_k)
    base = [struct.unpack("<Q", hashlib.blake2b(g, digest_size=8).digest())[0] for g in grams]
    if not base:
        return tuple(0 for _ in perms)
    return tuple(min(((a * h + b) % _MERSENNE) & _MAX64 for h in base) for a, b in perms)


def jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Signature-level Jaccard estimate."""
    if not sig_a:
        return 0.0
    agree = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return agree / len(sig_a)


def collision_probability(similarity: float, bands: int, rows: int) -> float:
    """The LSH banding curve ``1 - (1 - J**r)**b`` of Prop. 1."""
    return 1.0 - (1.0 - similarity**rows) ** bands


@dataclass(slots=True)
class MinHashIndex:
    """Banded LSH index over document signatures.

    Parameters follow the paper: ``num_perm=128`` permutations banded as
    ``bands=32`` bands of ``rows=4``, giving a collision curve with its knee
    near Jaccard 0.5 and near-certain collision above 0.8.
    """

    num_perm: int = 128
    bands: int = 32
    rows: int = 4
    threshold: float = 0.85
    shingle_k: int = 5
    seed: int = 0

    _perms: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _buckets: dict[tuple[int, int], list[str]] = field(default_factory=dict, repr=False)
    signatures: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)
    exact: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.bands * self.rows != self.num_perm:
            raise ValueError(
                f"bands*rows must equal num_perm ({self.bands}*{self.rows} != {self.num_perm})"
            )
        self._perms = _permutations(self.num_perm, self.seed)

    # -- construction -------------------------------------------------------

    def add(self, key: str, text: str) -> tuple[int, ...]:
        sig = signature(text, self._perms, self.shingle_k)
        self.signatures[key] = sig
        for band in range(self.bands):
            chunk = sig[band * self.rows : (band + 1) * self.rows]
            self._buckets.setdefault((band, hash(chunk)), []).append(key)
        self.exact.setdefault(content_hash(text), key)
        return sig

    def build(self, items: list[tuple[str, str]]) -> "MinHashIndex":
        for key, text in items:
            self.add(key, text)
        return self

    # -- queries ------------------------------------------------------------

    def candidate_pairs(self, key: str) -> set[str]:
        """Keys sharing at least one band with ``key`` (no verification)."""
        sig = self.signatures.get(key)
        if sig is None:
            return set()
        out: set[str] = set()
        for band in range(self.bands):
            chunk = sig[band * self.rows : (band + 1) * self.rows]
            out.update(self._buckets.get((band, hash(chunk)), ()))
        out.discard(key)
        return out

    def neighbours(self, key: str, threshold: float | None = None) -> list[tuple[str, float]]:
        """Verified near-duplicates: candidate pairs above the threshold."""
        thr = self.threshold if threshold is None else threshold
        sig = self.signatures[key]
        out = []
        for other in self.candidate_pairs(key):
            sim = jaccard(sig, self.signatures[other])
            if sim >= thr:
                out.append((other, sim))
        return sorted(out, key=lambda t: -t[1])

    def duplicate_of(self, key: str, text: str) -> tuple[str | None, str]:
        """Return ``(source_key, kind)`` with kind in ``exact`` | ``near`` | ``""``."""
        exact_key = self.exact.get(content_hash(text))
        if exact_key is not None and exact_key != key:
            return exact_key, "exact"
        near = self.neighbours(key)
        if near:
            return near[0][0], "near"
        return None, ""

    def all_neighbour_pairs(self, threshold: float | None = None) -> list[tuple[str, str, float]]:
        """Every verified near-duplicate pair, each unordered pair once."""
        thr = self.threshold if threshold is None else threshold
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str, float]] = []
        for key in self.signatures:
            for other, sim in self.neighbours(key, thr):
                pair = (key, other) if key < other else (other, key)
                if pair not in seen:
                    seen.add(pair)
                    out.append((pair[0], pair[1], sim))
        return out
