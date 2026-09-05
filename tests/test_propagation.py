"""Layer I: Lemma 1 (equivariance) and Proposition 1 (amplification, leak bound)."""
from __future__ import annotations

import random

import pytest

from aegis.layer1_candidates.minhash import (
    MinHashIndex,
    collision_probability,
    content_hash,
    jaccard,
)
from aegis.layer1_candidates.propagation import (
    PropagationConfig,
    build_entity_graph,
    cross_entity_leak_bound,
    normalise_value,
    propagate,
    recall_amplification,
)
from aegis.types import Document, Span


def _batch(seed: int = 0) -> list[Document]:
    rng = random.Random(seed)
    docs = []
    for i in range(12):
        name = rng.choice(["Melanie Riley", "Yuki Tanaka", "Sofia Marino"])
        text = f"Wire memo {i}. Borrower {name}. Card 4532 0151 1283 0366."
        gold = [Span(text.index(name), text.index(name) + len(name), "PERSON", name)]
        doc = Document(doc_id=f"d{i}", text=text, gold=gold)
        doc.candidates = [
            Span(text.index(name), text.index(name) + len(name), "PERSON", name,
                 rng.choice([0.35, 0.9]), "rule")
        ]
        docs.append(doc)
    return docs


def test_lemma1_permutation_equivariance():
    """Permuting the batch must permute the propagated scores, not change them.

    This is Lemma 1's hypothesis, and it is what lets the downstream conformal
    guarantee apply to propagated scores unchanged.
    """
    docs = _batch()
    graph = build_entity_graph(docs)
    base = propagate(graph)
    by_key = {(graph.doc_of[i], graph.nodes[i].key()): v for i, v in enumerate(base)}

    order = list(range(len(docs)))
    random.Random(7).shuffle(order)
    permuted = [docs[i] for i in order]
    pgraph = build_entity_graph(permuted)
    pscores = propagate(pgraph)

    for i, value in enumerate(pscores):
        original_doc = order[pgraph.doc_of[i]]
        assert by_key[(original_doc, pgraph.nodes[i].key())] == pytest.approx(value, abs=1e-12)


def test_propagation_is_monotone_and_bounded():
    docs = _batch()
    graph = build_entity_graph(docs)
    before = [n.score for n in graph.nodes]
    after = propagate(graph)
    assert all(a >= b - 1e-12 for a, b in zip(after, before)), "propagation must not lower a score"
    assert all(0.0 <= v <= 1.0 for v in after)


def test_identity_channel_lifts_a_weak_clone():
    """A low-scoring occurrence in one document is lifted by a strong clone."""
    docs = _batch()
    graph = build_entity_graph(docs)
    after = propagate(graph, config=PropagationConfig(beta1=0.95, beta2=0.0))
    weak = [i for i, n in enumerate(graph.nodes) if n.score < 0.5]
    assert weak, "fixture should contain weak candidates"
    assert any(after[i] > graph.nodes[i].score + 1e-9 for i in weak)


def test_disabling_propagation_is_identity():
    docs = _batch()
    graph = build_entity_graph(docs)
    off = propagate(graph, config=PropagationConfig(beta1=0.0, beta2=0.0))
    assert off == pytest.approx([n.score for n in graph.nodes], abs=1e-12)


def test_person_normalisation_merges_orderings_and_roles():
    assert normalise_value("Melanie Riley", "PERSON") == normalise_value("Riley, Melanie", "PERSON")
    assert normalise_value("Borrower Melanie Riley", "PERSON") == \
        normalise_value("Melanie Riley", "PERSON")


def test_checksummed_normalisation_separates_verification_classes():
    good = normalise_value("4532 0151 1283 0366", "CREDIT_CARD")
    bad = normalise_value("4532 0151 1283 0367", "CREDIT_CARD")
    assert good.endswith("luhn-ok") and bad.endswith("luhn-bad")


@pytest.mark.parametrize("q,m", [(0.85, 1), (0.85, 3), (0.5, 4), (0.99, 2)])
def test_recall_amplification_matches_closed_form(q, m):
    assert recall_amplification(q, m) == pytest.approx(1 - (1 - q) ** m)


def test_recall_amplification_is_monotone_in_multiplicity():
    values = [recall_amplification(0.7, m) for m in range(1, 8)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_cross_entity_leak_bound_grows_with_damping():
    assert cross_entity_leak_bound(0.6, 0.1) < cross_entity_leak_bound(0.9, 0.1)
    with pytest.raises(ValueError):
        cross_entity_leak_bound(1.0, 0.1)


def test_lsh_collision_curve_is_monotone():
    values = [collision_probability(j / 20, 32, 4) for j in range(21)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)


def test_minhash_finds_near_duplicates_and_ignores_unrelated():
    base = "KYC summary for Melanie Riley, account 95732841, card 4532 0151 1283 0366."
    near = base + "\n\nRouting footer: EU-WEST-2."
    far = "Safety data sheet. Toxicity assessment of substance X. No personal data."
    index = MinHashIndex(threshold=0.6).build([("a", base), ("b", near), ("c", far)])
    assert {k for k, _ in index.neighbours("a")} == {"b"}
    assert index.neighbours("c") == []


def test_content_hash_folds_whitespace_and_case():
    assert content_hash("Hello  World") == content_hash("hello world")
    assert content_hash("Hello") != content_hash("Goodbye")


def test_signature_jaccard_tracks_true_similarity():
    index = MinHashIndex()
    a = index.add("a", "the quick brown fox jumps over the lazy dog")
    b = index.add("b", "the quick brown fox jumps over the lazy cat")
    c = index.add("c", "completely unrelated financial disclosure text")
    assert jaccard(a, b) > jaccard(a, c)
