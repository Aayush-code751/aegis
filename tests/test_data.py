"""Corpus loading, label mapping, and the calibration/test split."""
from __future__ import annotations

import pytest

from aegis.data import load_corpus, split_calibration_test
from aegis.taxonomy import CANONICAL_TYPES, FINANCE_TYPES, GRETEL_TO_CANON, map_label


def test_finance_taxonomy_is_a_subset_of_canonical():
    assert set(FINANCE_TYPES) <= set(CANONICAL_TYPES)
    assert len(FINANCE_TYPES) == 9


def test_label_mapping_passes_canonical_labels_through():
    assert map_label("PERSON", GRETEL_TO_CANON) == "PERSON"
    assert map_label("person", GRETEL_TO_CANON) == "PERSON"
    assert map_label("DATE-OF-BIRTH", {}) is None


def test_label_mapping_translates_native_vocabulary():
    assert map_label("first_name", GRETEL_TO_CANON) == "PERSON"
    assert map_label("bank_account_number", GRETEL_TO_CANON) == "ACCOUNT_NUMBER"


def test_unmapped_labels_are_dropped_not_forced():
    assert map_label("api_key", GRETEL_TO_CANON) is None
    assert map_label("password", GRETEL_TO_CANON) is None


def test_synpii_loads_with_in_range_spans(synpii):
    assert len(synpii) > 100
    for doc in synpii[:50]:
        for span in doc.gold:
            assert 0 <= span.start < span.end <= len(doc.text)
            assert doc.text[span.start:span.end] == span.text
            assert span.type in FINANCE_TYPES


def test_split_is_deterministic_and_disjoint(synpii):
    a_cal, a_test = split_calibration_test(synpii, cal_frac=0.4, seed=3)
    b_cal, b_test = split_calibration_test(synpii, cal_frac=0.4, seed=3)
    assert [d.doc_id for d in a_cal] == [d.doc_id for d in b_cal]
    assert not ({d.doc_id for d in a_cal} & {d.doc_id for d in a_test})
    assert len(a_cal) + len(a_test) == len(synpii)


def test_different_seeds_give_different_splits(synpii):
    a, _ = split_calibration_test(synpii, cal_frac=0.4, seed=0)
    b, _ = split_calibration_test(synpii, cal_frac=0.4, seed=1)
    assert [d.doc_id for d in a] != [d.doc_id for d in b]


def test_injected_duplicates_ride_the_test_stream(synpii):
    """Calibration must stay duplicate-free so propagation is exercised on test."""
    cal, test = split_calibration_test(synpii, cal_frac=0.4, seed=0)
    assert not any(d.meta.get("is_dup") for d in cal)
    assert any(d.meta.get("is_dup") for d in test)


def test_unknown_corpus_is_rejected():
    with pytest.raises(ValueError):
        load_corpus("not-a-corpus")


def test_external_corpora_map_onto_the_taxonomy(has_data):
    if not has_data:
        pytest.skip("bundled corpora not present")
    for name in ("gretel", "nemotron"):
        docs, stats = load_corpus(name)
        assert len(docs) > 1000
        assert stats.kept > 1000
        assert stats.kept / stats.total > 0.5, "most labels should survive mapping"
        for doc in docs[:20]:
            for span in doc.gold:
                assert span.type in FINANCE_TYPES
                assert 0 <= span.start < span.end <= len(doc.text)
