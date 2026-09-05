"""Layer V: submodular escalation, surrogates, and the audit ledger."""
from __future__ import annotations

import math

import pytest

from aegis.checksums import iban_valid, luhn_valid, ssn_plausible
from aegis.layer5_act import (
    AuditLedger,
    Surrogate,
    SurrogateKey,
    brute_force_optimum,
    coverage_value,
    lazy_greedy_escalate,
)
from aegis.layer5_act.escalate import banded_spans, oracle_resolver, resolve_escalated
from aegis.types import Document, Span


# --------------------------------------------------------------------------
# Proposition 3
# --------------------------------------------------------------------------


def test_greedy_meets_the_one_minus_one_over_e_bound(small_batch):
    docs, graph = small_batch
    pi = [1.0 if n.score >= 0.2 else 0.0 for n in graph.nodes]
    bound = 1.0 - 1.0 / math.e
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(8):
        eligible = sorted(rng.choice(len(graph.nodes), size=12, replace=False).tolist())
        plan = lazy_greedy_escalate(graph, pi, budget=3, eligible=eligible)
        optimum = brute_force_optimum(graph, pi, 3, eligible)
        if optimum > 0:
            assert coverage_value(graph, plan.chosen, pi) >= bound * optimum - 1e-9


def test_coverage_value_is_submodular_on_the_real_graph(small_batch):
    """F(A + v) - F(A) must not exceed F(B + v) - F(B) for B subset of A."""
    docs, graph = small_batch
    pi = [1.0] * len(graph.nodes)
    import numpy as np

    rng = np.random.default_rng(1)
    for _ in range(30):
        pool = rng.choice(len(graph.nodes), size=8, replace=False).tolist()
        smaller = pool[:3]
        larger = pool[:6]
        v = pool[7]
        gain_small = coverage_value(graph, smaller + [v], pi) - coverage_value(graph, smaller, pi)
        gain_large = coverage_value(graph, larger + [v], pi) - coverage_value(graph, larger, pi)
        assert gain_large <= gain_small + 1e-9


def test_coverage_value_is_monotone(small_batch):
    docs, graph = small_batch
    pi = [1.0] * len(graph.nodes)
    a = coverage_value(graph, [0, 1], pi)
    b = coverage_value(graph, [0, 1, 2], pi)
    assert b >= a


def test_lazy_greedy_respects_the_budget_and_stops_when_dry(small_batch):
    docs, graph = small_batch
    pi = [1.0] * len(graph.nodes)
    plan = lazy_greedy_escalate(graph, pi, budget=5)
    assert len(plan.chosen) <= 5
    zero = lazy_greedy_escalate(graph, [0.0] * len(graph.nodes), budget=5)
    assert zero.chosen == [] and zero.value == 0.0
    empty = lazy_greedy_escalate(graph, pi, budget=0)
    assert empty.chosen == []


def test_lazy_evaluation_saves_work(small_batch):
    """Lazy greedy must not evaluate every element at every step."""
    docs, graph = small_batch
    pi = [1.0] * len(graph.nodes)
    budget = 8
    plan = lazy_greedy_escalate(graph, pi, budget=budget)
    assert plan.evaluations < len(graph.nodes) * budget


def test_escalation_resolves_whole_identity_components(small_batch):
    """Resolving one occurrence must buy every clone -- the source of submodularity."""
    docs, graph = small_batch
    pi = [1.0] * len(graph.nodes)
    multi = [
        indices for indices in graph.components.values() if len(indices) > 1
    ]
    if not multi:
        pytest.skip("no multi-occurrence entity in this batch")
    component = multi[0]
    plan = lazy_greedy_escalate(graph, pi, budget=1, eligible=[component[0]])
    keys = resolve_escalated(docs, graph, plan)
    assert {graph.nodes[i].key() for i in component} <= keys


def test_oracle_resolver_only_confirms_true_spans():
    text = "Borrower Melanie Riley, ref XYZ."
    gold = [Span(9, 22, "PERSON", "Melanie Riley")]
    doc = Document("d", text, gold=gold)
    resolve = oracle_resolver()
    assert resolve(doc, Span(9, 22, "PERSON", "Melanie Riley"))
    assert not resolve(doc, Span(28, 31, "PERSON", "XYZ"))


def test_banded_spans_selects_the_escalation_window(small_batch):
    from aegis.layer2_risk import MaskingFamily

    docs, _ = small_batch
    family = MaskingFamily().fit(docs)
    grid = family.path(64)
    mid = grid[len(grid) // 2]
    for i, j in banded_spans(docs, mid, band_floor=0.2):
        span = docs[i].candidates[j]
        assert 0.2 <= span.score < mid.thresholds[span.type]
    assert banded_spans(docs, grid[-1]) == [], "full redaction escalates nothing"


# --------------------------------------------------------------------------
# Surrogates
# --------------------------------------------------------------------------


def test_surrogates_are_deterministic_under_normalisation():
    s = Surrogate("key-a")
    a = s.for_span(Span(0, 13, "PERSON", "Melanie Riley"))
    b = s.for_span(Span(50, 63, "PERSON", "melanie  riley"))
    c = s.for_span(Span(0, 13, "PERSON", "Riley, Melanie"))
    assert a == b == c


def test_surrogates_change_with_the_key():
    span = Span(0, 13, "PERSON", "Melanie Riley")
    assert Surrogate("key-a").for_span(span) != Surrogate("key-b").for_span(span)


def test_card_surrogate_stays_luhn_valid_and_shape_preserving():
    s = Surrogate("k")
    original = "4532 0151 1283 0366"
    out = s.for_span(Span(0, len(original), "CREDIT_CARD", original))
    assert luhn_valid(out)
    assert len(out) == len(original)
    assert [c.isdigit() for c in out] == [c.isdigit() for c in original]


def test_iban_surrogate_stays_iso7064_valid_and_keeps_country():
    out = Surrogate("k").for_span(Span(0, 22, "IBAN", "DE89370400440532013000"))
    assert iban_valid(out) and out.startswith("DE")


def test_ssn_surrogate_stays_plausible():
    assert ssn_plausible(Surrogate("k").for_span(Span(0, 11, "SSN", "406-44-8691")))


def test_dob_surrogate_preserves_iso_format():
    s = Surrogate("k")
    assert s.for_span(Span(0, 10, "DOB", "1962-01-11")).count("-") == 2
    assert s.for_span(Span(0, 10, "DOB", "11/01/1962")).count("/") == 2


def test_apply_replaces_right_to_left_without_corrupting_offsets():
    text = "Call 555-010-2020 or email a.b@c.com about 4532 0151 1283 0366."
    spans = [
        Span(5, 17, "PHONE", "555-010-2020"),
        Span(27, 36, "EMAIL", "a.b@c.com"),
        Span(43, 62, "CREDIT_CARD", "4532 0151 1283 0366"),
    ]
    out = Surrogate("k").apply(text, spans)
    assert out.startswith("Call ") and " or email " in out and out.endswith(".")
    for span in spans:
        assert span.text not in out


def test_full_redaction_replaces_the_document():
    out = Surrogate("k").apply("anything at all", [Span(0, 15, "__ALL__", "anything at all")])
    assert out == "[REDACTED]"


def test_key_fingerprint_is_stable_and_not_the_secret():
    key = SurrogateKey.from_string("top-secret", "k7")
    fp = key.fingerprint()
    assert fp == SurrogateKey.from_string("top-secret", "k7").fingerprint()
    assert "top-secret" not in fp and len(fp) == 16
    assert fp != SurrogateKey.from_string("other", "k7").fingerprint()


# --------------------------------------------------------------------------
# Audit ledger
# --------------------------------------------------------------------------


def test_ledger_chain_verifies_and_detects_tampering():
    ledger = AuditLedger()
    ledger.append(text="doc one", stage="mask", n_masked=3)
    ledger.append(text="doc two", stage="mask", n_masked=1)
    assert ledger.verify()
    ledger.rows[0].n_masked = 99
    assert not ledger.verify()


def test_ledger_detects_reordering():
    ledger = AuditLedger()
    for i in range(4):
        ledger.append(text=f"doc {i}", stage="mask")
    ledger.rows[1], ledger.rows[2] = ledger.rows[2], ledger.rows[1]
    assert not ledger.verify()


def test_ledger_round_trips_through_jsonl():
    import tempfile
    from pathlib import Path

    ledger = AuditLedger()
    ledger.append(text="a", stage="mask", leakage=0.0, verdict="valid")
    ledger.append(text="b", stage="mask", leakage=0.5, verdict="void")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        ledger.to_jsonl(path)
        again = AuditLedger.from_jsonl(path)
    assert again.verify()
    assert again.head == ledger.head
    assert [r.verdict for r in again.rows] == ["valid", "void"]


def test_ledger_keys_rows_by_canonical_content():
    ledger = AuditLedger()
    a = ledger.append(text="Hello  World", stage="mask")
    b = ledger.append(text="hello world", stage="mask")
    assert a.doc_hash == b.doc_hash
