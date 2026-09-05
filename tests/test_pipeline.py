"""End-to-end integration: Algorithm 1, and the properties it must preserve."""
from __future__ import annotations

import pytest

from aegis.pipeline import AegisConfig, AegisPipeline
from aegis.types import Document


@pytest.fixture(scope="module")
def run(split):
    cal, test = split
    pipeline = AegisPipeline(AegisConfig(alpha=0.20, gamma=0.20, grid_size=256,
                                         escalation_budget=0.25))
    cert, report = pipeline.run(list(cal), list(test))
    return pipeline, cert, report


def test_pipeline_produces_a_certificate(run):
    _, cert, _ = run
    assert not cert.empty
    assert cert.alpha == 0.20
    assert cert.rho_star >= 0.0
    assert cert.verdict in {"valid", "void", "valid-at-P"}


def test_realised_leakage_respects_the_declared_level(run):
    _, cert, report = run
    assert report.realised_leakage is not None
    assert report.realised_leakage <= cert.alpha + 1e-9


def test_certificate_serialises_to_plain_json(run):
    import json

    _, cert, _ = run
    blob = json.dumps(cert.to_dict())          # must not contain Infinity/NaN
    assert "Infinity" not in blob and "NaN" not in blob
    assert json.loads(blob)["alpha"] == 0.20


def test_ledger_covers_every_document_and_verifies(run):
    pipeline, _, report = run
    assert report.ledger["chain_valid"]
    assert report.ledger["n_rows"] == report.n_docs


def test_propagation_ran_over_the_pooled_batch(run):
    """Lemma 1 needs one unordered multiset, so the graph must span cal + test."""
    pipeline, _, report = run
    graph = pipeline.graph
    assert graph is not None
    assert len(pipeline._pooled) > report.n_docs


def test_layer_four_diagnostics_are_present(run):
    _, _, report = run
    assert report.candidate_miss and "micro" in report.candidate_miss
    assert report.effective_miss and "micro" in report.effective_miss
    assert report.dark_matter, "the label-free estimate must always be reported"
    assert isinstance(report.alarms, list)


def test_tightening_alpha_cannot_increase_realised_leakage(split):
    cal, test = split
    results = {}
    for alpha in (0.50, 0.20):
        pipeline = AegisPipeline(AegisConfig(alpha=alpha, gamma=0.50, grid_size=256))
        cert, report = pipeline.run(list(cal), list(test))
        if not cert.empty:
            results[alpha] = report.realised_leakage
    if len(results) == 2:
        assert results[0.20] <= results[0.50] + 1e-9


def test_infeasible_alpha_yields_no_certificate_and_routes_to_review(split):
    cal, test = split
    pipeline = AegisPipeline(AegisConfig(alpha=1e-6, gamma=1e-6, grid_size=128))
    cert, report = pipeline.run(list(cal), list(test))
    assert cert.empty and cert.verdict == "no-certificate"
    assert report.escalated_fraction == 1.0
    assert report.realised_overmask == 1.0
    # a refusal is still auditable
    assert report.ledger["n_rows"] == report.n_docs
    assert "route-to-review" in report.ledger["stages"]


def test_uncertified_point_claims_no_radius(split):
    cal, test = split
    pipeline = AegisPipeline(AegisConfig(select_by="max_f1", grid_size=256))
    cert, report = pipeline.run(list(cal), list(test))
    assert cert.rho_star == 0.0, "an uncertified point must not advertise a radius"
    assert cert.diagnostics.get("uncertified") is True
    assert report.span_metrics["micro"]["F1"] > 0.0


def test_transfer_to_a_disjoint_corpus_is_declared_void(has_data, split):
    """The framework must predict its own failure, before reading target labels."""
    if not has_data:
        pytest.skip("bundled corpora not present")
    from aegis.data import load_corpus

    cal, _ = split
    target, _ = load_corpus("gretel")
    pipeline = AegisPipeline(AegisConfig(alpha=0.20, grid_size=128))
    cert = pipeline.calibrate(list(cal), list(target[:400]))
    assert cert.rho_hat is not None
    assert cert.rho_hat > cert.rho_star, "a template change must not read as in-scope"
    assert cert.verdict == "void"


def test_redaction_emits_text_without_the_original_spans(split):
    cal, test = split
    pipeline = AegisPipeline(AegisConfig(alpha=0.20, gamma=0.20, grid_size=128))
    cert = pipeline.calibrate(list(cal), list(test))
    target = [d for d in test if d.gold][:5]
    report = pipeline.deploy(target, cert, emit_text=True)
    assert len(report.anonymised) == len(target)
    for doc, out in zip(target, report.anonymised):
        masked = {s.text for s in pipeline.family.mask(doc, pipeline.grid[pipeline._selected_index])}
        for surface in masked:
            if len(surface) > 6:            # short surfaces can recur by chance
                assert surface not in out


def test_config_round_trips_through_a_dict():
    cfg = AegisConfig(alpha=0.07, engines="rule")
    blob = cfg.as_dict()
    assert blob["alpha"] == 0.07 and blob["engines"] == "rule"
    assert AegisConfig(**{k: v for k, v in blob.items() if k != "types"}).alpha == 0.07


def test_deploy_before_calibrate_is_an_error():
    pipeline = AegisPipeline()
    with pytest.raises(RuntimeError):
        pipeline.deploy([Document("d", "text")])
