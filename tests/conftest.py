"""Test fixtures. Adds ``src`` to the path so tests run without installation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aegis.data import DATA_ROOT, load_synpii, split_calibration_test  # noqa: E402
from aegis.layer1_candidates import build_ensemble  # noqa: E402


@pytest.fixture(scope="session")
def has_data() -> bool:
    return (DATA_ROOT / "synpii" / "synpii_finance_seed7.jsonl").exists()


@pytest.fixture(scope="session")
def synpii(has_data):
    if not has_data:
        pytest.skip("bundled corpora not present")
    docs, _ = load_synpii(7)
    return docs


@pytest.fixture(scope="session")
def small_batch(synpii):
    """A small pooled batch with candidates and propagated scores attached."""
    from aegis.layer1_candidates import build_entity_graph, propagate

    docs = [d for d in synpii[:80]]
    ensemble = build_ensemble("rule,heuristic-ner,shape")
    for doc in docs:
        doc.candidates = ensemble.detect(doc.text)
    graph = build_entity_graph(docs)
    for node, score in zip(graph.nodes, propagate(graph)):
        node.score = score
    return docs, graph


@pytest.fixture(scope="session")
def split(synpii):
    return split_calibration_test(synpii, cal_frac=0.4, seed=0)
