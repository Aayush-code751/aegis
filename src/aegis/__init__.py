"""AEGIS: Distribution-Shift-Certified Risk Control for PII Redaction.

Reference implementation for the paper

    AEGIS: Distribution-Shift-Certified Risk Control for PII Redaction in
    Financial Document Streams.

Layer map (paper section -> module):

    Layer I    candidate formation + entity-graph propagation
               aegis.layer1_candidates
    Layer II   document-level conformal risk control + Learn-then-Test
               aegis.layer2_risk
    Layer III  weighted CRC under estimated density ratios + chi^2-DRO radius
               aegis.layer3_shift
    Layer IV   anytime-valid e-processes, e-BH, Chao1 dark-matter bound
               aegis.layer4_monitor
    Layer V    submodular escalation, HMAC surrogates, audit ledger
               aegis.layer5_act
"""
from __future__ import annotations

__version__ = "1.0.0"

from .types import Span, Document, Certificate
from .taxonomy import FINANCE_TYPES, CANONICAL_TYPES

__all__ = [
    "__version__",
    "Span",
    "Document",
    "Certificate",
    "FINANCE_TYPES",
    "CANONICAL_TYPES",
]
