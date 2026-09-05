"""Layer V: cost-optimal escalation and safe output."""
from .escalate import (
    EscalationPlan,
    banded_spans,
    coverage_value,
    lazy_greedy_escalate,
    brute_force_optimum,
    oracle_resolver,
    resolve_escalated,
)
from .surrogate import Surrogate, SurrogateKey
from .ledger import AuditLedger, LedgerRow

__all__ = [
    "EscalationPlan",
    "banded_spans",
    "coverage_value",
    "lazy_greedy_escalate",
    "brute_force_optimum",
    "oracle_resolver",
    "resolve_escalated",
    "Surrogate",
    "SurrogateKey",
    "AuditLedger",
    "LedgerRow",
]
