"""Layer I: candidate formation and entity-graph propagation."""
from .engines import (
    Engine,
    RuleEngine,
    HeuristicNEREngine,
    ShapeClassEngine,
    GLiNEREngine,
    PresidioEngine,
    LLMEngine,
    Ensemble,
    build_ensemble,
    dedupe_overlaps,
)
from .minhash import MinHashIndex, shingles, canonicalise
from .propagation import (
    EntityGraph,
    build_entity_graph,
    propagate,
    PropagationConfig,
)

__all__ = [
    "Engine",
    "RuleEngine",
    "HeuristicNEREngine",
    "ShapeClassEngine",
    "GLiNEREngine",
    "PresidioEngine",
    "LLMEngine",
    "Ensemble",
    "build_ensemble",
    "dedupe_overlaps",
    "MinHashIndex",
    "shingles",
    "canonicalise",
    "EntityGraph",
    "build_entity_graph",
    "propagate",
    "PropagationConfig",
]
