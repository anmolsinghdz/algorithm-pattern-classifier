from dataclasses import dataclass, field

from algorithm_pattern_classifier.models.pattern import AlgorithmPattern


@dataclass(frozen=True)
class ClassificationResult:
    """The classification outcome for a single algorithmic pattern."""

    pattern: AlgorithmPattern
    confidence_score: float
    supporting_evidence: list[str] = field(default_factory=list)
