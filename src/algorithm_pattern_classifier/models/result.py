from dataclasses import dataclass, field

from algorithm_pattern_classifier.models.pattern import AlgorithmPattern


@dataclass(frozen=True)
class ClassificationResult:
    """The classification outcome for a single algorithmic pattern.

    Attributes:
        pattern: The algorithm pattern identified.
        confidence_score: Score from 0.0 (no confidence) to 1.0 (certainty).
        supporting_evidence: List of code snippets or line evidence supporting the classification.
    """

    pattern: AlgorithmPattern
    confidence_score: float
    supporting_evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate attribute bounds."""
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )
