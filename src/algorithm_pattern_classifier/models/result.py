import math
from dataclasses import dataclass, field

from algorithm_pattern_classifier.models.pattern import AlgorithmPattern


@dataclass(frozen=True)
class ClassificationResult:
    """The classification outcome for a single algorithmic pattern.

    Attributes:
        pattern: The algorithm pattern identified.
        confidence_score: Score from 0.0 (no confidence) to 1.0 (certainty).
        supporting_evidence: Tuple of code snippets or line evidence supporting the classification.
    """

    pattern: AlgorithmPattern
    confidence_score: float
    supporting_evidence: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate attribute bounds and normalize types."""
        if isinstance(self.confidence_score, bool) or not isinstance(
            self.confidence_score, (int, float)
        ):
            score_type = type(self.confidence_score).__name__
            raise TypeError(f"confidence_score must be a numeric float or int, got {score_type}")

        if math.isnan(self.confidence_score) or not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )

        if not isinstance(self.supporting_evidence, tuple):
            object.__setattr__(self, "supporting_evidence", tuple(self.supporting_evidence))
