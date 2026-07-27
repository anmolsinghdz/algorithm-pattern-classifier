from abc import ABC, abstractmethod

from algorithm_pattern_classifier.models.result import ClassificationResult


class BaseClassifier(ABC):
    """Abstract base class for algorithm pattern classifiers."""

    @abstractmethod
    def classify(self, source_code: str) -> list[ClassificationResult]:
        """Classify and rank the algorithmic design patterns found in the source code.

        Args:
            source_code: The raw source code of the solution.

        Returns:
            A list of ClassificationResult objects ranked by confidence score.
        """
        pass
