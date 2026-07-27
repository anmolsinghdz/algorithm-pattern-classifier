from abc import ABC, abstractmethod
from typing import Any

from algorithm_pattern_classifier.models.pattern import AlgorithmPattern
from algorithm_pattern_classifier.models.result import ClassificationResult


class BaseDetector(ABC):
    """Abstract base class for all algorithmic pattern detectors."""

    @property
    @abstractmethod
    def pattern(self) -> AlgorithmPattern:
        """The algorithmic design pattern this detector is designed to identify."""
        pass

    @abstractmethod
    def detect(self, source_code: str, ast_tree: Any = None) -> ClassificationResult:
        """Detect evidence of the specific pattern in the source code.

        Args:
            source_code: The raw source code of the solution.
            ast_tree: Optional pre-parsed AST of the source code.

        Returns:
            A ClassificationResult representing the detection outcome.
        """
        pass
