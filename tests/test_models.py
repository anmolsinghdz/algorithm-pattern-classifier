from typing import Any

import pytest

from algorithm_pattern_classifier.interfaces.classifier import BaseClassifier
from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.pattern import AlgorithmPattern
from algorithm_pattern_classifier.models.result import ClassificationResult


def test_algorithm_pattern_enum() -> None:
    """Test AlgorithmPattern enum members and values."""
    assert len(AlgorithmPattern) >= 6
    assert AlgorithmPattern.TWO_POINTER.value == "two-pointer"
    assert AlgorithmPattern.SLIDING_WINDOW.value == "sliding-window"
    assert AlgorithmPattern.DYNAMIC_PROGRAMMING.value == "dynamic-programming"
    assert AlgorithmPattern.BACKTRACKING.value == "backtracking"
    assert AlgorithmPattern.BFS_DFS.value == "bfs-dfs"
    assert AlgorithmPattern.GREEDY.value == "greedy"
    assert AlgorithmPattern.DIVIDE_AND_CONQUER.value == "divide-and-conquer"
    assert AlgorithmPattern.BINARY_SEARCH.value == "binary-search"


def test_classification_result_construction_and_equality() -> None:
    """Test ClassificationResult construction, defaults, value equality, and hashability."""
    result1 = ClassificationResult(
        pattern=AlgorithmPattern.TWO_POINTER,
        confidence_score=0.85,
        supporting_evidence=["line 10: left, right pointers initialized"],  # type: ignore[arg-type]
    )
    result2 = ClassificationResult(
        pattern=AlgorithmPattern.TWO_POINTER,
        confidence_score=0.85,
        supporting_evidence=("line 10: left, right pointers initialized",),
    )

    assert result1 == result2
    assert result1.pattern == AlgorithmPattern.TWO_POINTER
    assert result1.confidence_score == 0.85
    assert isinstance(result1.supporting_evidence, tuple)
    assert len(result1.supporting_evidence) == 1

    # Test defaults
    result_default = ClassificationResult(
        pattern=AlgorithmPattern.BINARY_SEARCH, confidence_score=0.5
    )
    assert result_default.supporting_evidence == ()

    # Test immutability and hashability
    assert isinstance(hash(result1), int)
    assert hash(result1) == hash(result2)
    assert len({result1, result2}) == 1
    assert {result1: "data"}[result2] == "data"
    with pytest.raises(AttributeError):
        result1.supporting_evidence.append("new evidence")  # type: ignore[attr-defined]


def test_classification_result_validation() -> None:
    """Test confidence_score bounds and edge-case validation in ClassificationResult."""
    # Scores outside [0.0, 1.0] should raise ValueError
    with pytest.raises(ValueError, match=r"confidence_score must be between 0\.0 and 1\.0"):
        ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=-0.1)

    with pytest.raises(ValueError, match=r"confidence_score must be between 0\.0 and 1\.0"):
        ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=1.1)

    # float("nan") and infinity values should raise ValueError
    with pytest.raises(ValueError, match=r"confidence_score must be between 0\.0 and 1\.0"):
        ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=float("nan"))

    with pytest.raises(ValueError, match=r"confidence_score must be between 0\.0 and 1\.0"):
        ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=float("inf"))

    with pytest.raises(ValueError, match=r"confidence_score must be between 0\.0 and 1\.0"):
        ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=-float("inf"))

    # Boolean values should raise TypeError
    with pytest.raises(TypeError, match=r"confidence_score must be a numeric float or int"):
        ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=True)

    with pytest.raises(TypeError, match=r"confidence_score must be a numeric float or int"):
        ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=False)

    # Valid boundary scores 0.0 and 1.0 should work
    res_min = ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=0.0)
    assert res_min.confidence_score == 0.0

    res_max = ClassificationResult(pattern=AlgorithmPattern.TWO_POINTER, confidence_score=1.0)
    assert res_max.confidence_score == 1.0


def test_interfaces_implementability() -> None:
    """Verify that BaseDetector and BaseClassifier can be concretely subclassed."""

    class MockDetector(BaseDetector):
        @property
        def pattern(self) -> AlgorithmPattern:
            return AlgorithmPattern.TWO_POINTER

        def detect(self, source_code: str, ast_tree: Any = None) -> ClassificationResult:
            _ = source_code, ast_tree
            return ClassificationResult(
                pattern=self.pattern,
                confidence_score=1.0,
                supporting_evidence=("found while loop with left < right",),
            )

    class MockClassifier(BaseClassifier):
        def classify(self, source_code: str) -> list[ClassificationResult]:
            _ = source_code
            return [
                ClassificationResult(
                    pattern=AlgorithmPattern.TWO_POINTER,
                    confidence_score=1.0,
                    supporting_evidence=("mocked",),
                )
            ]

    detector = MockDetector()
    assert detector.pattern == AlgorithmPattern.TWO_POINTER
    det_res = detector.detect("while left < right:")
    assert det_res.confidence_score == 1.0

    classifier = MockClassifier()
    cls_res = classifier.classify("code")
    assert len(cls_res) == 1
    assert cls_res[0].pattern == AlgorithmPattern.TWO_POINTER
