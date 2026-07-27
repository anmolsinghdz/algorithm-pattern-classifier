import ast

from algorithm_pattern_classifier.detectors.dynamic_programming import DynamicProgrammingDetector
from algorithm_pattern_classifier.detectors.sliding_window import SlidingWindowDetector
from algorithm_pattern_classifier.detectors.two_pointer import TwoPointerDetector
from algorithm_pattern_classifier.interfaces.classifier import BaseClassifier
from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.pattern import AlgorithmPattern
from algorithm_pattern_classifier.models.result import ClassificationResult


class PatternClassifier(BaseClassifier):
    """Aggregates and ranks algorithmic design pattern detections."""

    def __init__(self, detectors: list[BaseDetector] | None = None) -> None:
        """Initialize the classifier with optional custom detectors.

        Args:
            detectors: A list of detectors. If None, defaults to registering
                       TwoPointerDetector, SlidingWindowDetector, and DynamicProgrammingDetector.
        """
        if detectors is None:
            self.detectors: list[BaseDetector] = [
                TwoPointerDetector(),
                SlidingWindowDetector(),
                DynamicProgrammingDetector(),
            ]
        else:
            self.detectors = detectors

    def classify(self, source_code: str) -> list[ClassificationResult]:
        """Analyze source code using registered detectors and return ranked patterns.

        Args:
            source_code: The raw source code of the solution.

        Returns:
            A list of ClassificationResult objects ranked by confidence score.
        """
        try:
            ast_tree = ast.parse(source_code)
        except SyntaxError:
            ast_tree = None

        raw_results: list[ClassificationResult] = []
        for detector in self.detectors:
            try:
                res = detector.detect(source_code, ast_tree=ast_tree)
                if res.confidence_score > 0.0:
                    raw_results.append(res)
            except Exception:
                # Robustly proceed if an individual detector raises an error
                continue

        # Handle ties/overlaps:
        # 1. If dynamic-programming is detected, it is highly specific and suppresses
        #    sliding-window and two-pointer.
        has_dp = any(r.pattern == AlgorithmPattern.DYNAMIC_PROGRAMMING for r in raw_results)
        if has_dp:
            dp_results = [
                r for r in raw_results if r.pattern == AlgorithmPattern.DYNAMIC_PROGRAMMING
            ]
            max_dp_conf = max(r.confidence_score for r in dp_results)
            if max_dp_conf >= 0.5:
                raw_results = [
                    r
                    for r in raw_results
                    if r.pattern
                    not in (AlgorithmPattern.SLIDING_WINDOW, AlgorithmPattern.TWO_POINTER)
                ]

        # 2. If sliding-window is detected and has equal or higher confidence,
        #    suppress the generic two-pointer result.
        has_sliding_window = any(r.pattern == AlgorithmPattern.SLIDING_WINDOW for r in raw_results)
        if has_sliding_window:
            sw_results = [r for r in raw_results if r.pattern == AlgorithmPattern.SLIDING_WINDOW]
            max_sw_conf = max(r.confidence_score for r in sw_results)
            raw_results = [
                r
                for r in raw_results
                if not (
                    r.pattern == AlgorithmPattern.TWO_POINTER and r.confidence_score <= max_sw_conf
                )
            ]

        # Rank results by descending confidence score, then by pattern value for tie-breaking
        return sorted(
            raw_results,
            key=lambda r: (r.confidence_score, r.pattern.value),
            reverse=True,
        )
