import ast
import logging

from algorithm_pattern_classifier.classifiers.arbitrator import PatternArbitrator
from algorithm_pattern_classifier.detectors.backtracking import BacktrackingDetector
from algorithm_pattern_classifier.detectors.bfs import BFSDetector
from algorithm_pattern_classifier.detectors.dfs import DFSDetector
from algorithm_pattern_classifier.detectors.dynamic_programming import DynamicProgrammingDetector
from algorithm_pattern_classifier.detectors.sliding_window import SlidingWindowDetector
from algorithm_pattern_classifier.detectors.two_pointers import TwoPointersDetector
from algorithm_pattern_classifier.interfaces.classifier import BaseClassifier
from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import PatternMatch
from algorithm_pattern_classifier.utils.ast_normalizer import ASTNormalizer

logger = logging.getLogger(__name__)


class PatternClassifier(BaseClassifier):
    """Aggregates and ranks algorithmic design pattern detections."""

    def __init__(
        self,
        detectors: list[BaseDetector] | None = None,
        arbitrator: PatternArbitrator | None = None,
    ) -> None:
        """Initialize the classifier with optional custom detectors and arbitrator.

        Args:
            detectors: A list of detectors. If None, defaults to registering default detectors.
            arbitrator: An arbitrator instance. If None, defaults to PatternArbitrator.
        """
        if detectors is None:
            self.detectors = [
                TwoPointersDetector(),
                SlidingWindowDetector(),
                DynamicProgrammingDetector(),
                BFSDetector(),
                DFSDetector(),
                BacktrackingDetector(),
            ]
        else:
            self.detectors = detectors

        self.arbitrator = arbitrator or PatternArbitrator()

    def classify(self, source_code: str) -> list[PatternMatch]:
        """Classify and rank the algorithmic design patterns found in the source code.

        Args:
            source_code: The raw source code of the solution.

        Returns:
            A list of PatternMatch objects ranked by confidence.
        """
        ast_tree = ast.parse(source_code)
        ast_tree = ASTNormalizer().visit(ast_tree)
        ast.fix_missing_locations(ast_tree)

        raw_results: list[PatternMatch] = []
        for detector in self.detectors:
            try:
                res = detector.detect(ast_tree)
                if res is not None and res.confidence > 0.0:
                    raw_results.append(res)
            except Exception:
                logger.exception("Detector %s failed", type(detector).__name__)
                continue

        return self.arbitrator.arbitrate(raw_results)
