import ast
from typing import Any

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.pattern import AlgorithmPattern
from algorithm_pattern_classifier.models.result import ClassificationResult


class TwoPointerDetector(BaseDetector):
    """Detector for the Two-Pointer algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.TWO_POINTER

    def detect(self, source_code: str, ast_tree: Any = None) -> ClassificationResult:
        """Detect evidence of the Two-Pointer pattern in source code.

        Args:
            source_code: The raw source code of the solution.
            ast_tree: Optional pre-parsed AST of the source code.

        Returns:
            A ClassificationResult representing the detection outcome.
        """
        if ast_tree is None:
            try:
                ast_tree = ast.parse(source_code)
            except SyntaxError:
                return ClassificationResult(
                    pattern=self.pattern,
                    confidence_score=0.0,
                    supporting_evidence=["Syntax error during parsing"],
                )

        evidence: list[str] = []
        max_confidence = 0.0

        class TwoPointerVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_two_pointer = False
                self.evidence: list[str] = []
                self.confidence = 0.0

            def visit_While(self, node: ast.While) -> None:
                # 1. Look for loop condition that compares two variables
                if not isinstance(node.test, ast.Compare):
                    self.generic_visit(node)
                    return

                comp = node.test
                if not (
                    isinstance(comp.left, ast.Name)
                    and len(comp.comparators) == 1
                    and isinstance(comp.comparators[0], ast.Name)
                ):
                    self.generic_visit(node)
                    return

                var1 = comp.left.id
                var2 = comp.comparators[0].id
                cand_pointers = {var1, var2}

                # 2. Walk the loop body to see if BOTH variables are updated
                updated_vars: set[str] = set()

                class UpdateVisitor(ast.NodeVisitor):
                    def visit_Assign(self, assign_node: ast.Assign) -> None:
                        for target in assign_node.targets:
                            if isinstance(target, ast.Name):
                                updated_vars.add(target.id)
                        self.generic_visit(assign_node)

                    def visit_AugAssign(self, aug_node: ast.AugAssign) -> None:
                        if isinstance(aug_node.target, ast.Name):
                            updated_vars.add(aug_node.target.id)
                        self.generic_visit(aug_node)

                UpdateVisitor().visit(node)

                if cand_pointers.issubset(updated_vars):
                    # We have a candidate two-pointer pattern loop
                    self.found_two_pointer = True
                    self.confidence = 0.95
                    self.evidence.append(
                        f"Line {node.lineno}: found loop condition comparing '{var1}' and '{var2}' "
                        f"where both are updated in the loop body."
                    )

                self.generic_visit(node)

        visitor = TwoPointerVisitor()
        visitor.visit(ast_tree)

        if visitor.found_two_pointer:
            max_confidence = visitor.confidence
            evidence = visitor.evidence

        return ClassificationResult(
            pattern=self.pattern,
            confidence_score=max_confidence,
            supporting_evidence=evidence,
        )
