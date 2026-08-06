import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class BacktrackingDetector(BaseDetector):
    """Detector for the Backtracking algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.BACKTRACKING

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Parse AST and detect Backtracking pattern.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        evidence: list[str] = []
        best_confidence = 0.0

        class BacktrackingVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found = False
                self.confidence = 0.0
                self.evidence: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                func_name = node.name

                mutations: dict[str, list[int]] = {}
                rollbacks: dict[str, list[int]] = {}
                recursive_lines: list[int] = []

                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        is_recur = False
                        if (isinstance(child.func, ast.Name) and child.func.id == func_name) or (
                            isinstance(child.func, ast.Attribute)
                            and isinstance(child.func.value, ast.Name)
                            and child.func.value.id == "self"
                            and child.func.attr == func_name
                        ):
                            is_recur = True

                        if is_recur:
                            recursive_lines.append(getattr(child, "lineno", 0))

                        if isinstance(child.func, ast.Attribute) and isinstance(
                            child.func.value, ast.Name
                        ):
                            var_id = child.func.value.id
                            method_name = child.func.attr
                            if method_name in ("append", "add"):
                                mutations.setdefault(var_id, []).append(getattr(child, "lineno", 0))
                            elif method_name in ("pop", "remove"):
                                rollbacks.setdefault(var_id, []).append(getattr(child, "lineno", 0))

                has_rollback_pattern = False
                matched_var = ""
                mut_l = 0
                rec_l = 0
                roll_l = 0

                for var_id, mut_lines in mutations.items():
                    if var_id in rollbacks:
                        roll_lines = rollbacks[var_id]
                        for m_line in mut_lines:
                            for r_line in recursive_lines:
                                for roll_line in roll_lines:
                                    if m_line < r_line < roll_line:
                                        has_rollback_pattern = True
                                        matched_var = var_id
                                        mut_l = m_line
                                        rec_l = r_line
                                        roll_l = roll_line
                                        break
                                if has_rollback_pattern:
                                    break
                            if has_rollback_pattern:
                                break

                if has_rollback_pattern:
                    has_pruning = False
                    pruning_line = 0
                    for child in ast.walk(node):
                        if isinstance(child, ast.If):
                            for desc in ast.walk(child):
                                if isinstance(desc, (ast.Return, ast.Continue, ast.Break)):
                                    has_pruning = True
                                    pruning_line = getattr(child, "lineno", 0)
                                    break
                            if has_pruning:
                                break

                    self.found = True
                    self.confidence = 0.95 if has_pruning else 0.85
                    pruning_msg = f" with pruning at line {pruning_line}" if has_pruning else ""
                    self.evidence.append(
                        f"Line {node.lineno}: Found backtracking in function '{func_name}' "
                        f"using state variable '{matched_var}' (mutation at line {mut_l}, "
                        f"recursion at line {rec_l}, rollback at line {roll_l}){pruning_msg}."
                    )

                self.generic_visit(node)

        visitor = BacktrackingVisitor()
        visitor.visit(code_ast)

        if visitor.found:
            best_confidence = visitor.confidence
            evidence = visitor.evidence
            return PatternMatch(
                pattern=AlgorithmPattern.BACKTRACKING,
                confidence=round(best_confidence, 2),
                evidence=evidence,
            )

        return None
