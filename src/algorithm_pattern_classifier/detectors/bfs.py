import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class BFSDetector(BaseDetector):
    """Detector for the Breadth-First Search (BFS) algorithmic design pattern."""

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Parse AST and detect BFS pattern.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        evidence: list[str] = []
        best_confidence = 0.0

        class BFSVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_bfs = False
                self.evidence: list[str] = []
                self.confidence = 0.0
                self.initialized_queues: dict[str, ast.AST] = {}

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                old_queues = self.initialized_queues.copy()
                self.initialized_queues = {}

                # Find all queue/list initializations in this function body
                for child in node.body:
                    for sub_node in ast.walk(child):
                        if isinstance(sub_node, ast.Assign):
                            for target in sub_node.targets:
                                if isinstance(target, ast.Name) and self._is_queue_init(
                                    sub_node.value
                                ):
                                    self.initialized_queues[target.id] = sub_node
                        elif isinstance(sub_node, ast.AnnAssign) and (
                            isinstance(sub_node.target, ast.Name)
                            and sub_node.value
                            and self._is_queue_init(sub_node.value)
                        ):
                            self.initialized_queues[sub_node.target.id] = sub_node

                self.generic_visit(node)
                self.initialized_queues = old_queues

            def _is_queue_init(self, value: ast.expr) -> bool:
                if isinstance(value, ast.Call):
                    func = value.func
                    if isinstance(func, ast.Name) and func.id == "deque":
                        return True
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "deque"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "collections"
                    ):
                        return True
                return bool(isinstance(value, ast.List))

            def visit_While(self, node: ast.While) -> None:
                queue_var = self._get_loop_test_var(node.test)
                if queue_var and queue_var in self.initialized_queues:
                    has_pop_front = False
                    has_append = False
                    pop_line = 0
                    append_line = 0

                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            func = child.func
                            if (
                                isinstance(func, ast.Attribute)
                                and isinstance(func.value, ast.Name)
                                and func.value.id == queue_var
                            ):
                                if func.attr == "popleft":
                                    has_pop_front = True
                                    pop_line = child.lineno
                                elif func.attr == "pop" and len(child.args) == 1:
                                    arg = child.args[0]
                                    if isinstance(arg, ast.Constant) and arg.value == 0:
                                        has_pop_front = True
                                        pop_line = child.lineno
                                elif func.attr in ("append", "extend"):
                                    has_append = True
                                    append_line = child.lineno

                    if has_pop_front and has_append:
                        self.found_bfs = True
                        self.confidence = 0.9
                        init_line = getattr(self.initialized_queues[queue_var], "lineno", 0)
                        self.evidence.append(
                            f"Line {node.lineno}: Found BFS graph traversal pattern using queue "
                            f"'{queue_var}' initialized at line {init_line}. "
                            f"Front pop at line {pop_line}, node expansion at line {append_line}."
                        )

                self.generic_visit(node)

            def _get_loop_test_var(self, test: ast.expr) -> str | None:
                if isinstance(test, ast.Name):
                    return test.id
                if isinstance(test, ast.Call) and (
                    isinstance(test.func, ast.Name)
                    and test.func.id == "len"
                    and len(test.args) == 1
                ):
                    arg = test.args[0]
                    if isinstance(arg, ast.Name):
                        return arg.id
                if isinstance(test, ast.Compare):
                    left = test.left
                    if (
                        isinstance(left, ast.Call)
                        and isinstance(left.func, ast.Name)
                        and left.func.id == "len"
                        and len(left.args) == 1
                    ):
                        arg = left.args[0]
                        if isinstance(arg, ast.Name):
                            return arg.id
                    for comp in test.comparators:
                        if (
                            isinstance(comp, ast.Call)
                            and isinstance(comp.func, ast.Name)
                            and comp.func.id == "len"
                            and len(comp.args) == 1
                        ):
                            arg = comp.args[0]
                            if isinstance(arg, ast.Name):
                                return arg.id
                return None

        visitor = BFSVisitor()
        visitor.visit(code_ast)

        if visitor.found_bfs:
            best_confidence = visitor.confidence
            evidence = visitor.evidence
            return PatternMatch(
                pattern=AlgorithmPattern.BFS,
                confidence=round(best_confidence, 2),
                evidence=evidence,
            )

        return None
