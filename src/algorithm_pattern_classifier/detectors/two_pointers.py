import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class TwoPointersVisitor(ast.NodeVisitor):
    """Visitor to traverse AST and look for two pointer patterns."""

    def __init__(self) -> None:
        self.parent_stack: list[ast.AST] = []
        self.statements_stack: list[list[ast.stmt]] = []
        self.best_confidence: float = 0.0
        self.best_evidence: list[str] = []

    def visit_Module(self, node: ast.Module) -> None:
        self.parent_stack.append(node)
        self.statements_stack.append(node.body)
        self.generic_visit(node)
        self.statements_stack.pop()
        self.parent_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.parent_stack.append(node)
        self.statements_stack.append(node.body)
        self.generic_visit(node)
        self.statements_stack.pop()
        self.parent_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.parent_stack.append(node)
        self.statements_stack.append(node.body)
        self.generic_visit(node)
        self.statements_stack.pop()
        self.parent_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.parent_stack.append(node)
        self.statements_stack.append(node.body)
        self.generic_visit(node)
        self.statements_stack.pop()
        self.parent_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        # Check if loop condition compares two variables
        if (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Name)
        ):
            var1 = node.test.left.id
            var2 = node.test.comparators[0].id
            op = node.test.ops[0]

            # Loop condition checks left < right (or <=, >, >=, !=)
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.NotEq)):
                loop_confidence = 0.4
                loop_evidence = [
                    f"Line {node.lineno}: Loop condition compares variables '{var1}' and '{var2}'."
                ]

                # Rule 1: Both pointers initialized near each other (+0.2)
                # We look at preceding statements in the current block
                if self.statements_stack:
                    current_body = self.statements_stack[-1]
                    try:
                        loop_idx = current_body.index(node)
                        preceding = current_body[:loop_idx]
                    except ValueError:
                        preceding = []

                    assigned_vars = self._get_assigned_vars(preceding)

                    # Also check function arguments if the parent is a function
                    if self.parent_stack:
                        parent = self.parent_stack[-1]
                        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            arg_names: list[str] = []
                            if hasattr(parent.args, "posonlyargs"):
                                arg_names.extend(a.arg for a in parent.args.posonlyargs)
                            if parent.args.args:
                                arg_names.extend(a.arg for a in parent.args.args)
                            if parent.args.kwonlyargs:
                                arg_names.extend(a.arg for a in parent.args.kwonlyargs)
                            if parent.args.vararg:
                                arg_names.append(parent.args.vararg.arg)
                            if parent.args.kwarg:
                                arg_names.append(parent.args.kwarg.arg)

                            assigned_vars.update(arg_names)

                    if var1 in assigned_vars and var2 in assigned_vars:
                        loop_confidence += 0.2
                        loop_evidence.append(
                            f"Line {node.lineno}: Both pointer variables "
                            f"'{var1}' and '{var2}' are initialized before the loop."
                        )

                # Find updates (Rule 2 and Rule 3)
                v1_updated, v2_updated, v1_classic, v2_classic = self._find_updates(
                    node, var1, var2
                )

                # Rule 2: Both pointers must be updated inside the loop (Gate)
                if v1_updated and v2_updated:
                    loop_confidence += 0.2
                    loop_evidence.append(
                        f"Line {node.lineno}: Both pointer variables "
                        f"'{var1}' and '{var2}' are updated inside the loop."
                    )

                    # Rule 3: Classic/Arithmetic update bonus (+0.2)
                    if v1_classic or v2_classic:
                        loop_confidence += 0.2
                        loop_evidence.append(
                            f"Line {node.lineno}: At least one pointer variable "
                            f"has an arithmetic update (increment/decrement)."
                        )
                else:
                    loop_confidence = 0.0

                if loop_confidence > self.best_confidence:
                    self.best_confidence = loop_confidence
                    self.best_evidence = loop_evidence

        self.parent_stack.append(node)
        self.statements_stack.append(node.body)
        self.generic_visit(node)
        self.statements_stack.pop()
        self.parent_stack.pop()

    def _get_assigned_vars(self, statements: list[ast.stmt]) -> set[str]:
        assigned: set[str] = set()
        stack: list[ast.AST] = list(statements)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                assigned.add(elt.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assigned.add(node.target.id)

            for child in ast.iter_child_nodes(node):
                stack.append(child)
        return assigned

    def _find_updates(
        self, loop_node: ast.AST, var1: str, var2: str
    ) -> tuple[bool, bool, bool, bool]:
        var1_updated = False
        var2_updated = False
        var1_classic = False
        var2_classic = False

        for child in ast.walk(loop_node):
            if child == loop_node:
                continue

            if self._is_swap(child, var1, var2):
                continue

            if self._is_target(child, var1):
                var1_updated = True
                if self._is_classic_update(child, var1):
                    var1_classic = True
            if self._is_target(child, var2):
                var2_updated = True
                if self._is_classic_update(child, var2):
                    var2_classic = True

        return var1_updated, var2_updated, var1_classic, var2_classic

    def _is_swap(self, node: ast.AST, var1: str, var2: str) -> bool:
        if not isinstance(node, ast.Assign):
            return False
        for target in node.targets:
            if not isinstance(target, (ast.Tuple, ast.List)):
                continue
            if not isinstance(node.value, (ast.Tuple, ast.List)):
                continue
            t_ids = [t.id for t in target.elts if isinstance(t, ast.Name)]
            v_ids = [v.id for v in node.value.elts if isinstance(v, ast.Name)]
            if set(t_ids) == {var1, var2} and set(v_ids) == {var1, var2}:
                return True
        return False

    def _is_target(self, node: ast.AST, var_name: str) -> bool:
        if isinstance(node, ast.AugAssign):
            return isinstance(node.target, ast.Name) and node.target.id == var_name
        if isinstance(node, ast.AnnAssign):
            return isinstance(node.target, ast.Name) and node.target.id == var_name
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return True
                if isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name) and elt.id == var_name:
                            return True
        return False

    def _is_classic_update(self, node: ast.AST, var_name: str) -> bool:
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == var_name
            and isinstance(node.op, (ast.Add, ast.Sub))
        ):
            return True

        if not isinstance(node, ast.Assign):
            return False

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == var_name:
                val = node.value
                if isinstance(val, ast.BinOp) and isinstance(val.op, (ast.Add, ast.Sub)):
                    left_match = isinstance(val.left, ast.Name) and val.left.id == var_name
                    right_match = isinstance(val.right, ast.Name) and val.right.id == var_name
                    if left_match or right_match:
                        return True
            elif (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(target.elts) == len(node.value.elts)
            ):
                for i, t_elt in enumerate(target.elts):
                    if isinstance(t_elt, ast.Name) and t_elt.id == var_name:
                        val_elt = node.value.elts[i]
                        if isinstance(val_elt, ast.BinOp) and isinstance(
                            val_elt.op, (ast.Add, ast.Sub)
                        ):
                            left_match = (
                                isinstance(val_elt.left, ast.Name) and val_elt.left.id == var_name
                            )
                            right_match = (
                                isinstance(val_elt.right, ast.Name) and val_elt.right.id == var_name
                            )
                            if left_match or right_match:
                                return True
        return False


class TwoPointersDetector(BaseDetector):
    """Detector for the Two Pointers algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.TWO_POINTERS

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Parse AST and detect Two Pointers pattern.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        visitor = TwoPointersVisitor()
        visitor.visit(code_ast)

        if visitor.best_confidence > 0.0:
            return PatternMatch(
                pattern=AlgorithmPattern.TWO_POINTERS,
                confidence=round(visitor.best_confidence, 2),
                evidence=visitor.best_evidence,
            )

        return None
