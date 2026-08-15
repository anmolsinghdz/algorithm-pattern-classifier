import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


def _get_target_names(target: ast.AST) -> list[str]:
    """Recursively extract all identifier names from an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_get_target_names(elt))
        return names
    return []


def _is_var_increment_binop(node: ast.AST, var_name: str) -> bool:
    """Check if AST expression is `var_name + ...` or `... + var_name`."""
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and (
            (isinstance(node.left, ast.Name) and node.left.id == var_name)
            or (isinstance(node.right, ast.Name) and node.right.id == var_name)
        )
    )


def _is_var_decrement_binop(node: ast.AST, var_name: str) -> bool:
    """Check if AST expression is `var_name - ...`."""
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and isinstance(node.left, ast.Name)
        and node.left.id == var_name
    )


def _find_incremented_vars(node: ast.AST) -> set[str]:
    """Find variables incremented in the AST node."""
    incremented: set[str] = set()

    class IncrementVisitor(ast.NodeVisitor):
        def visit_AugAssign(self, aug_node: ast.AugAssign) -> None:
            if isinstance(aug_node.target, ast.Name) and isinstance(aug_node.op, ast.Add):
                incremented.add(aug_node.target.id)
            self.generic_visit(aug_node)

        def visit_Assign(self, assign_node: ast.Assign) -> None:
            for target in assign_node.targets:
                if isinstance(target, ast.Name) and _is_var_increment_binop(
                    assign_node.value, target.id
                ):
                    incremented.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                    assign_node.value, (ast.Tuple, ast.List)
                ):
                    for t_elt, v_elt in zip(target.elts, assign_node.value.elts, strict=False):
                        if isinstance(t_elt, ast.Name) and _is_var_increment_binop(v_elt, t_elt.id):
                            incremented.add(t_elt.id)
            self.generic_visit(assign_node)

    IncrementVisitor().visit(node)
    return incremented


def _find_decremented_vars(node: ast.AST) -> set[str]:
    """Find variables decremented in the AST node."""
    decremented: set[str] = set()

    class DecrementVisitor(ast.NodeVisitor):
        def visit_AugAssign(self, aug_node: ast.AugAssign) -> None:
            if isinstance(aug_node.target, ast.Name) and isinstance(aug_node.op, ast.Sub):
                decremented.add(aug_node.target.id)
            self.generic_visit(aug_node)

        def visit_Assign(self, assign_node: ast.Assign) -> None:
            for target in assign_node.targets:
                if isinstance(target, ast.Name) and _is_var_decrement_binop(
                    assign_node.value, target.id
                ):
                    decremented.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                    assign_node.value, (ast.Tuple, ast.List)
                ):
                    for t_elt, v_elt in zip(target.elts, assign_node.value.elts, strict=False):
                        if isinstance(t_elt, ast.Name) and _is_var_decrement_binop(v_elt, t_elt.id):
                            decremented.add(t_elt.id)
            self.generic_visit(assign_node)

    DecrementVisitor().visit(node)
    return decremented


def _find_modified_vars(node: ast.AST) -> set[str]:
    """Find all variable names assigned or modified in the AST node."""
    modified: set[str] = set()

    class ModifyVisitor(ast.NodeVisitor):
        def visit_Assign(self, assign_node: ast.Assign) -> None:
            for target in assign_node.targets:
                modified.update(_get_target_names(target))
            self.generic_visit(assign_node)

        def visit_AugAssign(self, aug_node: ast.AugAssign) -> None:
            if isinstance(aug_node.target, ast.Name):
                modified.add(aug_node.target.id)
            self.generic_visit(aug_node)

    ModifyVisitor().visit(node)
    return modified


def _find_outer_incremented_vars(while_node: ast.While) -> set[str]:
    """Find variables incremented in a while loop body, excluding nested loops."""
    incremented: set[str] = set()

    class OuterIncrementVisitor(ast.NodeVisitor):
        def visit_While(self, node: ast.While) -> None:
            if node is while_node:
                self.generic_visit(node)
            # Skip nested while loops

        def visit_For(self, _node: ast.For) -> None:
            # Skip nested for loops
            pass

        def visit_AugAssign(self, aug_node: ast.AugAssign) -> None:
            if isinstance(aug_node.target, ast.Name) and isinstance(aug_node.op, ast.Add):
                incremented.add(aug_node.target.id)
            self.generic_visit(aug_node)

        def visit_Assign(self, assign_node: ast.Assign) -> None:
            for target in assign_node.targets:
                if isinstance(target, ast.Name) and _is_var_increment_binop(
                    assign_node.value, target.id
                ):
                    incremented.add(target.id)
            self.generic_visit(assign_node)

    OuterIncrementVisitor().visit(while_node)
    return incremented


def _find_pointer_vars(test_node: ast.AST, known_pointers: set[str]) -> set[str]:
    """Find pointer variables by expanding known pointers with direct boundary comparisons."""
    pointers = set(known_pointers)
    for node in ast.walk(test_node):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            for comp in node.comparators:
                if isinstance(comp, ast.Name):
                    if node.left.id in known_pointers:
                        pointers.add(comp.id)
                    elif comp.id in known_pointers:
                        pointers.add(node.left.id)
    return pointers


def _is_dynamic_condition(test_node: ast.AST, dynamic_vars: set[str]) -> bool:
    """Check if loop condition relies on dynamic metrics (collections, counters, etc.)."""
    for sub in ast.walk(test_node):
        # 1. Calls to functions/methods (e.g. len(counts), sum(w), is_valid())
        if isinstance(sub, ast.Call):
            return True
        # 2. Subscript access (e.g. counts[char] > 1, freq[s[left]])
        if isinstance(sub, ast.Subscript):
            return True
        # 3. Attribute access (e.g. window.size > k)
        if isinstance(sub, ast.Attribute):
            return True

    # 4. Check for dynamic accumulator/metric variables modified in the loop
    names = {
        n.id
        for n in ast.walk(test_node)
        if isinstance(n, ast.Name) and n.id not in {"True", "False", "None"}
    }
    if names.intersection(dynamic_vars):
        return True

    # 5. Check if sub-elements of BoolOp or UnaryOp are dynamic
    if isinstance(test_node, ast.BoolOp):
        return any(_is_dynamic_condition(val, dynamic_vars) for val in test_node.values)
    if isinstance(test_node, ast.UnaryOp):
        return _is_dynamic_condition(test_node.operand, dynamic_vars)

    return False


class SlidingWindowDetector(BaseDetector):
    """Detector for the Sliding Window algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.SLIDING_WINDOW

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Detect evidence of the Sliding Window pattern in parsed AST.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """

        evidence: list[str] = []
        max_confidence = 0.0

        class SlidingWindowVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_sliding_window = False
                self.evidence: list[str] = []
                self.confidence = 0.0

            def _check_nested_while(
                self,
                outer_node: ast.For | ast.While,
                expansion_vars: list[str],
                body_stmts: list[ast.stmt],
            ) -> bool:
                """Check for nested while loops with dynamic shrink condition."""
                nested_whiles: list[ast.While] = []
                for stmt in body_stmts:
                    for n in ast.walk(stmt):
                        if isinstance(n, ast.While) and n is not outer_node:
                            nested_whiles.append(n)

                all_modified = _find_modified_vars(outer_node)

                for while_node in nested_whiles:
                    inner_incremented = _find_incremented_vars(while_node)
                    shrink_candidates = [v for v in inner_incremented if v not in expansion_vars]
                    if not shrink_candidates:
                        continue

                    for shrink_var in shrink_candidates:
                        known_pointers = set(expansion_vars) | {shrink_var}
                        pointer_vars = _find_pointer_vars(while_node.test, known_pointers)
                        dynamic_vars = all_modified - pointer_vars
                        if _is_dynamic_condition(while_node.test, dynamic_vars):
                            exp_var = expansion_vars[0] if expansion_vars else "expansion_pointer"
                            self.found_sliding_window = True
                            self.confidence = max(self.confidence, 0.95)
                            evidence_msg = (
                                f"Line {outer_node.lineno}: found sliding window loop with "
                                f"expansion pointer '{exp_var}' and dynamic shrink pointer "
                                f"'{shrink_var}' (inner while loop at line {while_node.lineno})."
                            )
                            if evidence_msg not in self.evidence:
                                self.evidence.append(evidence_msg)
                            return True
                return False

            def _check_metric_visitor(
                self,
                node: ast.For | ast.While,
                end_var: str,
                updated_vars: set[str],
            ) -> None:
                """Check for subtraction or joint subscripting window metrics."""
                has_window_metric = False
                detected_start_var = None

                class MetricVisitor(ast.NodeVisitor):
                    def __init__(self) -> None:
                        self.start_vars_in_subtraction: set[str] = set()
                        self.subscript_vars: dict[str, set[str]] = {}

                    def visit_BinOp(self, bin_node: ast.BinOp) -> None:
                        if isinstance(bin_node.op, ast.Sub):
                            left_id = (
                                bin_node.left.id if isinstance(bin_node.left, ast.Name) else None
                            )
                            right_id = (
                                bin_node.right.id if isinstance(bin_node.right, ast.Name) else None
                            )
                            if left_id == end_var and right_id in updated_vars:
                                assert right_id is not None
                                self.start_vars_in_subtraction.add(right_id)
                            elif right_id == end_var and left_id in updated_vars:
                                assert left_id is not None
                                self.start_vars_in_subtraction.add(left_id)
                        self.generic_visit(bin_node)

                    def visit_Subscript(self, sub_node: ast.Subscript) -> None:
                        # Handle slicing like arr[start:end]
                        if isinstance(sub_node.slice, ast.Slice):
                            lower_id = (
                                sub_node.slice.lower.id
                                if isinstance(sub_node.slice.lower, ast.Name)
                                else None
                            )
                            upper_id = (
                                sub_node.slice.upper.id
                                if isinstance(sub_node.slice.upper, ast.Name)
                                else None
                            )
                            if lower_id in updated_vars and upper_id == end_var:
                                assert lower_id is not None
                                self.start_vars_in_subtraction.add(lower_id)
                        # Track indexing access to detect joint subscripting
                        elif isinstance(sub_node.value, ast.Name):
                            seq_name = sub_node.value.id
                            if isinstance(sub_node.slice, ast.Name):
                                self.subscript_vars.setdefault(seq_name, set()).add(
                                    sub_node.slice.id
                                )
                            elif (
                                isinstance(sub_node.slice, ast.BinOp)
                                and isinstance(sub_node.slice.op, ast.Sub)
                                and isinstance(sub_node.slice.left, ast.Name)
                                and sub_node.slice.left.id == end_var
                            ):
                                right = sub_node.slice.right
                                if isinstance(right, ast.Name):
                                    self.subscript_vars.setdefault(seq_name, set()).add(
                                        f"sub-{right.id}"
                                    )
                                elif isinstance(right, ast.Constant) and isinstance(
                                    right.value, (int, str)
                                ):
                                    self.subscript_vars.setdefault(seq_name, set()).add(
                                        f"sub-{right.value}"
                                    )
                        self.generic_visit(sub_node)

                metric_visitor = MetricVisitor()
                metric_visitor.visit(node)

                # Check subtraction criteria
                if metric_visitor.start_vars_in_subtraction:
                    has_window_metric = True
                    detected_start_var = next(iter(metric_visitor.start_vars_in_subtraction))

                # Check joint subscripting criteria
                if not has_window_metric:
                    for _seq, indices in metric_visitor.subscript_vars.items():
                        if end_var in indices:
                            # Check dynamic start variables (variables updated in the loop)
                            common_starts = indices.intersection(updated_vars)
                            if common_starts:
                                has_window_metric = True
                                detected_start_var = next(iter(common_starts))
                                break
                            # Check fixed window size subtraction elements (e.g. sub-k or sub-3)
                            sub_starts = [idx for idx in indices if idx.startswith("sub-")]
                            if sub_starts:
                                has_window_metric = True
                                # Extract variable name / constant value from prefix
                                detected_start_var = sub_starts[0].split("-", 1)[1]
                                break

                if has_window_metric and detected_start_var:
                    self.found_sliding_window = True
                    self.confidence = max(self.confidence, 0.95)
                    evidence_msg = (
                        f"Line {node.lineno}: found sliding window loop with "
                        f"expansion pointer '{end_var}' and boundary offset/pointer "
                        f"'{detected_start_var}'."
                    )
                    if evidence_msg not in self.evidence:
                        self.evidence.append(evidence_msg)

            def visit_For(self, node: ast.For) -> None:
                expansion_vars = _get_target_names(node.target)
                if not expansion_vars:
                    self.generic_visit(node)
                    return

                end_var = expansion_vars[0]

                # Check 1: Nested while loop with dynamic shrink condition
                self._check_nested_while(node, expansion_vars, node.body)

                # Check 2: Subtraction / joint subscripting
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
                for ev in expansion_vars:
                    updated_vars.discard(ev)

                # Exclude variables updated in converging nested while loops (e.g. 3Sum)
                for stmt in node.body:
                    for n in ast.walk(stmt):
                        if isinstance(n, ast.While) and _find_decremented_vars(n):
                            for inc in _find_incremented_vars(n):
                                updated_vars.discard(inc)
                            for dec in _find_decremented_vars(n):
                                updated_vars.discard(dec)

                self._check_metric_visitor(node, end_var, updated_vars)

                self.generic_visit(node)

            def visit_While(self, node: ast.While) -> None:
                # If this while loop is a converging two-pointer loop, skip treating it as outer
                if _find_decremented_vars(node):
                    self.generic_visit(node)
                    return

                outer_incremented = _find_outer_incremented_vars(node)
                if outer_incremented:
                    expansion_vars = list(outer_incremented)
                    self._check_nested_while(node, expansion_vars, node.body)

                    # Also check subtraction / joint subscripting for while loops if present
                    end_var = expansion_vars[0]
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
                    for ev in expansion_vars:
                        updated_vars.discard(ev)

                    self._check_metric_visitor(node, end_var, updated_vars)

                self.generic_visit(node)

        visitor = SlidingWindowVisitor()
        visitor.visit(code_ast)

        if visitor.found_sliding_window:
            max_confidence = visitor.confidence
            evidence = visitor.evidence
            return PatternMatch(
                pattern=AlgorithmPattern.SLIDING_WINDOW,
                confidence=max_confidence,
                evidence=evidence,
            )

        return None
