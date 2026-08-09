import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


def _get_node_name(node: ast.AST) -> str | None:
    """Extract a canonical identifier string from Name or Attribute AST nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _get_subscript_base_name(node: ast.AST) -> str | None:
    """Extract the root container name from a Subscript chain."""
    curr = node
    while isinstance(curr, ast.Subscript):
        curr = curr.value
    return _get_node_name(curr)


def _get_index_offset(slice_node: ast.AST, loop_var: str | None = None) -> int | None:
    """Extract integer offset relative to loop variable or constant index."""
    if isinstance(slice_node, ast.Name):
        if loop_var is None or slice_node.id == loop_var:
            return 0
    elif (
        isinstance(slice_node, ast.BinOp)
        and isinstance(slice_node.left, ast.Name)
        and (loop_var is None or slice_node.left.id == loop_var)
        and isinstance(slice_node.right, ast.Constant)
        and isinstance(slice_node.right.value, int)
    ):
        if isinstance(slice_node.op, ast.Add):
            return slice_node.right.value
        if isinstance(slice_node.op, ast.Sub):
            return -slice_node.right.value
    elif (
        isinstance(slice_node, ast.UnaryOp)
        and isinstance(slice_node.op, ast.USub)
        and isinstance(slice_node.operand, ast.Constant)
        and isinstance(slice_node.operand.value, int)
    ):
        return -slice_node.operand.value
    return None


def _is_prior_index(
    source_slice: ast.AST,
    target_slice: ast.AST | None = None,
    loop_var: str | None = None,
) -> bool:
    """Check if source_slice accesses a prior index relative to target_slice or offset."""
    source_offset = _get_index_offset(source_slice, loop_var)
    if source_offset is not None:
        if target_slice is not None:
            target_offset = _get_index_offset(target_slice, loop_var)
            if target_offset is not None:
                return target_offset > source_offset
        return source_offset < 0

    return False


def _extract_subscripts_from_expr(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """Recursively extract (base_name, slice_node) pairs from an expression."""
    results: list[tuple[str, ast.AST]] = []

    class SubscriptCollector(ast.NodeVisitor):
        def visit_Subscript(self, sub_node: ast.Subscript) -> None:
            base = _get_subscript_base_name(sub_node)
            if base:
                results.append((base, sub_node.slice))
            self.generic_visit(sub_node)

    SubscriptCollector().visit(node)
    return results


class PrefixSumDetector(BaseDetector):
    """Detector for the Prefix Sum and Difference Array algorithmic design patterns."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.PREFIX_SUM

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Detect evidence of Prefix Sum pattern in parsed AST.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        evidence: list[str] = []

        has_accumulative_loop = False
        has_running_sum_append = False
        has_inplace_prefix_sum = False
        has_itertools_accumulate = False
        has_subarray_query = False
        has_prefix_hash_map = False
        has_difference_array = False

        class PrefixSumVisitor(ast.NodeVisitor):
            def visit_Call(self, call_node: ast.Call) -> None:
                nonlocal has_itertools_accumulate
                # Check for itertools.accumulate or accumulate(...)
                func_name = _get_node_name(call_node.func)
                if func_name in ("accumulate", "itertools.accumulate"):
                    has_itertools_accumulate = True
                    evidence.append(
                        f"Line {call_node.lineno}: found accumulate() call for prefix sum."
                    )
                self.generic_visit(call_node)

            def visit_BinOp(self, bin_node: ast.BinOp) -> None:
                nonlocal has_subarray_query
                # Check for subarray sum subtraction: prefix[right] - prefix[left-1]
                if isinstance(bin_node.op, ast.Sub):
                    left_subs = _extract_subscripts_from_expr(bin_node.left)
                    right_subs = _extract_subscripts_from_expr(bin_node.right)

                    if left_subs and right_subs:
                        for l_base, _l_slice in left_subs:
                            for r_base, _r_slice in right_subs:
                                if l_base == r_base:
                                    has_subarray_query = True
                                    evidence.append(
                                        f"Line {bin_node.lineno}: found subarray subtraction query "
                                        f"on sequence '{l_base}'."
                                    )
                                    break
                self.generic_visit(bin_node)

            def visit_For(self, loop_node: ast.For) -> None:
                self._check_loop(loop_node)
                self.generic_visit(loop_node)

            def visit_While(self, loop_node: ast.While) -> None:
                self._check_loop(loop_node)
                self.generic_visit(loop_node)

            def _check_loop(self, loop_node: ast.For | ast.While) -> None:
                nonlocal has_accumulative_loop, has_running_sum_append
                nonlocal has_inplace_prefix_sum, has_prefix_hash_map
                nonlocal has_difference_array

                loop_var: str | None = None
                if isinstance(loop_node, ast.For) and isinstance(loop_node.target, ast.Name):
                    loop_var = loop_node.target.id

                # Track augmented/updated variables for running sum tracking
                accumulated_vars: set[str] = set()
                appended_targets: set[str] = set()
                diff_add_targets: set[str] = set()
                diff_sub_targets: set[str] = set()

                class LoopBodyVisitor(ast.NodeVisitor):
                    def visit_AugAssign(self, node: ast.AugAssign) -> None:
                        # 1. Running sum accumulation: running_sum += num
                        if isinstance(node.op, ast.Add) and isinstance(node.target, ast.Name):
                            accumulated_vars.add(node.target.id)

                        # 2. In-place prefix sum: nums[i] += nums[i-1]
                        if isinstance(node.op, ast.Add) and isinstance(node.target, ast.Subscript):
                            tgt_base = _get_subscript_base_name(node.target)
                            if (
                                tgt_base
                                and isinstance(node.value, ast.Subscript)
                                and _get_subscript_base_name(node.value) == tgt_base
                                and _is_prior_index(
                                    node.value.slice, node.target.slice, loop_var
                                )
                            ):
                                nonlocal has_inplace_prefix_sum
                                has_inplace_prefix_sum = True
                                evidence.append(
                                    f"Line {node.lineno}: found in-place accumulative prefix sum "
                                    f"update on '{tgt_base}'."
                                )

                        # 3. Difference array updates: diff[l] += val, diff[r] -= val
                        if isinstance(node.target, ast.Subscript):
                            base = _get_subscript_base_name(node.target)
                            if base:
                                if isinstance(node.op, ast.Add):
                                    diff_add_targets.add(base)
                                elif isinstance(node.op, ast.Sub):
                                    diff_sub_targets.add(base)

                        self.generic_visit(node)

                    def visit_Assign(self, node: ast.Assign) -> None:
                        for target in node.targets:
                            # Running sum via assign: running_sum = running_sum + num
                            if (
                                isinstance(target, ast.Name)
                                and isinstance(node.value, ast.BinOp)
                                and isinstance(node.value.op, ast.Add)
                            ):
                                left_name = _get_node_name(node.value.left)
                                right_name = _get_node_name(node.value.right)
                                if target.id in (left_name, right_name):
                                    accumulated_vars.add(target.id)

                            # Tabular prefix sum: prefix[i] = prefix[i-1] + nums[i]
                            # or prefix[i+1] = prefix[i] + nums[i]
                            if isinstance(target, ast.Subscript):
                                tgt_base = _get_subscript_base_name(target)
                                if (
                                    tgt_base
                                    and isinstance(node.value, ast.BinOp)
                                    and isinstance(node.value.op, ast.Add)
                                ):
                                    # Check if one operand is prior entry of tgt_base
                                    left_sub = (
                                        node.value.left
                                        if isinstance(node.value.left, ast.Subscript)
                                        else None
                                    )
                                    right_sub = (
                                        node.value.right
                                        if isinstance(node.value.right, ast.Subscript)
                                        else None
                                    )

                                    # If both operands are prior entries of tgt_base,
                                    # this is Fibonacci / DP recurrence, not prefix sum.
                                    left_is_prior = (
                                        left_sub is not None
                                        and _get_subscript_base_name(left_sub) == tgt_base
                                        and _is_prior_index(
                                            left_sub.slice, target.slice, loop_var
                                        )
                                    )
                                    right_is_prior = (
                                        right_sub is not None
                                        and _get_subscript_base_name(right_sub) == tgt_base
                                        and _is_prior_index(
                                            right_sub.slice, target.slice, loop_var
                                        )
                                    )

                                    if left_is_prior and right_is_prior:
                                        # DP recurrence, skip
                                        pass
                                    elif left_is_prior or right_is_prior:
                                        other_operand = right_sub if left_is_prior else left_sub

                                        # Check if target is i or i+1
                                        is_valid_target = (
                                            _get_node_name(target.slice) == loop_var
                                            or (
                                                isinstance(target.slice, ast.BinOp)
                                                and isinstance(target.slice.op, ast.Add)
                                                and _get_node_name(target.slice.left) == loop_var
                                            )
                                            or isinstance(target.slice, ast.Constant)
                                        )

                                        if is_valid_target:
                                            # If other operand is tgt_base[i], it's in-place
                                            if (
                                                isinstance(other_operand, ast.Subscript)
                                                and _get_subscript_base_name(other_operand)
                                                == tgt_base
                                                and _get_node_name(other_operand.slice) == loop_var
                                            ):
                                                nonlocal has_inplace_prefix_sum
                                                has_inplace_prefix_sum = True
                                                evidence.append(
                                                    f"Line {node.lineno}: found in-place prefix "
                                                    f"sum accumulation on '{tgt_base}'."
                                                )
                                            else:
                                                nonlocal has_accumulative_loop
                                                has_accumulative_loop = True
                                                evidence.append(
                                                    f"Line {node.lineno}: found accumulative "
                                                    f"prefix sum loop building '{tgt_base}'."
                                                )

                        self.generic_visit(node)

                    def visit_Call(self, node: ast.Call) -> None:
                        # Check for prefix.append(running_sum) or prefix.append(prefix[-1] + num)
                        if isinstance(node.func, ast.Attribute) and node.func.attr == "append":
                            base_name = _get_node_name(node.func.value)
                            if base_name and node.args:
                                arg = node.args[0]
                                arg_name = _get_node_name(arg)
                                if arg_name and arg_name in accumulated_vars:
                                    appended_targets.add(base_name)
                                elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                                    # e.g., prefix.append(prefix[-1] + num)
                                    left_sub = (
                                        arg.left if isinstance(arg.left, ast.Subscript) else None
                                    )
                                    right_sub = (
                                        arg.right if isinstance(arg.right, ast.Subscript) else None
                                    )
                                    for sub in (left_sub, right_sub):
                                        if (
                                            sub
                                            and _get_subscript_base_name(sub) == base_name
                                            and _is_prior_index(sub.slice, loop_var=loop_var)
                                        ):
                                            appended_targets.add(base_name)

                        self.generic_visit(node)

                    def visit_Compare(self, node: ast.Compare) -> None:
                        # Check for curr_sum - k in prefix_map / seen
                        if (
                            len(node.ops) == 1
                            and isinstance(node.ops[0], (ast.In, ast.NotIn))
                            and isinstance(node.left, ast.BinOp)
                            and isinstance(node.left.op, ast.Sub)
                        ):
                            left_term = _get_node_name(node.left.left)
                            if left_term and left_term in accumulated_vars:
                                nonlocal has_prefix_hash_map
                                has_prefix_hash_map = True
                                comp_name = _get_node_name(node.comparators[0]) or "map"
                                evidence.append(
                                    f"Line {node.lineno}: found prefix sum lookup "
                                    f"'{left_term} - k' in '{comp_name}'."
                                )
                        self.generic_visit(node)

                LoopBodyVisitor().visit(loop_node)

                if appended_targets:
                    has_running_sum_append = True
                    for tgt in appended_targets:
                        evidence.append(
                            f"Line {loop_node.lineno}: found running sum accumulation and append "
                            f"into '{tgt}'."
                        )

                common_diff = diff_add_targets.intersection(diff_sub_targets)
                if common_diff:
                    has_difference_array = True
                    for diff_name in common_diff:
                        evidence.append(
                            f"Line {loop_node.lineno}: found difference array paired updates "
                            f"on '{diff_name}'."
                        )

        visitor = PrefixSumVisitor()
        visitor.visit(code_ast)

        confidence_score = 0.0

        # Calculate confidence
        if (
            has_prefix_hash_map
            or (has_difference_array and (has_accumulative_loop or has_running_sum_append))
            or (
                (
                    has_accumulative_loop
                    or has_running_sum_append
                    or has_inplace_prefix_sum
                    or has_itertools_accumulate
                )
                and has_subarray_query
            )
        ):
            confidence_score = 0.95
        elif (
            has_accumulative_loop
            or has_inplace_prefix_sum
            or has_running_sum_append
            or has_itertools_accumulate
            or has_difference_array
        ):
            confidence_score = 0.90
        elif has_subarray_query:
            confidence_score = 0.85

        if confidence_score > 0.0:
            return PatternMatch(
                pattern=self.pattern,
                confidence=confidence_score,
                evidence=evidence,
            )

        return None
