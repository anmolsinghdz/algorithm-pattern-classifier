import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


def _get_func_param_names(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Extract formal parameter names of a function, excluding self and cls."""
    params: set[str] = set()
    if hasattr(func_node.args, "posonlyargs"):
        params.update(a.arg for a in func_node.args.posonlyargs)
    params.update(a.arg for a in func_node.args.args)
    params.update(a.arg for a in func_node.args.kwonlyargs)
    if func_node.args.vararg:
        params.add(func_node.args.vararg.arg)
    if func_node.args.kwarg:
        params.add(func_node.args.kwarg.arg)
    params.discard("self")
    params.discard("cls")
    return params


def _is_navigational_arg(arg: ast.expr, func_args: set[str]) -> bool:
    """Check if argument passed in recursive call represents tree/graph navigation."""
    # 1. Attribute access like node.left, node.right, curr.next, child.val
    if isinstance(arg, ast.Attribute):
        return True
    # 2. Variable from neighbor loop or not in original parameter names (e.g. neighbor, nxt)
    if isinstance(arg, ast.Name) and arg.id not in func_args:
        return True
    # 3. Expressions like r + 1, idx + 1, path + [node]
    return isinstance(
        arg,
        (
            ast.BinOp,
            ast.Subscript,
            ast.Call,
            ast.Constant,
            ast.UnaryOp,
            ast.Tuple,
            ast.List,
            ast.Dict,
        ),
    )


def _is_recursive_dfs_func(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[bool, list[ast.Call]]:
    """Check if a function recursively calls itself in a graph/tree traversal pattern."""
    func_name = func_node.name
    recursive_calls: list[ast.Call] = []

    class RecursionVisitor(ast.NodeVisitor):
        def visit_Call(self, call_node: ast.Call) -> None:
            is_name_call = isinstance(call_node.func, ast.Name) and call_node.func.id == func_name
            is_self_call = (
                isinstance(call_node.func, ast.Attribute)
                and isinstance(call_node.func.value, ast.Name)
                and call_node.func.value.id == "self"
                and call_node.func.attr == func_name
            )
            if is_name_call or is_self_call:
                recursive_calls.append(call_node)
            self.generic_visit(call_node)

    RecursionVisitor().visit(func_node)
    if not recursive_calls:
        return False, []

    func_args = _get_func_param_names(func_node)
    valid_calls: list[ast.Call] = []

    for call in recursive_calls:
        is_in_loop_or_branch = False
        for stmt in ast.walk(func_node):
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                for desc in ast.walk(stmt):
                    if desc is call:
                        is_in_loop_or_branch = True
                        break
            if is_in_loop_or_branch:
                break

        modified_arg_passed = any(_is_navigational_arg(arg, func_args) for arg in call.args)

        if (is_in_loop_or_branch or len(recursive_calls) >= 2) and (
            modified_arg_passed or len(func_args) == 0
        ):
            valid_calls.append(call)

    return (len(valid_calls) > 0), valid_calls


class DFSDetector(BaseDetector):
    """Detector for the Depth-First Search (DFS) algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.DFS

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Parse AST and detect DFS pattern.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        evidence: list[str] = []
        best_confidence = 0.0

        class DFSVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_dfs = False
                self.evidence: list[str] = []
                self.confidence = 0.0
                self.initialized_stacks: dict[str, ast.AST] = {}

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                old_stacks = self.initialized_stacks.copy()
                self._analyze_function(node)
                self.generic_visit(node)
                self.initialized_stacks = old_stacks

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                old_stacks = self.initialized_stacks.copy()
                self._analyze_function(node)
                self.generic_visit(node)
                self.initialized_stacks = old_stacks

            def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                # Find all stack/list initializations in this function body
                for child in node.body:
                    for sub_node in ast.walk(child):
                        if isinstance(sub_node, ast.Assign):
                            for target in sub_node.targets:
                                if isinstance(target, ast.Name) and isinstance(
                                    sub_node.value, ast.List
                                ):
                                    self.initialized_stacks[target.id] = sub_node
                        elif isinstance(sub_node, ast.AnnAssign) and (
                            isinstance(sub_node.target, ast.Name)
                            and sub_node.value
                            and isinstance(sub_node.value, ast.List)
                        ):
                            self.initialized_stacks[sub_node.target.id] = sub_node

                # Check 1: Direct recursive DFS in this function
                is_recur, recur_calls = _is_recursive_dfs_func(node)
                if is_recur:
                    self.found_dfs = True
                    self.confidence = max(self.confidence, 0.9)
                    first_call = recur_calls[0]
                    evidence_msg = (
                        f"Line {node.lineno}: Found recursive DFS pattern in "
                        f"function '{node.name}' with recursive call at line "
                        f"{first_call.lineno}."
                    )
                    if evidence_msg not in self.evidence:
                        self.evidence.append(evidence_msg)

                # Check 2: Nested helper functions and upward propagation
                self._check_nested_helpers(node)

            def _check_nested_helpers(
                self, outer_node: ast.FunctionDef | ast.AsyncFunctionDef
            ) -> None:
                outer_params = _get_func_param_names(outer_node)

                inner_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
                for child in outer_node.body:
                    for n in ast.walk(child):
                        if (
                            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and n is not outer_node
                            and n not in inner_funcs
                        ):
                            inner_funcs.append(n)

                for inner_func in inner_funcs:
                    is_recur, _ = _is_recursive_dfs_func(inner_func)
                    if not is_recur:
                        continue

                    # Search for invocations of inner_func in outer_node body
                    outer_calls: list[ast.Call] = []
                    for child in outer_node.body:
                        if child is inner_func:
                            continue
                        for n in ast.walk(child):
                            if isinstance(n, ast.Call) and (
                                (isinstance(n.func, ast.Name) and n.func.id == inner_func.name)
                                or (
                                    isinstance(n.func, ast.Attribute)
                                    and isinstance(n.func.value, ast.Name)
                                    and n.func.value.id == "self"
                                    and n.func.attr == inner_func.name
                                )
                            ):
                                outer_calls.append(n)

                    if outer_calls:
                        self.found_dfs = True
                        self.confidence = max(self.confidence, 0.95)
                        first_outer_call = outer_calls[0]
                        call_arg_names = {
                            arg.id for arg in first_outer_call.args if isinstance(arg, ast.Name)
                        }
                        passed_params = call_arg_names.intersection(outer_params)
                        if passed_params:
                            param_str = ", ".join(f"'{p}'" for p in sorted(passed_params))
                            evidence_msg = (
                                f"Line {outer_node.lineno}: Found recursive DFS pattern in "
                                f"function '{outer_node.name}' via helper '{inner_func.name}' "
                                f"(line {inner_func.lineno}) invoked at line "
                                f"{first_outer_call.lineno} with outer parameter(s) {param_str}."
                            )
                        else:
                            evidence_msg = (
                                f"Line {outer_node.lineno}: Found recursive DFS pattern in "
                                f"function '{outer_node.name}' via helper '{inner_func.name}' "
                                f"(line {inner_func.lineno}) invoked at line "
                                f"{first_outer_call.lineno}."
                            )
                        if evidence_msg not in self.evidence:
                            self.evidence.append(evidence_msg)

            def visit_While(self, node: ast.While) -> None:
                stack_var = self._get_loop_test_var(node.test)
                if stack_var and stack_var in self.initialized_stacks:
                    has_pop_back = False
                    has_append = False
                    pop_line = 0
                    append_line = 0

                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            func = child.func
                            if (
                                isinstance(func, ast.Attribute)
                                and isinstance(func.value, ast.Name)
                                and func.value.id == stack_var
                            ):
                                if func.attr == "pop":
                                    if len(child.args) == 0:
                                        has_pop_back = True
                                        pop_line = getattr(child, "lineno", 0)
                                    elif len(child.args) == 1:
                                        arg = child.args[0]
                                        if isinstance(arg, ast.UnaryOp) and isinstance(
                                            arg.op, ast.USub
                                        ):
                                            if (
                                                isinstance(arg.operand, ast.Constant)
                                                and arg.operand.value == 1
                                            ):
                                                has_pop_back = True
                                                pop_line = getattr(child, "lineno", 0)
                                        elif isinstance(arg, ast.Constant) and arg.value == -1:
                                            has_pop_back = True
                                            pop_line = child.lineno
                                elif func.attr in ("append", "extend"):
                                    has_append = True
                                    append_line = child.lineno

                    if has_pop_back and has_append:
                        self.found_dfs = True
                        self.confidence = max(self.confidence, 0.9)
                        init_line = getattr(self.initialized_stacks[stack_var], "lineno", 0)
                        evidence_msg = (
                            f"Line {node.lineno}: Found iterative DFS graph traversal pattern "
                            f"using stack '{stack_var}' initialized at line {init_line}. "
                            f"End pop at line {pop_line}, node expansion at line {append_line}."
                        )
                        if evidence_msg not in self.evidence:
                            self.evidence.append(evidence_msg)

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

        visitor = DFSVisitor()
        visitor.visit(code_ast)

        if visitor.found_dfs:
            best_confidence = visitor.confidence
            evidence = visitor.evidence
            return PatternMatch(
                pattern=AlgorithmPattern.DFS,
                confidence=round(best_confidence, 2),
                evidence=evidence,
            )

        return None
