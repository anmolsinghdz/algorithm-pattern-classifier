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


def _is_queue_init(value: ast.expr) -> bool:
    """Check if an AST expression initializes a queue."""
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


def _get_loop_test_var(test: ast.expr) -> str | None:
    """Extract queue variable name from while loop condition."""
    if isinstance(test, ast.Name):
        return test.id
    if isinstance(test, ast.Call) and (
        isinstance(test.func, ast.Name) and test.func.id == "len" and len(test.args) == 1
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


def _check_func_bfs(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    inherited_queues: dict[str, ast.AST] | None = None,
) -> tuple[bool, str, int, int, int]:
    """Check if a function body contains BFS queue traversal operations."""
    initialized_queues: dict[str, ast.AST] = inherited_queues.copy() if inherited_queues else {}

    for child in func_node.body:
        for sub_node in ast.walk(child):
            if isinstance(sub_node, ast.Assign):
                for target in sub_node.targets:
                    if isinstance(target, ast.Name) and _is_queue_init(sub_node.value):
                        initialized_queues[target.id] = sub_node
            elif isinstance(sub_node, ast.AnnAssign) and (
                isinstance(sub_node.target, ast.Name)
                and sub_node.value
                and _is_queue_init(sub_node.value)
            ):
                initialized_queues[sub_node.target.id] = sub_node

    for stmt in func_node.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.While):
                queue_var = _get_loop_test_var(node.test)
                if queue_var and queue_var in initialized_queues:
                    has_pop_front = False
                    has_append = False
                    pop_line = 0
                    append_line = 0

                    for item in ast.walk(node):
                        if isinstance(item, ast.Call):
                            func = item.func
                            if (
                                isinstance(func, ast.Attribute)
                                and isinstance(func.value, ast.Name)
                                and func.value.id == queue_var
                            ):
                                if func.attr == "popleft":
                                    has_pop_front = True
                                    pop_line = item.lineno
                                elif func.attr == "pop" and len(item.args) == 1:
                                    arg = item.args[0]
                                    if isinstance(arg, ast.Constant) and arg.value == 0:
                                        has_pop_front = True
                                        pop_line = item.lineno
                                elif func.attr in ("append", "extend"):
                                    has_append = True
                                    append_line = item.lineno

                    if has_pop_front and has_append:
                        init_line = getattr(initialized_queues[queue_var], "lineno", 0)
                        return True, queue_var, init_line, pop_line, append_line

    return False, "", 0, 0, 0


class BFSDetector(BaseDetector):
    """Detector for the Breadth-First Search (BFS) algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.BFS

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
                self._analyze_function(node)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._analyze_function(node)
                self.generic_visit(node)

            def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                old_queues = self.initialized_queues.copy()

                # Find all queue initializations in this function body
                for child in node.body:
                    for sub_node in ast.walk(child):
                        if isinstance(sub_node, ast.Assign):
                            for target in sub_node.targets:
                                if isinstance(target, ast.Name) and _is_queue_init(sub_node.value):
                                    self.initialized_queues[target.id] = sub_node
                        elif isinstance(sub_node, ast.AnnAssign) and (
                            isinstance(sub_node.target, ast.Name)
                            and sub_node.value
                            and _is_queue_init(sub_node.value)
                        ):
                            self.initialized_queues[sub_node.target.id] = sub_node

                # Check 1: Direct BFS in this function body
                found, q_var, init_line, pop_line, app_line = _check_func_bfs(
                    node, self.initialized_queues
                )
                if found:
                    self.found_bfs = True
                    self.confidence = max(self.confidence, 0.9)
                    evidence_msg = (
                        f"Line {node.lineno}: Found BFS graph traversal pattern using queue "
                        f"'{q_var}' initialized at line {init_line}. "
                        f"Front pop at line {pop_line}, node expansion at line {app_line}."
                    )
                    if evidence_msg not in self.evidence:
                        self.evidence.append(evidence_msg)

                # Check 2: Nested helper functions with BFS and upward propagation
                self._check_nested_helpers(node)

                self.initialized_queues = old_queues

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
                    found, _, _, _, _ = _check_func_bfs(inner_func, self.initialized_queues)
                    if not found:
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
                        self.found_bfs = True
                        self.confidence = max(self.confidence, 0.95)
                        first_outer_call = outer_calls[0]
                        call_arg_names = {
                            arg.id for arg in first_outer_call.args if isinstance(arg, ast.Name)
                        }
                        passed_params = call_arg_names.intersection(outer_params)
                        if passed_params:
                            param_str = ", ".join(f"'{p}'" for p in sorted(passed_params))
                            evidence_msg = (
                                f"Line {outer_node.lineno}: Found BFS graph traversal pattern in "
                                f"function '{outer_node.name}' via helper '{inner_func.name}' "
                                f"(line {inner_func.lineno}) invoked at line "
                                f"{first_outer_call.lineno} with outer parameter(s) {param_str}."
                            )
                        else:
                            evidence_msg = (
                                f"Line {outer_node.lineno}: Found BFS graph traversal pattern in "
                                f"function '{outer_node.name}' via helper '{inner_func.name}' "
                                f"(line {inner_func.lineno}) invoked at line "
                                f"{first_outer_call.lineno}."
                            )
                        if evidence_msg not in self.evidence:
                            self.evidence.append(evidence_msg)

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
