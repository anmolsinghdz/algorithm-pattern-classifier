import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class DFSDetector(BaseDetector):
    """Detector for the Depth-First Search (DFS) algorithmic design pattern."""

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
                self.initialized_stacks = {}

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

                # Check for recursive DFS in this function or its inner functions
                self._check_recursive_dfs(node)

                self.generic_visit(node)
                self.initialized_stacks = old_stacks

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
                        self.confidence = 0.9
                        init_line = getattr(self.initialized_stacks[stack_var], "lineno", 0)
                        self.evidence.append(
                            f"Line {node.lineno}: Found iterative DFS graph traversal pattern "
                            f"using stack '{stack_var}' initialized at line {init_line}. "
                            f"End pop at line {pop_line}, node expansion at line {append_line}."
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

            def _check_recursive_dfs(self, func_node: ast.FunctionDef) -> None:
                func_name = func_node.name
                recursive_calls: list[ast.Call] = []

                class RecursionVisitor(ast.NodeVisitor):
                    def visit_Call(self, call_node: ast.Call) -> None:
                        is_name_call = (
                            isinstance(call_node.func, ast.Name) and call_node.func.id == func_name
                        )
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
                    return

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

                    if is_in_loop_or_branch:
                        func_args = {arg.arg for arg in func_node.args.args}
                        modified_arg_passed = False

                        for call_arg in call.args:
                            is_not_arg = (
                                isinstance(call_arg, ast.Name) and call_arg.id not in func_args
                            )
                            is_expr = isinstance(
                                call_arg,
                                (ast.BinOp, ast.Subscript, ast.Call, ast.Constant),
                            )
                            if is_not_arg or is_expr:
                                modified_arg_passed = True

                        if modified_arg_passed or len(func_args) == 0:
                            self.found_dfs = True
                            self.confidence = 0.9
                            self.evidence.append(
                                f"Line {func_node.lineno}: Found recursive DFS pattern in "
                                f"function '{func_name}' with recursive call at line "
                                f"{call.lineno} inside a loop/branch."
                            )
                            break

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
