import ast
from dataclasses import dataclass, field
from typing import Any

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.pattern import AlgorithmPattern
from algorithm_pattern_classifier.models.result import ClassificationResult


@dataclass
class PointerInfo:
    """Tracks a pointer variable: its name, init source, max step count,
    and which other pointers it is compared with (== / !=)."""

    name: str
    initialized_from: str | None = None
    max_step: int = 0
    compared_with: set[str] = field(default_factory=set)


class MovementAnalyzer:
    """Counts how many logical 'steps' a pointer advances per assignment.

    The step count is inferred from the AST depth of attribute / subscript /
    call nesting beyond the variable's own name.

    Examples
    --------
    slow.next                 -> 1
    fast.next.next            -> 2
    nums[fast]                -> 1
    nums[nums[fast]]          -> 2
    move(fast)                -> 1
    move(move(fast))          -> 2
    """

    @classmethod
    def count_steps(cls, node: ast.AST, variable: str) -> int:
        """Return the number of dereference steps from `variable` in `node`.

        Returns -1 when `variable` does not appear as a base of the chain.
        """
        # Base case: the variable itself, 0 steps removed.
        if isinstance(node, ast.Name):
            return 0 if node.id == variable else -1

        # Each attribute access (`.next`) adds one step.
        if isinstance(node, ast.Attribute):
            inner = cls.count_steps(node.value, variable)
            if inner >= 0:
                return inner + 1
            return -1

        # Subscript - check the index first (e.g. nums[nums[fast]] has an
        # extra step for the nested subscript), then fall back to the value.
        if isinstance(node, ast.Subscript):
            inner = cls.count_steps(node.slice, variable)
            if inner >= 0:
                return inner + 1

            inner = cls.count_steps(node.value, variable)
            if inner >= 0:
                return inner

            return -1

        # Function call - each argument that references `variable` adds one
        # step (e.g. move(fast) -> 1, move(move(fast)) -> 2).
        if isinstance(node, ast.Call):
            best = -1
            for arg in node.args:
                score = cls.count_steps(arg, variable)
                if score >= 0:
                    best = max(best, score + 1)
            return best

        # Tuple / list destructuring - pick the deepest reference found.
        if isinstance(node, (ast.Tuple, ast.List)):
            best = -1
            for elt in node.elts:
                score = cls.count_steps(elt, variable)
                best = max(best, score)
            return best

        return -1


class InitializationAnalyzer(ast.NodeVisitor):
    """Walks the AST to collect variables that are assigned (potential pointers).

    Records each assignment target as a PointerInfo, optionally noting the
    source variable name when the RHS is a simple Name (e.g. ``slow = head``).
    """

    def __init__(self) -> None:
        self.pointers: dict[str, PointerInfo] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only track simple single-target assignments to a simple name.
        if len(node.targets) != 1:
            self.generic_visit(node)
            return

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            self.generic_visit(node)
            return

        source: str | None = None
        if isinstance(node.value, ast.Name):
            source = node.value.id

        self.pointers[target.id] = PointerInfo(
            name=target.id,
            initialized_from=source,
        )
        self.generic_visit(node)


class UpdateAnalyzer(ast.NodeVisitor):
    """Visits assignments inside a loop to determine each pointer's step rate.

    Handles both regular assignments (``slow = slow.next``) and augmented
    assignments (``fast += 2``). The maximum step count across all assignments
    for a given pointer is stored in ``PointerInfo.max_step``.
    """

    def __init__(self, pointers: dict[str, PointerInfo]) -> None:
        self.pointers = pointers

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in self.pointers:
                continue

            step = MovementAnalyzer.count_steps(node.value, target.id)
            if step >= 0:
                self.pointers[target.id].max_step = max(
                    self.pointers[target.id].max_step,
                    step,
                )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle ``slow += 1`` / ``fast += 2`` style updates."""
        if not isinstance(node.target, ast.Name):
            self.generic_visit(node)
            return
        if node.target.id not in self.pointers:
            self.generic_visit(node)
            return

        if (
            isinstance(node.op, ast.Add)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            self.pointers[node.target.id].max_step = max(
                self.pointers[node.target.id].max_step,
                node.value.value,
            )
        self.generic_visit(node)


class EqualityAnalyzer(ast.NodeVisitor):
    """Finds ``==`` / ``!=`` comparisons between tracked pointer variables.

    The presence of such a comparison is a strong signal that two pointers
    are used in a fast/slow pattern, since the fast pointer laps the slow one.
    """

    def __init__(self, pointers: dict[str, PointerInfo]) -> None:
        self.pointers = pointers

    def visit_Compare(self, node: ast.Compare) -> None:
        """Record a cross-pointer equality/disequality comparison."""
        if (
            isinstance(node.ops[0], (ast.Eq, ast.NotEq))
            and isinstance(node.left, ast.Name)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Name)
        ):
            v1 = node.left.id
            v2 = node.comparators[0].id
            if v1 != v2 and v1 in self.pointers and v2 in self.pointers:
                self.pointers[v1].compared_with.add(v2)
                self.pointers[v2].compared_with.add(v1)
        self.generic_visit(node)


class FastSlowPointersDetector(BaseDetector):
    """Detector for the Fast and Slow Pointers (Floyd's Tortoise and Hare) pattern.

    How it works
    ------------
    1. Walk every function / module body looking for ``while`` or ``for`` loops.
    2. Within each loop, collect all assigned variable names (potential pointers).
    3. For each pair of variables inside the loop:
       - Determine their movement step rates via `UpdateAnalyzer`.
       - Check if they are compared with ``==`` or ``!=`` via `EqualityAnalyzer`.
    4. If one moves 1 step and the other ≥2 steps *and* they have an equality
       check, it is classified as the fast/slow pointer pattern.

    Confidence scoring
    ------------------
    - Base                             0.3  (two loop variables found)
    - Differential step rates (1 vs 2+) +0.4
    - Both initialized before the loop  +0.3
    - Max                              1.0
    """

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.FAST_SLOW_POINTERS

    def detect(self, source_code: str, ast_tree: Any = None) -> ClassificationResult:
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

        # Inner visitor that carries the detection state per invocation.
        class FastSlowVisitor(ast.NodeVisitor):
            """Stateful AST walker that pinpoints fast/slow pointer loops."""

            def __init__(self) -> None:
                self.found = False
                self.evidence: list[str] = []
                self.confidence = 0.0

            @staticmethod
            def _collect_assigned(vars_set: set[str], stmt: ast.stmt) -> None:
                """Add any variable assigned in *stmt* to *vars_set*."""
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            vars_set.add(target.id)
                elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)) and isinstance(
                    stmt.target, ast.Name
                ):
                    vars_set.add(stmt.target.id)

            @staticmethod
            def _find_loop_vars(loop_node: ast.While | ast.For) -> set[str]:
                """Return the set of variable names assigned inside *loop_node*."""
                vars_set: set[str] = set()

                class LoopVarFinder(ast.NodeVisitor):
                    def visit_Assign(self, node: ast.Assign) -> None:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                vars_set.add(target.id)
                        self.generic_visit(node)

                    def visit_AugAssign(self, node: ast.AugAssign) -> None:
                        if isinstance(node.target, ast.Name):
                            vars_set.add(node.target.id)
                        self.generic_visit(node)

                LoopVarFinder().visit(loop_node)
                return vars_set

            def _check_loop(
                self,
                loop_node: ast.While | ast.For,
                initialized_before: set[str],
            ) -> None:
                """Analyse a single loop for the fast/slow pattern.

                1. Collect loop-local variables.
                2. Measure step rates and equality checks.
                3. Award confidence if a 1-step / ≥2-step pair with a compare exists.
                """
                loop_vars = self._find_loop_vars(loop_node)
                if len(loop_vars) < 2:
                    return

                pointers = {name: PointerInfo(name=name) for name in loop_vars}

                UpdateAnalyzer(pointers).visit(loop_node)
                EqualityAnalyzer(pointers).visit(loop_node)

                # Deduplicate pairs - each unordered pair is processed once.
                checked_pairs: set[tuple[str, str]] = set()
                for name, info in pointers.items():
                    for other_name in info.compared_with:
                        if other_name not in pointers:
                            continue
                        pair = (name, other_name) if name < other_name else (other_name, name)
                        if pair in checked_pairs:
                            continue
                        checked_pairs.add(pair)

                        s1, s2 = info.max_step, pointers[other_name].max_step
                        # The defining signal: one pointer moves 1 step per
                        # iteration, the other moves 2+ steps.
                        if not ((s1 == 1 and s2 >= 2) or (s2 == 1 and s1 >= 2)):
                            continue

                        slow = name if s1 < s2 else other_name
                        fast = other_name if s1 < s2 else name

                        # Confidence composition.
                        confidence = 0.3  # two pointers exist in the loop
                        confidence += 0.4  # differential step rates confirmed
                        if slow in initialized_before and fast in initialized_before:
                            confidence += 0.3  # both set up before the loop

                        self.found = True
                        self.confidence = max(self.confidence, confidence)
                        self.evidence.append(
                            f"Line {loop_node.lineno}: found fast/slow pointers "
                            f"'{slow}' (1 step) and '{fast}' ({max(s1, s2)} steps) "
                            f"with equality check."
                        )

            def _walk_body(self, body: list[ast.stmt], initialized: set[str]) -> None:
                """Walk a statement list, checking loops and tracking assignments."""
                for stmt in body:
                    if isinstance(stmt, (ast.While, ast.For)):
                        self._check_loop(stmt, initialized)
                    self._collect_assigned(initialized, stmt)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Enter a function - start with an empty initialized set."""
                initialized: set[str] = set()
                self._walk_body(node.body, initialized)
                self.generic_visit(node)

            def visit_Module(self, node: ast.Module) -> None:
                """Entry point for top-level code."""
                initialized: set[str] = set()
                self._walk_body(node.body, initialized)
                self.generic_visit(node)

        visitor = FastSlowVisitor()
        visitor.visit(ast_tree)

        if visitor.found:
            max_confidence = visitor.confidence
            evidence = visitor.evidence

        return ClassificationResult(
            pattern=self.pattern,
            confidence_score=max_confidence,
            supporting_evidence=evidence,
        )
