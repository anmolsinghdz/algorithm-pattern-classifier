import ast

from algorithm_pattern_classifier.detectors.bfs import BFSDetector
from algorithm_pattern_classifier.detectors.dfs import DFSDetector


def test_bfs_detector_positives() -> None:
    """Test BFSDetector flags actual BFS implementations."""
    detector = BFSDetector()

    # Case 1: Standard BFS queue with collections.deque, popleft and append
    deque_bfs_code = (
        "def bfs_deque(graph, start):\n"
        "    visited = set()\n"
        "    queue = collections.deque([start])\n"
        "    while queue:\n"
        "        node = queue.popleft()\n"
        "        for neighbor in graph[node]:\n"
        "            if neighbor not in visited:\n"
        "                visited.add(neighbor)\n"
        "                queue.append(neighbor)\n"
    )
    result = detector.detect(ast.parse(deque_bfs_code))
    assert result is not None
    assert result.confidence >= 0.9
    assert "bfs" in result.evidence[0].lower()

    # Case 2: BFS queue using standard list, pop(0) and extend
    list_bfs_code = (
        "def bfs_list(graph, start):\n"
        "    queue = [start]\n"
        "    while len(queue) > 0:\n"
        "        node = queue.pop(0)\n"
        "        queue.extend(graph[node])\n"
    )
    result2 = detector.detect(ast.parse(list_bfs_code))
    assert result2 is not None
    assert result2.confidence >= 0.9


def test_bfs_detector_negatives() -> None:
    """Test BFSDetector does not flag near-misses or stack operations."""
    detector = BFSDetector()

    # Case 1: Stack operations (DF-like popping)
    stack_code = (
        "def stack_traverse(graph, start):\n"
        "    queue = [start]\n"
        "    while queue:\n"
        "        node = queue.pop()\n"
        "        queue.append(node)\n"
    )
    result = detector.detect(ast.parse(stack_code))
    assert result is None


def test_dfs_detector_positives() -> None:
    """Test DFSDetector flags actual DFS implementations."""
    detector = DFSDetector()

    # Case 1: Iterative DFS stack with pop and append
    iter_dfs_code = (
        "def dfs_iterative(graph, start):\n"
        "    visited = set()\n"
        "    stack = [start]\n"
        "    while stack:\n"
        "        node = stack.pop()\n"
        "        for neighbor in graph[node]:\n"
        "            if neighbor not in visited:\n"
        "                visited.add(neighbor)\n"
        "                stack.append(neighbor)\n"
    )
    result = detector.detect(ast.parse(iter_dfs_code))
    assert result is not None
    assert result.confidence >= 0.9

    # Case 2: Recursive DFS
    recur_dfs_code = (
        "def dfs_recursive(graph, node, visited):\n"
        "    visited.add(node)\n"
        "    for neighbor in graph[node]:\n"
        "        if neighbor not in visited:\n"
        "            dfs_recursive(graph, neighbor, visited)\n"
    )
    result2 = detector.detect(ast.parse(recur_dfs_code))
    assert result2 is not None
    assert result2.confidence >= 0.9


def test_dfs_detector_negatives() -> None:
    """Test DFSDetector does not flag queue operations or non-traversal recursions."""
    detector = DFSDetector()

    # Case 1: BFS implementation (should be detected as BFS, not DFS)
    bfs_code = (
        "def bfs_traverse(graph, start):\n"
        "    queue = [start]\n"
        "    while queue:\n"
        "        node = queue.pop(0)\n"
        "        queue.append(node)\n"
    )
    result = detector.detect(ast.parse(bfs_code))
    assert result is None

    # Case 2: Ordinary mathematical recursion (e.g. factorial)
    factorial_code = (
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n"
    )
    result2 = detector.detect(ast.parse(factorial_code))
    assert result2 is None
