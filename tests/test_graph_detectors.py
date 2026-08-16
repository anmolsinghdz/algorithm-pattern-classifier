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


def test_dfs_nested_helper_tree_traversal() -> None:
    """Test DFS detector identifies recursive helper functions inside an outer function."""
    detector = DFSDetector()
    code = (
        "def solve(root):\n"
        "    res = []\n"
        "    def dfs(node):\n"
        "        if not node:\n"
        "            return\n"
        "        res.append(node.val)\n"
        "        dfs(node.left)\n"
        "        dfs(node.right)\n"
        "    dfs(root)\n"
        "    return res\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.9
    assert any("solve" in e and "dfs" in e for e in result.evidence)
    assert any("root" in e for e in result.evidence)


def test_dfs_nested_helper_grid_islands() -> None:
    """Test DFS detector identifies nested helper for multi-source grid traversal."""
    detector = DFSDetector()
    code = (
        "def num_islands(grid):\n"
        "    def dfs(r, c):\n"
        "        if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] != '1':\n"
        "            return\n"
        "        grid[r][c] = '0'\n"
        "        dfs(r + 1, c)\n"
        "        dfs(r - 1, c)\n"
        "        dfs(r, c + 1)\n"
        "        dfs(r, c - 1)\n"
        "    for i in range(len(grid)):\n"
        "        for j in range(len(grid[0])):\n"
        "            if grid[i][j] == '1':\n"
        "                dfs(i, j)\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.9
    assert any("num_islands" in e for e in result.evidence)


def test_dfs_nested_helper_in_class_method() -> None:
    """Test DFS detector identifies nested recursive helper inside class method."""
    detector = DFSDetector()
    code = (
        "class Solution:\n"
        "    def maxDepth(self, root):\n"
        "        def dfs(node):\n"
        "            if not node:\n"
        "                return 0\n"
        "            return 1 + max(dfs(node.left), dfs(node.right))\n"
        "        return dfs(root)\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.9


def test_bfs_nested_helper_tree_level_order() -> None:
    """Test BFS detector identifies nested helper queue traversal."""
    detector = BFSDetector()
    code = (
        "def level_order(root):\n"
        "    def bfs(start_node):\n"
        "        if not start_node:\n"
        "            return []\n"
        "        queue = collections.deque([start_node])\n"
        "        levels = []\n"
        "        while queue:\n"
        "            node = queue.popleft()\n"
        "            levels.append(node.val)\n"
        "            if node.left:\n"
        "                queue.append(node.left)\n"
        "            if node.right:\n"
        "                queue.append(node.right)\n"
        "        return levels\n"
        "    return bfs(root)\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.9
    assert any("level_order" in e and "bfs" in e for e in result.evidence)
    assert any("root" in e for e in result.evidence)


def test_bfs_nested_helper_shortest_path() -> None:
    """Test BFS detector identifies nested helper list queue for shortest path."""
    detector = BFSDetector()
    code = (
        "def shortest_path(graph, start, target):\n"
        "    def bfs(source):\n"
        "        visited = {source}\n"
        "        queue = [source]\n"
        "        dist = 0\n"
        "        while queue:\n"
        "            curr = queue.pop(0)\n"
        "            if curr == target:\n"
        "                return dist\n"
        "            for neighbor in graph[curr]:\n"
        "                if neighbor not in visited:\n"
        "                    visited.add(neighbor)\n"
        "                    queue.append(neighbor)\n"
        "            dist += 1\n"
        "        return -1\n"
        "    return bfs(start)\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.9
    assert any("shortest_path" in e for e in result.evidence)
