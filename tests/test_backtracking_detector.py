import ast

from algorithm_pattern_classifier.classifiers.pattern_classifier import PatternClassifier
from algorithm_pattern_classifier.detectors.backtracking import BacktrackingDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern


def test_backtracking_permutations() -> None:
    """Test backtracking detection on a standard permutations implementation."""
    code = (
        "def permute(nums):\n"
        "    res = []\n"
        "    def backtrack(path, used):\n"
        "        if len(path) == len(nums):\n"
        "            res.append(list(path))\n"
        "            return\n"
        "        for i in range(len(nums)):\n"
        "            if used[i]:\n"
        "                continue\n"
        "            path.append(nums[i])\n"
        "            used[i] = True\n"
        "            backtrack(path, used)\n"
        "            used[i] = False\n"
        "            path.pop()\n"
        "    backtrack([], [False] * len(nums))\n"
        "    return res\n"
    )
    detector = BacktrackingDetector()
    match = detector.detect(ast.parse(code))
    assert match is not None
    assert match.pattern == AlgorithmPattern.BACKTRACKING
    assert match.confidence >= 0.85


def test_backtracking_subsets() -> None:
    """Test backtracking detection on a standard subsets implementation."""
    code = (
        "def subsets(nums):\n"
        "    res = []\n"
        "    def backtrack(start, path):\n"
        "        res.append(list(path))\n"
        "        for i in range(start, len(nums)):\n"
        "            path.append(nums[i])\n"
        "            backtrack(i + 1, path)\n"
        "            path.pop()\n"
        "    backtrack(0, [])\n"
        "    return res\n"
    )
    detector = BacktrackingDetector()
    match = detector.detect(ast.parse(code))
    assert match is not None
    assert match.pattern == AlgorithmPattern.BACKTRACKING
    assert match.confidence >= 0.85


def test_standard_dfs_not_backtracking() -> None:
    """Verify that a standard recursive DFS on a binary tree is not classified as backtracking."""
    code = (
        "def maxDepth(root):\n"
        "    if not root:\n"
        "        return 0\n"
        "    left = maxDepth(root.left)\n"
        "    right = maxDepth(root.right)\n"
        "    return max(left, right) + 1\n"
    )
    detector = BacktrackingDetector()
    match = detector.detect(ast.parse(code))
    assert match is None


def test_backtracking_arbitration() -> None:
    """Verify that backtracking is prioritized over recursive DFS in PatternClassifier."""
    code = (
        "def subsets(nums):\n"
        "    res = []\n"
        "    def backtrack(start, path):\n"
        "        res.append(list(path))\n"
        "        for i in range(start, len(nums)):\n"
        "            path.append(nums[i])\n"
        "            backtrack(i + 1, path)\n"
        "            path.pop()\n"
        "    backtrack(0, [])\n"
        "    return res\n"
    )
    classifier = PatternClassifier()
    results = classifier.classify(code)

    # Ensure backtracking is detected and ranks higher/suppresses DFS
    assert len(results) > 0
    assert results[0].pattern == AlgorithmPattern.BACKTRACKING
    assert not any(r.pattern == AlgorithmPattern.DFS for r in results)
