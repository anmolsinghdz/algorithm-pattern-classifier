import ast

from algorithm_pattern_classifier.detectors.sliding_window import SlidingWindowDetector
from algorithm_pattern_classifier.detectors.two_pointers import TwoPointersDetector


def test_two_pointer_detector_positives() -> None:
    """Test TwoPointersDetector flags actual two-pointer implementations."""
    detector = TwoPointersDetector()

    # Case 1: Two Sum reference implementation
    two_sum_code = (
        "def two_sum(arr, target):\n"
        "    left = 0\n"
        "    right = len(arr) - 1\n"
        "    while left < right:\n"
        "        current = arr[left] + arr[right]\n"
        "        if current == target:\n"
        "            return [left, right]\n"
        "        if current < target:\n"
        "            left += 1\n"
        "        else:\n"
        "            right -= 1\n"
        "    return []\n"
    )
    result = detector.detect(ast.parse(two_sum_code))
    assert result is not None
    assert result.confidence > 0.8
    assert "left" in result.evidence[0]
    assert "right" in result.evidence[0]

    # Case 2: Tuple unpacking implementation
    tuple_unpacking_code = (
        "def two_sum_unpack(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left < right:\n"
        "        val = arr[left] + arr[right]\n"
        "        if val == target:\n"
        "            return True\n"
        "        left, right = left + 1, right - 1\n"
        "    return False\n"
    )
    result2 = detector.detect(ast.parse(tuple_unpacking_code))
    assert result2 is not None
    assert result2.confidence > 0.8


def test_two_pointer_detector_negatives() -> None:
    """Test TwoPointersDetector does not flag near-misses or simple linear scans."""
    detector = TwoPointersDetector()

    # Case 1: Single pointer linear scan
    linear_scan_code = (
        "def linear_scan(arr, target):\n"
        "    for i in range(len(arr)):\n"
        "        if arr[i] == target:\n"
        "            return i\n"
        "    return -1\n"
    )
    result = detector.detect(ast.parse(linear_scan_code))
    assert result is None

    # Case 2: Near-miss (loop condition compares two variables, but only one is updated)
    single_update_code = (
        "def single_update(arr):\n"
        "    i = 0\n"
        "    j = 10\n"
        "    while i < j:\n"
        "        print(arr[i])\n"
        "        i += 1\n"
    )
    result = detector.detect(ast.parse(single_update_code))
    assert result is None

    # Case 3: Two independent loops
    two_loops_code = (
        "def two_loops(arr):\n"
        "    for i in range(len(arr)):\n"
        "        pass\n"
        "    for j in range(len(arr)):\n"
        "        pass\n"
    )
    result = detector.detect(ast.parse(two_loops_code))
    assert result is None


def test_sliding_window_detector_positives() -> None:
    """Test SlidingWindowDetector flags actual sliding window implementations."""
    detector = SlidingWindowDetector()

    # Case 1: Longest Substring Without Repeat reference implementation
    longest_substring_code = (
        "def longest_substring(s):\n"
        "    char_map = {}\n"
        "    max_len = 0\n"
        "    start = 0\n"
        "    for end in range(len(s)):\n"
        "        if s[end] in char_map and char_map[s[end]] >= start:\n"
        "            start = char_map[s[end]] + 1\n"
        "        char_map[s[end]] = end\n"
        "        max_len = max(max_len, end - start + 1)\n"
        "    return max_len\n"
    )
    result = detector.detect(ast.parse(longest_substring_code))
    assert result is not None
    assert result.confidence > 0.8
    assert "end" in result.evidence[0]
    assert "start" in result.evidence[0]


def test_sliding_window_detector_negatives() -> None:
    """Test SlidingWindowDetector does not flag linear scans or near-misses."""
    detector = SlidingWindowDetector()

    # Case 1: Simple linear scan
    linear_scan_code = (
        "def linear_scan(arr, target):\n"
        "    for i in range(len(arr)):\n"
        "        if arr[i] == target:\n"
        "            return i\n"
        "    return -1\n"
    )
    result = detector.detect(ast.parse(linear_scan_code))
    assert result is None

    # Case 2: Nested loops that are not sliding window (e.g. bubble sort)
    bubble_sort_code = (
        "def bubble_sort(arr):\n"
        "    n = len(arr)\n"
        "    for i in range(n):\n"
        "        for j in range(n - i - 1):\n"
        "            if arr[j] > arr[j + 1]:\n"
        "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
    )
    result = detector.detect(ast.parse(bubble_sort_code))
    assert result is None


def test_two_pointer_detector_swap() -> None:
    """Test that a pure swap does not count as updates and scores 0.0."""
    detector = TwoPointersDetector()
    code = (
        "def swap_only(arr):\n"
        "    left = 0\n"
        "    right = len(arr) - 1\n"
        "    while left < right:\n"
        "        left, right = right, left\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is None


def test_two_pointer_detector_binary_search() -> None:
    """Test binary-search shape with converging pointers assigned from midpoint."""
    detector = TwoPointersDetector()
    code = (
        "def binary_search(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left < right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid\n"
        "    return left\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence == 0.8  # both updated generally (not classic updates of left/right)


def test_two_pointer_detector_async() -> None:
    """Test that async functions containing two-pointer logic are detected correctly."""
    detector = TwoPointersDetector()
    code = (
        "async def async_two_sum(arr, target):\n"
        "    left = 0\n"
        "    right = len(arr) - 1\n"
        "    while left < right:\n"
        "        current = arr[left] + arr[right]\n"
        "        if current == target:\n"
        "            return [left, right]\n"
        "        if current < target:\n"
        "            left += 1\n"
        "        else:\n"
        "            right -= 1\n"
        "    return []\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence == 1.0


def test_two_pointer_detector_tiers() -> None:
    """Test different confidence tiers of two-pointer detection (e.g. 0.8 vs 0.6)."""
    detector = TwoPointersDetector()

    # Tier 0.8: Initialized and updated, but neither pointer is updated arithmetically
    # (e.g. linked list / general pointers)
    non_classic_code = (
        "def list_search(nodes):\n"
        "    left = nodes[0]\n"
        "    right = nodes[-1]\n"
        "    while left < right:\n"
        "        left = left.next\n"
        "        right = right.prev\n"
    )
    result_0_8 = detector.detect(ast.parse(non_classic_code))
    assert result_0_8 is not None
    assert result_0_8.confidence == 0.8

    # Tier 0.8: Updates only, no pre-loop initialization detected, but both classic updates
    no_init_code = (
        "def no_init(arr):\n"
        "    while left < right:\n"
        "        left = left + 1\n"
        "        right = right - 1\n"
    )
    result_no_init = detector.detect(ast.parse(no_init_code))
    assert result_no_init is not None
    assert result_no_init.confidence == 0.8

    # Tier 0.6: Updates only, no pre-loop initialization, and no classic updates
    no_init_non_classic_code = (
        "def no_init_non_classic(nodes):\n"
        "    while left < right:\n"
        "        left = nodes[0]\n"
        "        right = nodes[-1]\n"
    )
    result_0_6 = detector.detect(ast.parse(no_init_non_classic_code))
    assert result_0_6 is not None
    assert result_0_6.confidence == 0.6


def test_two_pointer_detector_parameter_initializers() -> None:
    """Test that posonlyargs, kwonlyargs, and varargs are treated as pointer initializers."""
    detector = TwoPointersDetector()

    # Keyword-only arguments
    code_kwonly = (
        "def two_sum(arr, *, left, right):\n"
        "    while left < right:\n"
        "        left += 1\n"
        "        right -= 1\n"
    )
    result = detector.detect(ast.parse(code_kwonly))
    assert result is not None
    assert result.confidence == 1.0
