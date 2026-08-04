from algorithm_pattern_classifier.detectors.fast_slow_pointers import FastSlowPointersDetector
from algorithm_pattern_classifier.detectors.sliding_window import SlidingWindowDetector
from algorithm_pattern_classifier.detectors.two_pointer import TwoPointerDetector


def test_two_pointer_detector_positives() -> None:
    """Test TwoPointerDetector flags actual two-pointer implementations."""
    detector = TwoPointerDetector()

    # Case 1: Two Sum reference implementation
    two_sum_code = (
        "def two_sum(arr, target):\n"
        "    left = 0\n"
        "    right = len(arr) - 1\n"
        "    while left < right:\n"
        "        current = arr[left] + arr[right]\n"
        "        if current == target:\n"
        "            return [left, right]\n"
        "        elif current < target:\n"
        "            left += 1\n"
        "        else:\n"
        "            right -= 1\n"
        "    return []\n"
    )
    result = detector.detect(two_sum_code)
    assert result.confidence_score > 0.8
    assert "left" in result.supporting_evidence[0]
    assert "right" in result.supporting_evidence[0]


def test_two_pointer_detector_negatives() -> None:
    """Test TwoPointerDetector does not flag near-misses or simple linear scans."""
    detector = TwoPointerDetector()

    # Case 1: Single pointer linear scan
    linear_scan_code = (
        "def linear_scan(arr, target):\n"
        "    for i in range(len(arr)):\n"
        "        if arr[i] == target:\n"
        "            return i\n"
        "    return -1\n"
    )
    result = detector.detect(linear_scan_code)
    assert result.confidence_score == 0.0

    # Case 2: Near-miss (loop condition compares two variables, but only one is updated)
    single_update_code = (
        "def single_update(arr):\n"
        "    i = 0\n"
        "    j = 10\n"
        "    while i < j:\n"
        "        print(arr[i])\n"
        "        i += 1\n"
    )
    result = detector.detect(single_update_code)
    assert result.confidence_score == 0.0

    # Case 3: Two independent loops
    two_loops_code = (
        "def two_loops(arr):\n"
        "    for i in range(len(arr)):\n"
        "        pass\n"
        "    for j in range(len(arr)):\n"
        "        pass\n"
    )
    result = detector.detect(two_loops_code)
    assert result.confidence_score == 0.0


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
    result = detector.detect(longest_substring_code)
    assert result.confidence_score > 0.8
    assert "end" in result.supporting_evidence[0]
    assert "start" in result.supporting_evidence[0]


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
    result = detector.detect(linear_scan_code)
    assert result.confidence_score == 0.0

    # Case 2: Nested loops that are not sliding window (e.g. bubble sort)
    bubble_sort_code = (
        "def bubble_sort(arr):\n"
        "    n = len(arr)\n"
        "    for i in range(n):\n"
        "        for j in range(n - i - 1):\n"
        "            if arr[j] > arr[j + 1]:\n"
        "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
    )
    result = detector.detect(bubble_sort_code)
    assert result.confidence_score == 0.0


def test_fast_slow_pointers_linked_list_cycle() -> None:
    """Test detection of Floyd's algorithm for linked list cycle (LC 141)."""
    detector = FastSlowPointersDetector()

    code = (
        "def hasCycle(head):\n"
        "    if not head or not head.next:\n"
        "        return False\n"
        "    slow = head\n"
        "    fast = head\n"
        "    while fast and fast.next:\n"
        "        slow = slow.next\n"
        "        fast = fast.next.next\n"
        "        if slow == fast:\n"
        "            return True\n"
        "    return False\n"
    )
    result = detector.detect(code)
    assert result.confidence_score > 0.8
    assert "slow" in result.supporting_evidence[0]
    assert "fast" in result.supporting_evidence[0]


def test_fast_slow_pointers_find_duplicate() -> None:
    """Test detection of fast/slow pointers in array (LC 287)."""
    detector = FastSlowPointersDetector()

    code = (
        "def findDuplicate(nums):\n"
        "    slow = nums[0]\n"
        "    fast = nums[0]\n"
        "    while True:\n"
        "        slow = nums[slow]\n"
        "        fast = nums[nums[fast]]\n"
        "        if slow == fast:\n"
        "            break\n"
        "    slow = nums[0]\n"
        "    while slow != fast:\n"
        "        slow = nums[slow]\n"
        "        fast = nums[fast]\n"
        "    return slow\n"
    )
    result = detector.detect(code)
    assert result.confidence_score > 0.8
    assert "slow" in result.supporting_evidence[0]
    assert "fast" in result.supporting_evidence[0]


def test_fast_slow_pointers_augassign() -> None:
    """Test detection using augmented assignment (+= 1 vs += 2)."""
    detector = FastSlowPointersDetector()

    code = (
        "def find_cycle(arr):\n"
        "    slow = 0\n"
        "    fast = 0\n"
        "    while True:\n"
        "        slow += 1\n"
        "        fast += 2\n"
        "        if slow == fast:\n"
        "            return True\n"
        "    return False\n"
    )
    result = detector.detect(code)
    assert result.confidence_score > 0.8


def test_fast_slow_pointers_while_condition_eq() -> None:
    """Test detection when equality check is in the while condition."""
    detector = FastSlowPointersDetector()

    code = (
        "def find_cycle(arr):\n"
        "    slow = 0\n"
        "    fast = 1\n"
        "    while slow != fast:\n"
        "        slow = arr[slow]\n"
        "        fast = arr[arr[fast]]\n"
        "    return slow\n"
    )
    result = detector.detect(code)
    assert result.confidence_score > 0.8


def test_fast_slow_pointers_no_differential() -> None:
    """Test that same-step-rate pointers do not trigger detection."""
    detector = FastSlowPointersDetector()

    code = (
        "def find_duplicate(nums):\n"
        "    slow = nums[0]\n"
        "    fast = nums[0]\n"
        "    while slow != fast:\n"
        "        slow = nums[slow]\n"
        "        fast = nums[fast]\n"
        "    return slow\n"
    )
    result = detector.detect(code)
    assert result.confidence_score == 0.0


def test_fast_slow_pointers_no_equality_check() -> None:
    """Test that no equality check between pointers does not trigger."""
    detector = FastSlowPointersDetector()

    code = (
        "def example():\n"
        "    slow = 0\n"
        "    fast = 0\n"
        "    while True:\n"
        "        slow += 1\n"
        "        fast += 2\n"
        "        print(slow, fast)\n"
    )
    result = detector.detect(code)
    assert result.confidence_score == 0.0


def test_fast_slow_pointers_single_pointer() -> None:
    """Test that a single-pointer linear scan does not trigger."""
    detector = FastSlowPointersDetector()

    code = (
        "def linear_scan(arr, target):\n"
        "    for i in range(len(arr)):\n"
        "        if arr[i] == target:\n"
        "            return i\n"
        "    return -1\n"
    )
    result = detector.detect(code)
    assert result.confidence_score == 0.0


def test_fast_slow_pointers_not_initialized() -> None:
    """Test that pointers not initialized before the loop give partial confidence."""
    detector = FastSlowPointersDetector()

    code = (
        "def example():\n"
        "    while True:\n"
        "        slow += 1\n"
        "        fast += 2\n"
        "        if slow == fast:\n"
        "            break\n"
    )
    result = detector.detect(code)
    # Should still detect (base 0.3 + differential 0.4 = 0.7, no init bonus)
    assert result.confidence_score == 0.7
