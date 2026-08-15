import ast

from algorithm_pattern_classifier.classifiers.pattern_classifier import PatternClassifier
from algorithm_pattern_classifier.detectors.sliding_window import SlidingWindowDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern


def test_sliding_window_dynamic_shrink_hashmap_condition() -> None:
    """Test dynamic shrink condition using collection size (len(counts) > k) without subtraction."""
    detector = SlidingWindowDetector()
    code = (
        "def longest_k_distinct(s, k):\n"
        "    counts = {}\n"
        "    left = 0\n"
        "    ans = 0\n"
        "    for right in range(len(s)):\n"
        "        c = s[right]\n"
        "        counts[c] = counts.get(c, 0) + 1\n"
        "        while len(counts) > k:\n"
        "            left_char = s[left]\n"
        "            counts[left_char] -= 1\n"
        "            if counts[left_char] == 0:\n"
        "                del counts[left_char]\n"
        "            left += 1\n"
        "        ans = max(ans, right)\n"
        "    return ans\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert "right" in result.evidence[0]
    assert "left" in result.evidence[0]
    assert "dynamic shrink pointer" in result.evidence[0]


def test_sliding_window_dynamic_shrink_accumulator_comparison() -> None:
    """Test dynamic shrink condition comparing running accumulator (curr_sum >= target)."""
    detector = SlidingWindowDetector()
    code = (
        "def min_sub_array_len(target, nums):\n"
        "    left = 0\n"
        "    curr_sum = 0\n"
        "    min_count = 0\n"
        "    for right in range(len(nums)):\n"
        "        curr_sum += nums[right]\n"
        "        while curr_sum >= target:\n"
        "            curr_sum -= nums[left]\n"
        "            left += 1\n"
        "        min_count += 1\n"
        "    return min_count\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert "right" in result.evidence[0]
    assert "left" in result.evidence[0]


def test_sliding_window_dynamic_shrink_counter_comparison() -> None:
    """Test dynamic shrink condition comparing counter metric (zeros > k)."""
    detector = SlidingWindowDetector()
    code = (
        "def max_consecutive_ones(nums, k):\n"
        "    left = 0\n"
        "    zeros = 0\n"
        "    for right in range(len(nums)):\n"
        "        if nums[right] == 0:\n"
        "            zeros += 1\n"
        "        while zeros > k:\n"
        "            if nums[left] == 0:\n"
        "                zeros -= 1\n"
        "            left = left + 1\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert "right" in result.evidence[0]
    assert "left" in result.evidence[0]


def test_sliding_window_dynamic_shrink_subscript_condition() -> None:
    """Test dynamic shrink condition relying on collection subscript (counts[c] > 1)."""
    detector = SlidingWindowDetector()
    code = (
        "def length_of_longest_substring(s):\n"
        "    counts = {}\n"
        "    left = 0\n"
        "    for right in range(len(s)):\n"
        "        c = s[right]\n"
        "        counts[c] = counts.get(c, 0) + 1\n"
        "        while counts[c] > 1:\n"
        "            counts[s[left]] -= 1\n"
        "            left += 1\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert "right" in result.evidence[0]
    assert "left" in result.evidence[0]


def test_sliding_window_outer_while_loop() -> None:
    """Test sliding window where outer loop is a while loop expanding pointer."""
    detector = SlidingWindowDetector()
    code = (
        "def sliding_window_while(s, k):\n"
        "    left = 0\n"
        "    right = 0\n"
        "    counts = {}\n"
        "    while right < len(s):\n"
        "        c = s[right]\n"
        "        counts[c] = counts.get(c, 0) + 1\n"
        "        while len(counts) > k:\n"
        "            counts[s[left]] -= 1\n"
        "            if counts[s[left]] == 0:\n"
        "                del counts[s[left]]\n"
        "            left += 1\n"
        "        right += 1\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert "right" in result.evidence[0]
    assert "left" in result.evidence[0]


def test_sliding_window_enumerate_tuple_target() -> None:
    """Test sliding window with tuple unpacking in for loop (for right, char in enumerate(s))."""
    detector = SlidingWindowDetector()
    code = (
        "def sliding_enumerate(s, k):\n"
        "    counts = {}\n"
        "    left = 0\n"
        "    for right, char in enumerate(s):\n"
        "        counts[char] = counts.get(char, 0) + 1\n"
        "        while len(counts) > k:\n"
        "            counts[s[left]] -= 1\n"
        "            if counts[s[left]] == 0:\n"
        "                del counts[s[left]]\n"
        "            left += 1\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert "right" in result.evidence[0]
    assert "left" in result.evidence[0]


def test_sliding_window_composite_boolean_condition() -> None:
    """Test sliding window with composite boolean condition (left <= right and sum > k)."""
    detector = SlidingWindowDetector()
    code = (
        "def composite_condition(nums, k):\n"
        "    left = 0\n"
        "    total = 0\n"
        "    for right in range(len(nums)):\n"
        "        total += nums[right]\n"
        "        while left <= right and total > k:\n"
        "            total -= nums[left]\n"
        "            left += 1\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert "right" in result.evidence[0]
    assert "left" in result.evidence[0]


def test_sliding_window_negatives_three_sum() -> None:
    """Test 3Sum nested two-pointer search is not misclassified as sliding window."""
    detector = SlidingWindowDetector()
    code = (
        "def three_sum(nums):\n"
        "    nums.sort()\n"
        "    res = []\n"
        "    for i in range(len(nums)):\n"
        "        left = i + 1\n"
        "        right = len(nums) - 1\n"
        "        while left < right:\n"
        "            total = nums[i] + nums[left] + nums[right]\n"
        "            if total == 0:\n"
        "                res.append([nums[i], nums[left], nums[right]])\n"
        "                left += 1\n"
        "                right -= 1\n"
        "            elif total < 0:\n"
        "                left += 1\n"
        "            else:\n"
        "                right -= 1\n"
        "    return res\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is None


def test_sliding_window_negatives_nested_indices_only() -> None:
    """Test nested while loops with only pointer comparisons are not classified as sliding."""
    detector = SlidingWindowDetector()
    code = (
        "def nested_indices(n, m):\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        j = 0\n"
        "        while j < m:\n"
        "            j += 1\n"
        "        i += 1\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is None


def test_sliding_window_full_classifier_integration() -> None:
    """Test end-to-end classification through PatternClassifier."""
    classifier = PatternClassifier()
    code = (
        "def longest_k_distinct(s, k):\n"
        "    counts = {}\n"
        "    left = 0\n"
        "    for right in range(len(s)):\n"
        "        char = s[right]\n"
        "        counts[char] = counts.get(char, 0) + 1\n"
        "        while len(counts) > k:\n"
        "            counts[s[left]] -= 1\n"
        "            if counts[s[left]] == 0:\n"
        "                del counts[s[left]]\n"
        "            left += 1\n"
    )
    results = classifier.classify(code)
    assert len(results) >= 1
    assert results[0].pattern == AlgorithmPattern.SLIDING_WINDOW
    assert results[0].confidence >= 0.95
