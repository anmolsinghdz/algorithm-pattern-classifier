import ast

from algorithm_pattern_classifier.classifiers.pattern_classifier import PatternClassifier
from algorithm_pattern_classifier.detectors.prefix_sum import PrefixSumDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern


def test_prefix_sum_direct_accumulation_and_query() -> None:
    """Test standard prefix sum array construction and subarray query."""
    detector = PrefixSumDetector()

    code = (
        "class NumArray:\n"
        "    def __init__(self, nums):\n"
        "        self.prefix = [0] * len(nums)\n"
        "        self.prefix[0] = nums[0]\n"
        "        for i in range(1, len(nums)):\n"
        "            self.prefix[i] = self.prefix[i - 1] + nums[i]\n"
        "\n"
        "    def sumRange(self, left, right):\n"
        "        if left == 0:\n"
        "            return self.prefix[right]\n"
        "        return self.prefix[right] - self.prefix[left - 1]\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert any("accumulative" in e for e in result.evidence)
    assert any("subarray subtraction" in e for e in result.evidence)


def test_prefix_sum_offset_accumulation() -> None:
    """Test prefix sum with offset index (prefix length n+1)."""
    detector = PrefixSumDetector()

    code = (
        "def range_sum(nums, queries):\n"
        "    n = len(nums)\n"
        "    prefix = [0] * (n + 1)\n"
        "    for i in range(n):\n"
        "        prefix[i + 1] = prefix[i] + nums[i]\n"
        "    res = []\n"
        "    for l, r in queries:\n"
        "        res.append(prefix[r + 1] - prefix[l])\n"
        "    return res\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert any("accumulative" in e for e in result.evidence)


def test_prefix_sum_running_sum_append() -> None:
    """Test prefix sum built via running sum and append."""
    detector = PrefixSumDetector()

    code = (
        "def runningSum(nums):\n"
        "    prefix = []\n"
        "    running_sum = 0\n"
        "    for num in nums:\n"
        "        running_sum += num\n"
        "        prefix.append(running_sum)\n"
        "    return prefix\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.90
    assert any("running sum" in e for e in result.evidence)


def test_prefix_sum_append_prior_element() -> None:
    """Test prefix sum built by appending prefix[-1] + num."""
    detector = PrefixSumDetector()

    code = (
        "def get_prefix(nums):\n"
        "    prefix = [0]\n"
        "    for x in nums:\n"
        "        prefix.append(prefix[-1] + x)\n"
        "    return prefix\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.90
    assert any("running sum" in e for e in result.evidence)


def test_prefix_sum_hash_map() -> None:
    """Test prefix sum with hash map lookup (e.g. Subarray Sum Equals K)."""
    detector = PrefixSumDetector()

    code = (
        "def subarraySum(nums, k):\n"
        "    count = {0: 1}\n"
        "    curr_sum = 0\n"
        "    ans = 0\n"
        "    for num in nums:\n"
        "        curr_sum += num\n"
        "        if curr_sum - k in count:\n"
        "            ans += count[curr_sum - k]\n"
        "        count[curr_sum] = count.get(curr_sum, 0) + 1\n"
        "    return ans\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert any("prefix sum lookup" in e for e in result.evidence)


def test_prefix_sum_inplace() -> None:
    """Test in-place prefix sum accumulation."""
    detector = PrefixSumDetector()

    code = (
        "def runningSumInPlace(nums):\n"
        "    for i in range(1, len(nums)):\n"
        "        nums[i] += nums[i - 1]\n"
        "    return nums\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.90
    assert any("in-place" in e for e in result.evidence)


def test_prefix_sum_itertools_accumulate() -> None:
    """Test detection with itertools.accumulate."""
    detector = PrefixSumDetector()

    code = (
        "from itertools import accumulate\ndef get_sums(nums):\n    return list(accumulate(nums))\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.90
    assert any("accumulate" in e for e in result.evidence)


def test_difference_array_pattern() -> None:
    """Test difference array range updates and prefix reconstruction."""
    detector = PrefixSumDetector()

    code = (
        "def corpFlightBookings(bookings, n):\n"
        "    diff = [0] * (n + 2)\n"
        "    for first, last, seats in bookings:\n"
        "        diff[first] += seats\n"
        "        diff[last + 1] -= seats\n"
        "    res = [0] * n\n"
        "    curr = 0\n"
        "    for i in range(1, n + 1):\n"
        "        curr += diff[i]\n"
        "        res.append(curr)\n"
        "    return res\n"
    )
    result = detector.detect(ast.parse(code))
    assert result is not None
    assert result.confidence >= 0.95
    assert any("difference array" in e for e in result.evidence)


def test_prefix_sum_negatives() -> None:
    """Test non-prefix-sum code does not trigger false positives."""
    detector = PrefixSumDetector()

    # Case 1: Simple linear scan
    linear_scan = (
        "def find_target(arr, target):\n"
        "    for i in range(len(arr)):\n"
        "        if arr[i] == target:\n"
        "            return i\n"
        "    return -1\n"
    )
    assert detector.detect(ast.parse(linear_scan)) is None

    # Case 2: Two sum with two pointers
    two_sum = (
        "def two_sum(arr, target):\n"
        "    left = 0\n"
        "    right = len(arr) - 1\n"
        "    while left < right:\n"
        "        s = arr[left] + arr[right]\n"
        "        if s == target:\n"
        "            return [left, right]\n"
        "        elif s < target:\n"
        "            left += 1\n"
        "        else:\n"
        "            right -= 1\n"
        "    return []\n"
    )
    assert detector.detect(ast.parse(two_sum)) is None

    # Case 3: Fibonacci DP (adds two prior DP states, not an input stream)
    fib_dp = (
        "def fib(n):\n"
        "    dp = [0] * (n + 1)\n"
        "    dp[1] = 1\n"
        "    for i in range(2, n + 1):\n"
        "        dp[i] = dp[i - 1] + dp[i - 2]\n"
        "    return dp[n]\n"
    )
    assert detector.detect(ast.parse(fib_dp)) is None


def test_classifier_with_prefix_sum() -> None:
    """Test PatternClassifier correctly identifies and ranks prefix sum."""
    classifier = PatternClassifier()

    code = (
        "class NumArray:\n"
        "    def __init__(self, nums):\n"
        "        self.prefix = [0] * len(nums)\n"
        "        self.prefix[0] = nums[0]\n"
        "        for i in range(1, len(nums)):\n"
        "            self.prefix[i] = self.prefix[i - 1] + nums[i]\n"
        "\n"
        "    def sumRange(self, left, right):\n"
        "        if left == 0:\n"
        "            return self.prefix[right]\n"
        "        return self.prefix[right] - self.prefix[left - 1]\n"
    )
    results = classifier.classify(code)
    assert len(results) >= 1
    assert results[0].pattern == AlgorithmPattern.PREFIX_SUM
    assert results[0].confidence >= 0.95
