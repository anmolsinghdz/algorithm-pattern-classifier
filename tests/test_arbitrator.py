from algorithm_pattern_classifier.classifiers.arbitrator import PatternArbitrator
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


def test_arbitrator_default_subsumption() -> None:
    """Test default subsumption rules (e.g. DP subsumes other patterns)."""
    arbitrator = PatternArbitrator()

    # Case 1: DP confidence >= 0.5 suppresses sliding window and two pointers
    matches = [
        PatternMatch(AlgorithmPattern.DYNAMIC_PROGRAMMING, 0.6, ["dp evidence"]),
        PatternMatch(AlgorithmPattern.SLIDING_WINDOW, 0.8, ["sw evidence"]),
        PatternMatch(AlgorithmPattern.TWO_POINTERS, 0.7, ["tp evidence"]),
    ]
    result = arbitrator.arbitrate(matches)

    assert len(result) == 1
    assert result[0].pattern == AlgorithmPattern.DYNAMIC_PROGRAMMING
    # Check telemetry/explanation
    assert any("Suppressed sliding-window pattern" in ev for ev in result[0].evidence)
    assert any("Suppressed two-pointers pattern" in ev for ev in result[0].evidence)


def test_arbitrator_sliding_window_two_pointers() -> None:
    """Test sliding window suppresses two pointers if it has equal or higher confidence."""
    arbitrator = PatternArbitrator()

    # Case 1: SW (0.8) vs TP (0.7) -> SW wins, TP is suppressed
    matches = [
        PatternMatch(AlgorithmPattern.SLIDING_WINDOW, 0.8, ["sw evidence"]),
        PatternMatch(AlgorithmPattern.TWO_POINTERS, 0.7, ["tp evidence"]),
    ]
    result = arbitrator.arbitrate(matches)
    assert len(result) == 1
    assert result[0].pattern == AlgorithmPattern.SLIDING_WINDOW

    # Case 2: SW (0.6) vs TP (0.8) -> Both are kept since TP has higher confidence
    matches = [
        PatternMatch(AlgorithmPattern.SLIDING_WINDOW, 0.6, ["sw evidence"]),
        PatternMatch(AlgorithmPattern.TWO_POINTERS, 0.8, ["tp evidence"]),
    ]
    result = arbitrator.arbitrate(matches)
    assert len(result) == 2


def test_arbitrator_mutual_exclusion() -> None:
    """Test mutual exclusion rules (e.g., BFS vs DFS)."""
    arbitrator = PatternArbitrator()

    matches = [
        PatternMatch(AlgorithmPattern.BFS, 0.8, ["bfs evidence"]),
        PatternMatch(AlgorithmPattern.DFS, 0.7, ["dfs evidence"]),
    ]
    result = arbitrator.arbitrate(matches)

    assert len(result) == 1
    assert result[0].pattern == AlgorithmPattern.BFS
    assert any("Suppressed mutually exclusive dfs pattern" in ev for ev in result[0].evidence)


def test_arbitrator_custom_rules() -> None:
    """Test PatternArbitrator with custom rules config."""
    custom_rules = {
        AlgorithmPattern.TWO_POINTERS: {
            "subsumes": [AlgorithmPattern.SLIDING_WINDOW],
            "threshold": 0.9,
        }
    }
    arbitrator = PatternArbitrator(rules=custom_rules)

    matches = [
        PatternMatch(AlgorithmPattern.TWO_POINTERS, 0.95, ["tp evidence"]),
        PatternMatch(AlgorithmPattern.SLIDING_WINDOW, 0.8, ["sw evidence"]),
    ]
    result = arbitrator.arbitrate(matches)

    assert len(result) == 1
    assert result[0].pattern == AlgorithmPattern.TWO_POINTERS
