from algorithm_pattern_classifier.classifiers.arbitrator import (
    PatternArbitrator,
    SubsumptionRule,
)
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
    """Test explicit mutual exclusion rules."""
    arbitrator = PatternArbitrator(mutual_exclusion=[{AlgorithmPattern.BFS, AlgorithmPattern.DFS}])

    matches = [
        PatternMatch(AlgorithmPattern.BFS, 0.8, ["bfs evidence"]),
        PatternMatch(AlgorithmPattern.DFS, 0.7, ["dfs evidence"]),
    ]
    result = arbitrator.arbitrate(matches)

    assert len(result) == 1
    assert result[0].pattern == AlgorithmPattern.BFS
    assert any("Suppressed mutually exclusive dfs pattern" in ev for ev in result[0].evidence)


def test_arbitrator_bfs_dfs_coexist() -> None:
    """Test that BFS and DFS are both retained by default without exclusion."""
    arbitrator = PatternArbitrator()

    matches = [
        PatternMatch(AlgorithmPattern.BFS, 0.8, ["bfs evidence"]),
        PatternMatch(AlgorithmPattern.DFS, 0.7, ["dfs evidence"]),
    ]
    result = arbitrator.arbitrate(matches)

    assert len(result) == 2
    patterns = {r.pattern for r in result}
    assert patterns == {AlgorithmPattern.BFS, AlgorithmPattern.DFS}


def test_arbitrator_custom_rules() -> None:
    """Test PatternArbitrator with custom rules config."""
    custom_rules = {
        AlgorithmPattern.TWO_POINTERS: SubsumptionRule(
            subsumes=[AlgorithmPattern.SLIDING_WINDOW],
            threshold=0.9,
        )
    }
    arbitrator = PatternArbitrator(rules=custom_rules)

    matches = [
        PatternMatch(AlgorithmPattern.TWO_POINTERS, 0.95, ["tp evidence"]),
        PatternMatch(AlgorithmPattern.SLIDING_WINDOW, 0.8, ["sw evidence"]),
    ]
    result = arbitrator.arbitrate(matches)

    assert len(result) == 1
    assert result[0].pattern == AlgorithmPattern.TWO_POINTERS


def test_arbitrator_non_mutating() -> None:
    """Test that PatternArbitrator.arbitrate does not mutate its input match objects."""
    arbitrator = PatternArbitrator()
    matches = [
        PatternMatch(AlgorithmPattern.DYNAMIC_PROGRAMMING, 0.6, ["dp evidence"]),
        PatternMatch(AlgorithmPattern.SLIDING_WINDOW, 0.8, ["sw evidence"]),
    ]
    original_evidence_dp = list(matches[0].evidence)
    original_evidence_sw = list(matches[1].evidence)

    _ = arbitrator.arbitrate(matches)

    assert matches[0].evidence == original_evidence_dp
    assert matches[1].evidence == original_evidence_sw


def test_arbitrator_order_independence() -> None:
    """Test that input order of matches does not affect the output or evidence trail."""
    arbitrator = PatternArbitrator()

    matches1 = [
        PatternMatch(AlgorithmPattern.DFS, 0.6, ["dfs evidence"]),
        PatternMatch(AlgorithmPattern.BACKTRACKING, 0.7, ["backtracking evidence"]),
        PatternMatch(AlgorithmPattern.DYNAMIC_PROGRAMMING, 0.8, ["dp evidence"]),
    ]
    matches2 = [
        PatternMatch(AlgorithmPattern.DYNAMIC_PROGRAMMING, 0.8, ["dp evidence"]),
        PatternMatch(AlgorithmPattern.BACKTRACKING, 0.7, ["backtracking evidence"]),
        PatternMatch(AlgorithmPattern.DFS, 0.6, ["dfs evidence"]),
    ]

    res1 = arbitrator.arbitrate(matches1)
    res2 = arbitrator.arbitrate(matches2)

    assert len(res1) == 1
    assert len(res2) == 1
    assert res1[0].pattern == AlgorithmPattern.DYNAMIC_PROGRAMMING
    assert res2[0].pattern == AlgorithmPattern.DYNAMIC_PROGRAMMING
    assert sorted(res1[0].evidence) == sorted(res2[0].evidence)


def test_arbitrator_transitive_suppression_trail() -> None:
    """Test that transitive suppression notes are preserved when suppressing a match."""
    arbitrator = PatternArbitrator()
    matches = [
        PatternMatch(AlgorithmPattern.DFS, 0.6, ["dfs evidence"]),
        PatternMatch(AlgorithmPattern.BACKTRACKING, 0.7, ["backtracking evidence"]),
        PatternMatch(AlgorithmPattern.DYNAMIC_PROGRAMMING, 0.8, ["dp evidence"]),
    ]
    result = arbitrator.arbitrate(matches)

    assert len(result) == 1
    dp_match = result[0]
    assert dp_match.pattern == AlgorithmPattern.DYNAMIC_PROGRAMMING

    assert any("Suppressed dfs pattern because backtracking" in ev for ev in dp_match.evidence)
    assert any("Suppressed backtracking pattern" in ev for ev in dp_match.evidence)


def test_arbitrator_custom_mutual_exclusion() -> None:
    """Test that mutual exclusion is configurable and can be customized."""
    custom_me = [{AlgorithmPattern.BFS, AlgorithmPattern.DYNAMIC_PROGRAMMING}]
    arbitrator = PatternArbitrator(mutual_exclusion=custom_me)

    matches = [
        PatternMatch(AlgorithmPattern.BFS, 0.8, ["bfs evidence"]),
        PatternMatch(AlgorithmPattern.DYNAMIC_PROGRAMMING, 0.7, ["dp evidence"]),
    ]
    result = arbitrator.arbitrate(matches)
    assert len(result) == 1
    assert result[0].pattern == AlgorithmPattern.BFS
    assert any("Suppressed mutually exclusive" in ev for ev in result[0].evidence)


def test_arbitrator_tie_breaking() -> None:
    """Test that tie-breaking uses pattern name alphabetically for deterministic results."""
    arbitrator = PatternArbitrator(mutual_exclusion=[{AlgorithmPattern.BFS, AlgorithmPattern.DFS}])
    matches = [
        PatternMatch(AlgorithmPattern.DFS, 0.8, ["dfs"]),
        PatternMatch(AlgorithmPattern.BFS, 0.8, ["bfs"]),
    ]
    result = arbitrator.arbitrate(matches)
    assert len(result) == 1
    assert result[0].pattern == AlgorithmPattern.BFS
