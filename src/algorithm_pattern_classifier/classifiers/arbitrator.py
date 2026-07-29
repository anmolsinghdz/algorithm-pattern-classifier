from typing import Any, cast

from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class PatternArbitrator:
    """Resolves overlapping pattern classifications using a declarative rules matrix."""

    def __init__(self, rules: dict[AlgorithmPattern, dict[str, Any]] | None = None) -> None:
        """Initialize the PatternArbitrator with an optional rules matrix.

        Args:
            rules: A dictionary mapping patterns to their subsumption config.
        """
        if rules is None:
            # Default precedence rules matrix
            self.rules = {
                AlgorithmPattern.DYNAMIC_PROGRAMMING: {
                    "subsumes": [
                        AlgorithmPattern.SLIDING_WINDOW,
                        AlgorithmPattern.TWO_POINTERS,
                        AlgorithmPattern.DFS,
                        AlgorithmPattern.BACKTRACKING,
                    ],
                    "threshold": 0.5,
                },
                AlgorithmPattern.SLIDING_WINDOW: {
                    "subsumes": [AlgorithmPattern.TWO_POINTERS],
                    "threshold": 0.0,
                    "compare_confidence": True,
                },
                AlgorithmPattern.BACKTRACKING: {
                    "subsumes": [AlgorithmPattern.DFS],
                    "threshold": 0.5,
                },
            }
        else:
            self.rules = rules

        self.mutual_exclusion: list[set[AlgorithmPattern]] = [
            {AlgorithmPattern.BFS, AlgorithmPattern.DFS}
        ]

    def arbitrate(self, matches: list[PatternMatch]) -> list[PatternMatch]:
        """Apply the arbitration matrix to filter and rank pattern matches.

        Args:
            matches: A raw list of detected pattern matches.

        Returns:
            A filtered and sorted list of PatternMatch objects.
        """
        suppressed_indices: set[int] = set()

        # 1. Apply Subsumption Rules
        for i, source in enumerate(matches):
            if i in suppressed_indices:
                continue
            rule = self.rules.get(source.pattern)
            if not rule:
                continue

            threshold = float(cast(float, rule.get("threshold", 0.0)))
            if source.confidence < threshold:
                continue

            subsumed_patterns = cast(list[AlgorithmPattern], rule.get("subsumes", []))
            compare_confidence = bool(rule.get("compare_confidence", False))

            for j, target in enumerate(matches):
                if i == j or j in suppressed_indices:
                    continue
                if target.pattern in subsumed_patterns:
                    if compare_confidence and target.confidence > source.confidence:
                        continue

                    suppressed_indices.add(j)
                    source.evidence.append(
                        f"Suppressed {target.pattern.value} pattern because "
                        f"{source.pattern.value} was detected with confidence {source.confidence}."
                    )

        # 2. Apply Mutual Exclusion Rules
        for group in self.mutual_exclusion:
            group_matches = [
                (idx, m)
                for idx, m in enumerate(matches)
                if m.pattern in group and idx not in suppressed_indices
            ]
            if len(group_matches) > 1:
                group_matches.sort(key=lambda x: x[1].confidence, reverse=True)
                _, winner_match = group_matches[0]
                for idx, match in group_matches[1:]:
                    suppressed_indices.add(idx)
                    winner_match.evidence.append(
                        f"Suppressed mutually exclusive {match.pattern.value} pattern "
                        f"in favor of {winner_match.pattern.value} with confidence "
                        f"{winner_match.confidence}."
                    )

        filtered_matches = [m for idx, m in enumerate(matches) if idx not in suppressed_indices]

        return sorted(
            filtered_matches,
            key=lambda r: (r.confidence, r.pattern.value),
            reverse=True,
        )
