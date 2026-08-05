import dataclasses
from typing import Any

from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


@dataclasses.dataclass(frozen=True)
class SubsumptionRule:
    """Rule defining pattern subsumption behavior."""

    subsumes: list[AlgorithmPattern] = dataclasses.field(default_factory=list)
    threshold: float = 0.0
    compare_confidence: bool = False


class PatternArbitrator:
    """Resolves overlapping pattern classifications using a declarative rules matrix."""

    def __init__(
        self,
        rules: dict[AlgorithmPattern, SubsumptionRule | dict[str, Any]] | None = None,
        mutual_exclusion: list[set[AlgorithmPattern]] | None = None,
    ) -> None:
        """Initialize the PatternArbitrator with rules and mutual exclusion configurations.

        Args:
            rules: A dictionary mapping patterns to their subsumption config.
            mutual_exclusion: A list of sets of mutually exclusive patterns.
        """
        self.rules: dict[AlgorithmPattern, SubsumptionRule] = {}

        if rules is None:
            # Default precedence rules matrix
            self.rules = {
                AlgorithmPattern.DYNAMIC_PROGRAMMING: SubsumptionRule(
                    subsumes=[
                        AlgorithmPattern.SLIDING_WINDOW,
                        AlgorithmPattern.TWO_POINTERS,
                        AlgorithmPattern.DFS,
                        AlgorithmPattern.BACKTRACKING,
                    ],
                    threshold=0.5,
                ),
                AlgorithmPattern.SLIDING_WINDOW: SubsumptionRule(
                    subsumes=[AlgorithmPattern.TWO_POINTERS],
                    threshold=0.0,
                    compare_confidence=True,
                ),
                AlgorithmPattern.BACKTRACKING: SubsumptionRule(
                    subsumes=[AlgorithmPattern.DFS],
                    threshold=0.5,
                ),
            }
        else:
            for pattern, rule_val in rules.items():
                if isinstance(rule_val, SubsumptionRule):
                    self.rules[pattern] = rule_val
                elif isinstance(rule_val, dict):
                    self.rules[pattern] = SubsumptionRule(
                        subsumes=rule_val.get("subsumes", []),
                        threshold=float(rule_val.get("threshold", 0.0)),
                        compare_confidence=bool(rule_val.get("compare_confidence", False)),
                    )

        if mutual_exclusion is None:
            self.mutual_exclusion: list[set[AlgorithmPattern]] = []
        else:
            self.mutual_exclusion = mutual_exclusion

    def arbitrate(self, matches: list[PatternMatch]) -> list[PatternMatch]:
        """Apply the arbitration matrix to filter and rank pattern matches.

        Args:
            matches: A raw list of detected pattern matches.

        Returns:
            A filtered and sorted list of PatternMatch objects.
        """
        # Shallow copy input matches and copy their evidence lists to prevent mutating caller state
        local_matches = [dataclasses.replace(m, evidence=list(m.evidence)) for m in matches]
        suppressed_indices: set[int] = set()

        # 1. Apply Subsumption Rules in descending confidence order (with pattern value tie-breaker)
        # to ensure the outcome is independent of the registration order of detectors.
        sorted_indices = sorted(
            range(len(local_matches)),
            key=lambda idx: (-local_matches[idx].confidence, local_matches[idx].pattern.value),
        )

        for i in sorted_indices:
            if i in suppressed_indices:
                continue
            source = local_matches[i]
            rule = self.rules.get(source.pattern)
            if not rule:
                continue

            if source.confidence < rule.threshold:
                continue

            for j in sorted_indices:
                if i == j or j in suppressed_indices:
                    continue
                target = local_matches[j]
                if target.pattern in rule.subsumes:
                    if rule.compare_confidence and target.confidence > source.confidence:
                        continue

                    # Allow target to suppress any matches it subsumes first
                    # to maintain a complete transitive suppression trail
                    target_rule = self.rules.get(target.pattern)
                    if target_rule and target.confidence >= target_rule.threshold:
                        for k in sorted_indices:
                            if k in (j, i) or k in suppressed_indices:
                                continue
                            sub_target = local_matches[k]
                            if sub_target.pattern in target_rule.subsumes:
                                if (
                                    target_rule.compare_confidence
                                    and sub_target.confidence > target.confidence
                                ):
                                    continue
                                suppressed_indices.add(k)
                                sub_transitive = [
                                    note
                                    for note in sub_target.evidence
                                    if note.startswith("Suppressed ")
                                ]
                                local_matches[j] = dataclasses.replace(
                                    local_matches[j],
                                    evidence=[
                                        *local_matches[j].evidence,
                                        *sub_transitive,
                                        f"Suppressed {sub_target.pattern.value} pattern because "
                                        f"{target.pattern.value} was detected with confidence "
                                        f"{target.confidence}.",
                                    ],
                                )
                                target = local_matches[j]

                    suppressed_indices.add(j)
                    transitive_notes = [
                        note for note in target.evidence if note.startswith("Suppressed ")
                    ]
                    # Use dataclasses.replace to update evidence on our local copy
                    local_matches[i] = dataclasses.replace(
                        source,
                        evidence=[
                            *source.evidence,
                            *transitive_notes,
                            f"Suppressed {target.pattern.value} pattern because "
                            f"{source.pattern.value} was detected with confidence "
                            f"{source.confidence}.",
                        ],
                    )
                    source = local_matches[i]

        # 2. Apply Mutual Exclusion Rules
        for group in self.mutual_exclusion:
            group_matches = [
                (idx, m)
                for idx, m in enumerate(local_matches)
                if m.pattern in group and idx not in suppressed_indices
            ]
            if len(group_matches) > 1:
                # Sort group matches by confidence descending, with pattern value as tie-breaker
                group_matches.sort(key=lambda x: (-x[1].confidence, x[1].pattern.value))
                winner_idx, winner_match = group_matches[0]
                for idx, match in group_matches[1:]:
                    suppressed_indices.add(idx)
                    transitive_notes = [
                        note for note in match.evidence if note.startswith("Suppressed ")
                    ]
                    local_matches[winner_idx] = dataclasses.replace(
                        winner_match,
                        evidence=[
                            *winner_match.evidence,
                            *transitive_notes,
                            f"Suppressed mutually exclusive {match.pattern.value} pattern "
                            f"in favor of {winner_match.pattern.value} with confidence "
                            f"{winner_match.confidence}.",
                        ],
                    )
                    winner_match = local_matches[winner_idx]

        filtered_matches = [
            m for idx, m in enumerate(local_matches) if idx not in suppressed_indices
        ]

        return sorted(
            filtered_matches,
            key=lambda r: (-r.confidence, r.pattern.value),
        )
