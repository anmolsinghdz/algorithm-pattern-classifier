import ast

from algorithm_pattern_classifier.classifiers.pattern_classifier import PatternClassifier
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern
from algorithm_pattern_classifier.utils.ast_normalizer import ASTNormalizer


def test_ast_normalizer_tuple_unpacking() -> None:
    """Test ASTNormalizer desugars tuple unpacking assignments."""
    code = "left, right = 0, len(arr) - 1"
    tree = ast.parse(code)
    normalized = ASTNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)

    # Should transform to separate simple assignments for left and right
    assert isinstance(normalized, ast.Module)
    assert len(normalized.body) == 2
    assert isinstance(normalized.body[0], ast.Assign)
    assert isinstance(normalized.body[0].targets[0], ast.Name)
    assert normalized.body[0].targets[0].id == "left"
    assert isinstance(normalized.body[1], ast.Assign)
    assert isinstance(normalized.body[1].targets[0], ast.Name)
    assert normalized.body[1].targets[0].id == "right"


def test_ast_normalizer_non_tuple_unpacking() -> None:
    """Test ASTNormalizer leaves function-matched unpacking as-is."""
    code = "left, right = get_bounds()"
    tree = ast.parse(code)
    normalized = ASTNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)

    assert isinstance(normalized, ast.Module)
    assert len(normalized.body) == 1
    assert isinstance(normalized.body[0], ast.Assign)
    assert isinstance(normalized.body[0].targets[0], ast.Tuple)


def test_two_pointers_unpacking_integration() -> None:
    """Test that a function using tuple unpacking is recognized correctly."""
    code = (
        "def two_sum(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left < right:\n"
        "        val = arr[left] + arr[right]\n"
        "        if val == target:\n"
        "            return True\n"
        "        if val < target:\n"
        "            left += 1\n"
        "        else:\n"
        "            right -= 1\n"
        "    return False\n"
    )
    classifier = PatternClassifier()
    results = classifier.classify(code)

    assert len(results) > 0
    assert results[0].pattern == AlgorithmPattern.TWO_POINTERS
    assert results[0].confidence >= 0.8
