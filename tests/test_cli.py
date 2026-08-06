import json
from typing import Any
from unittest.mock import patch

import pytest

from algorithm_pattern_classifier.cli import main


def test_cli_valid_input(tmp_path: Any, capsys: Any) -> None:
    """Test CLI behavior with valid Python code."""
    f = tmp_path / "valid.py"
    f.write_text("def foo():\n    pass\n", encoding="utf-8")

    with patch("sys.argv", ["pattern-classifier", "classify", str(f)]):
        main()

    captured = capsys.readouterr()
    assert "Algorithm Pattern Classification Report" in captured.out
    assert "No patterns detected" in captured.out


def test_cli_invalid_input(tmp_path: Any, capsys: Any) -> None:
    """Test CLI behavior with syntactically broken Python code (human readable)."""
    f = tmp_path / "invalid.py"
    f.write_text("def broken(:\n    pass\n", encoding="utf-8")

    with (
        patch("sys.argv", ["pattern-classifier", "classify", str(f)]),
        pytest.raises(SystemExit) as excinfo,
    ):
        main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Failed to parse Python code" in captured.err
    assert "Line:   1" in captured.err
    assert captured.out == ""


def test_cli_invalid_input_json(tmp_path: Any, capsys: Any) -> None:
    """Test CLI behavior with syntactically broken Python code in --json mode."""
    f = tmp_path / "invalid.py"
    f.write_text("def broken(:\n    pass\n", encoding="utf-8")

    with (
        patch("sys.argv", ["pattern-classifier", "classify", str(f), "--json"]),
        pytest.raises(SystemExit) as excinfo,
    ):
        main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.err)
    assert data["error"] == "Failed to parse Python code"
    assert data["line"] == 1
    assert captured.out == ""
