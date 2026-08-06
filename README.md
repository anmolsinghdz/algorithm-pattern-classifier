# algorithm-pattern-classifier

A static analysis engine designed to classify algorithm implementations into structural patterns.

## Development Setup

To set up the development environment, create a virtual environment and install the package with development dependencies:

```bash
python -m venv .venv
# On Unix:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements-dev.txt
```

To run linting and type checking:

```bash
ruff check .
ruff format --check .
mypy src/
```

## Running Tests

To run the test suite:

```bash
pytest
```

## API / Breaking Changes Note

The following public interface and API changes have been made:
- **`ClassificationResult` Deleted:** The legacy `ClassificationResult` has been removed and replaced with the `PatternMatch` dataclass.
- **Enum Member Updates (`AlgorithmPattern`):**
  - `TWO_POINTER` was renamed to `TWO_POINTERS` (its value changed from `"two-pointer"` to `"two-pointers"`).
  - `BFS_DFS` (`"bfs-dfs"`) has been split into individual `BFS` and `DFS` patterns.
  - `BINARY_SEARCH` has been removed as it is out of scope for structural pattern classification.
- **`BaseDetector` Interface changes:**
  - The `.detect()` method signature changed to accept a pre-parsed AST (`code_ast: ast.AST`) instead of `source_code` and an optional `ast_tree`.
  - Added an abstract `@property` named `pattern` to query the target pattern type of a detector without running it.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
