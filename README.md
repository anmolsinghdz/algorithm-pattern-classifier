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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
