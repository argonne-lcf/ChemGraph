# Code formatting and linting

ChemGraph uses Ruff configuration from `pyproject.toml`. Match surrounding code
and avoid reformatting unrelated files.

Run the required lint gate from the repository root:

```bash
ruff check .
```

The configured rules include Python syntax/error checks and unused-import/name
checks. The project targets Python 3.11 and uses an 88-character line length for
Ruff formatting configuration.

For the full contribution workflow and test commands, see
[Contributing](contributing.md).
