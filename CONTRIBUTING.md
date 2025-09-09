# Contributing to Bead-Spring Analytics

Thank you for your interest in improving the project. The following guidelines will help you set up a development environment and contribute effectively.

## Development workflow
1. Fork the repository and clone your fork.
2. Create a new branch for your feature or bugfix.
3. Install the development dependencies and make your changes.
4. Run the test suite to ensure all tests pass.
5. Commit your changes with clear commit messages and push to your fork.
6. Open a pull request against the main repository and request a review.

## Code style
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- The project favours functional programming; prefer functions over classes.
- Include informative docstrings and type hints where appropriate.
- Format code with [Black](https://black.readthedocs.io/en/stable/) and check linting with `flake8` if available.

## Testing
- Add or update tests for any new functionality or bug fixes under the `tests/` directory.
- Run the test suite locally before opening a pull request:

```bash
pytest
```

Thank you for contributing!
