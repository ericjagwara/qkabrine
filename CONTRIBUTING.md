# Contributing to Qkabrine AutoML

Thanks for your interest in contributing! This project is small and
maintained by one person, so please be patient with response times.

## Reporting issues

- Search [existing issues](https://github.com/ericjagwara/qkabrine/issues)
  first to avoid duplicates.
- Include a minimal reproducible example: the `QkabrineAutoML(...)` call
  you used, the dataset shape, your Python and `qkabrine-automl` versions,
  and the full error traceback if there is one.
- For feature requests, describe the use case, not just the feature —
  it helps to know what you're trying to accomplish.

## Setting up a development environment

```bash
git clone https://github.com/ericjagwara/qkabrine.git
cd qkabrine
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the tests locally

```bash
pytest tests/ -v
```

Please run the test suite before opening a pull request. If you're adding
a new circuit architecture, encoding, or search strategy, add a
corresponding test under `tests/` that exercises it on a small synthetic
dataset (a few qubits, a handful of samples) so CI stays fast.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/); we don't currently
  enforce a formatter in CI, but please keep new code consistent with the
  surrounding style.
- Prefer descriptive names over abbreviations for anything that's part of
  the public API (`QkabrineAutoML`, search strategy names, parameter names).
- Add or update docstrings for any new public function, class, or method.

## Submitting a pull request

1. Fork the repository and create a branch from `main`.
2. Make your changes, with tests, and confirm `pytest tests/` passes.
3. Open a pull request describing what changed and why. Link any related
   issue.
4. GitHub Actions will automatically run the test suite on your PR — please
   make sure it's green before requesting review.

## Questions

If something isn't covered here, feel free to open an issue and ask.
