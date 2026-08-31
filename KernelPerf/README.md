# KernelPerf evaluator

This directory contains the maintainer-run Python evaluator. Start with the
repository [README](../README.md), then read [`docs/PLUGINS.md`](../docs/PLUGINS.md)
for the source contract and [`docs/REGRESSION.md`](../docs/REGRESSION.md) for
worker execution.

Install and test from this directory:

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/validate_submissions.py submissions
```

The CLI evaluates reviewed submissions and writes deterministic CSV exports;
there is no HTTP submission server in this repository.
