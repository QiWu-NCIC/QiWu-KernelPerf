# QiWu-KernelPerf

Unified repository for the QiWu kernel evaluation runtime, benchmark submissions,
and the static databank leaderboard.

## Layout

- `KernelPerf/`: Python evaluation runtime, backend configurations, benchmark drivers,
  submissions, result export, and tests.
- `KernelPerf/submissions/<operator>/<method>/`: reviewed source or architecture-
  specific object submissions. SpMV submissions are under `submissions/spmv/`.
- `databank/`: Vue/Vite static leaderboard and immutable result CSV catalog.
- `docs/`: deployment, PR review, dataset, and future automation notes.

The public server does not accept arbitrary user submissions. A pull request is
validated by GitHub Actions, then a maintainer checks out the PR on a worker,
runs the local KernelPerf evaluator, and commits the resulting CSV to `databank/`.

## Local development

```bash
cd KernelPerf
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

```bash
cd databank
npm ci
npm run dev
```

Node.js 20+ is required for the databank build.

