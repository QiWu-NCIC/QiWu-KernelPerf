# Future: Protected Remote Evaluation

> Design draft. This is not part of the current release workflow.

The current workflow intentionally requires a maintainer to trigger evaluation.
A later GitHub Actions workflow can accept a reviewed PR SHA through
`workflow_dispatch`, authenticate with an environment-scoped secret, and submit
only that immutable SHA to a protected runner. The runner should use a short-
lived token, an allow-listed worker, isolated temporary storage, no outbound
network, CPU/GPU/time quotas, and an approval environment. Results should return
as an artifact or a pull request to `databank`; the action must never push directly
to the leaderboard without review.
