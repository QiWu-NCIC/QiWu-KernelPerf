# SpMV Configuration Sweeps

A sweep submits several `KernelArtifact` entries for the same operator. Every
entry must declare a unique `metadata.configuration_id`; entries that should be
compared must share `metadata.candidate_group`.

The service evaluates every configuration on every selected matrix. At terminal
job completion it writes one candidate CSV per configuration and, when a group
has at least two configurations, one additional CSV with
`configuration_id=per-matrix-best`. For each matrix, BEST keeps the passing row
with the smallest `solve_ms`; ties are resolved by configuration ID.

The CLI accepts a JSON sweep manifest:

```bash
python scripts/submit_spmv.py \
  --api http://127.0.0.1:18081 \
  --backend A100-SXM4-80GB \
  --operator spmv.csr.fp32 \
  --base-format csr \
  --candidate-group alpha-csr \
  --source-dir examples/alphasparse_submission \
  --sweep-manifest examples/alphasparse_submission/sweep.json \
  --wait --publish
```

The generated BEST CSV can be downloaded with
`configuration_id=per-matrix-best` and published through the same API. Static
databank entries retain `selected_from` so the selected candidate set remains
auditable.
