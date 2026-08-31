# SpMV configuration sweeps

Give every candidate a unique `configuration_id` and put comparable variants
in one `candidate_group`. Select one or more configurations with the CLI:

```bash
python -m kernelperf.cli evaluate \
  --config config/service.json \
  --submission submissions/spmv/cusparse \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --operator spmv.csr.fp32 \
  --configuration-id csr-default \
  --configuration-id csr-alg1
```

Each configuration gets its own CSV. The maintainer-side BEST aggregation
selects the smallest passing `solve_ms` per matrix and records the selected
configuration ID. See `scripts/aggregate_best.py` and
[`docs/REGRESSION.md`](../docs/REGRESSION.md).
