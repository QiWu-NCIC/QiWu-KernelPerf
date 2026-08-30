# KernelPerf

KernelPerf is the maintainer-run evaluator used by QiWu-KernelPerf. Reviewed
submission directories live under `submissions/<operator>/` and are evaluated
with the CLI after a pull request is merged or manually checked out.

```bash
python -m pip install -e '.[dev]'
python -m kernelperf.cli evaluate \
  --config config/service.json \
  --submission submissions/spmv/cusparse \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --operator spmv.csr.fp32
```

The evaluator writes one normalized CSV per method/backend/dataset/dtype under
`data/result_exports/`. The static site in the sibling `databank/` directory
consumes those files; no HTTP submission service is part of this repository.
