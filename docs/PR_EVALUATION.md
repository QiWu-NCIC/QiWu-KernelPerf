# Pull Request Evaluation

Users add one directory under `KernelPerf/submissions/<operator>/<method>/` and
open a pull request. The directory must contain `submission.json` and either a
UTF-8 CUDA source tree or an ELF relocatable object. Source is the normal mode;
objects are accepted only when their declared CUDA architecture matches the
worker being tested.

GitHub Actions performs schema, path, size, and compile smoke checks. It does not
connect to evaluation servers. After review, a maintainer checks out the PR on a
worker and runs:

```bash
cd KernelPerf
python -m kernelperf.cli evaluate \
  --submission submissions/spmv/my-method \
  --backend A100-SXM4-80GB \
  --dataset-id suitesparse_sample_100 \
  --operator spmv.csr.fp32
```

The evaluator writes canonical CSV files below `KernelPerf/data/result_exports/`.
The maintainer copies those CSVs into `databank/public/data/results/`, updates
`databank/public/data/index.json`, runs the audit, and commits the result.

All execution of PR code must use a restricted worker account, a read-only source
checkout, no network access, and explicit process/time limits.

