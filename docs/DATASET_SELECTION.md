# SuiteSparse Datasets

`suitesparse_validation_100` is the legacy small-matrix set retained for
correctness and smoke tests. Existing result CSVs use this dataset ID.

`suitesparse_sample_100` is the performance corpus selected from the official
SuiteSparse index. Generate it with:

```bash
cd KernelPerf
python scripts/select_suitesparse.py \
  --output config/datasets/suitesparse_sample_100.json \
  --metadata-output config/datasets/suitesparse_sample_100.selection.json
```

The selector keeps real matrices with `100 <= nnz <= 100000000`, allocates
deterministic quotas across logarithmic NNZ strata, and diversifies by average
row length, structural kind, symmetry, 2D/3D classification and
positive-definiteness. Local archives add measured row-length variance.

Download archives on a worker with:

```bash
python scripts/download_matrix_manifest.py \
  --manifest config/datasets/suitesparse_sample_100.json \
  --root /path/to/suitesparse \
  --proxy http://127.0.0.1:7890
```

The Matrix Market reader accepts real, integer and pattern coordinate input;
expands symmetric, Hermitian and skew-symmetric matrices; accepts duplicate
entries; and sorts CSR entries deterministically. Complex input is projected to
the real component and recorded, but complex matrices are excluded from the
performance corpus.

