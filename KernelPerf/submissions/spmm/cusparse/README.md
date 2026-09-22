# cuSPARSE CSR SpMM

The adapter calls the generic cuSPARSE SpMM API for CSR input and row-major
dense matrices. DEFAULT is retained for dispatch validation and excluded from
public BEST aggregation when it aliases an explicit CSR algorithm.
