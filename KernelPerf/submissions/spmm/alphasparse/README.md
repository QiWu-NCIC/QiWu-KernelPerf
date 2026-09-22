# AlphaSparse CSR SpMM

The adapter invokes the pinned upstream AlphaSparse level3 SpMM dispatcher for
CSR algorithms 1 through 5. It does not reimplement those kernels or emulate
SpMM with repeated SpMV calls.
