# QiWu legacy rocSPARSE CSRMV SpMV plugin

This comparison plugin calls the typed `rocsparse_scsrmv` and
`rocsparse_dcsrmv` API with a null analysis-info pointer. That is the API path
used by the archived Z100 benchmark. The plugin keeps KernelPerf's standard
SpMV semantics (`alpha=1`, `beta=0`) and timing protocol, so differences caused
by the old benchmark's host synchronization remain visible in the comparison.

The display name is `rocSPARSE Legacy CSRMV (No Analysis)`. This submission
has its own candidate group and is not included in the modern generic-API
rocSPARSE CSR BEST.
