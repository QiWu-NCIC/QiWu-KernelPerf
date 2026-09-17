# SpMV result submission inbox

Upload one KernelPerf schema-v2 CSV per configuration and precision here, then
open a pull request. Do not edit the public JSON indexes by hand. CI validates
the CSV schema, identities, matrix rows and derived solve-only metrics.

Before merge, a maintainer imports each accepted file with `npm run add:spmv`,
links the reviewed standalone source package, and removes the inbox copy. The
canonical CSV and generated index change are committed together.
