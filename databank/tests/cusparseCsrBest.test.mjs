import assert from "node:assert/strict";
import test from "node:test";
import { deriveCusparseCsrBest } from "../src/views/SPMV/cusparseCsrBest.mjs";

const candidate = (matrix_id, configuration_id, solve_ms, dtype = "fp32") => ({
  candidate_group: "cusparse",
  configuration_id,
  dtype,
  matrix_id,
  method_name: "cuSPARSE CUDA 13.4.1 CSR candidate",
  solve_ms,
  status: "pass",
});

test("cuSPARSE CSR BEST compares only the registered CSR configurations per matrix", () => {
  const best = deriveCusparseCsrBest([
    candidate("m1", "csr-default", 3),
    candidate("m1", "csr-alg1", 2),
    candidate("m1", "csr-alg2", 1),
    candidate("m1", "coo-alg1", 0.1),
    candidate("m1", "csr-alg3", 0.01),
    candidate("m2", "csr-default", 0.5),
    candidate("m2", "csr-alg1", 1),
  ]);

  assert.deepEqual(best.map((row) => [row.matrix_id, row.selected_configuration_id]), [
    ["m1", "csr-alg2"],
    ["m2", "csr-default"],
  ]);
  assert.equal(best[0].selected_from, "csr-default,csr-alg1,csr-alg2");
  assert.equal(best[0].method_name, "cuSPARSE CUDA 13.4.1 CSR BEST");
});
