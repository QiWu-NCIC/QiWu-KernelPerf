import assert from "node:assert/strict";
import test from "node:test";
import { selectContestEntries } from "../src/utils/contestData.js";

const entries = [
  { backend_id: "A100", dataset_id: "sample", dtype: "fp32", rhs_columns: 2 },
  { backend_id: "A100", dataset_id: "sample", dtype: "fp64", rhs_columns: 2 },
  { backend_id: "A100", dataset_id: "all", dtype: "fp32", rhs_columns: 2 },
  { backend_id: "H100", dataset_id: "sample", dtype: "fp32", rhs_columns: 2 },
  { backend_id: "A100", dataset_id: "sample", dtype: "fp32", rhs_columns: 128 },
];

test("SpMM loads both dtypes only for the selected platform, corpus and RHS", () => {
  assert.deepEqual(selectContestEntries(entries, { backend: "A100", dataset: "sample", rhs: 2 }), entries.slice(0, 2));
});

test("SpMV scope does not require an RHS field", () => {
  const spmv = entries.map(({ rhs_columns, ...entry }) => entry);
  assert.deepEqual(selectContestEntries(spmv, { backend: "A100", dataset: "all" }), [spmv[2]]);
});

test("an unavailable scope loads no unrelated results", () => {
  assert.deepEqual(selectContestEntries(entries, { backend: "Z100", dataset: "all", rhs: 2 }), []);
});
