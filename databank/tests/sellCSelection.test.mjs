import assert from "node:assert/strict";
import test from "node:test";
import {
  isPublicSpmvSellRow,
  sellCConfiguration,
  sellCWinnerByScope,
} from "../src/utils/sellCSelection.js";

function rowsFor(c, algorithm, dtype, count, score) {
  return Array.from({ length: count }, (_, index) => ({
    backend_id: "A100",
    dataset_id: "suite",
    matrix_id: `matrix-${index}`,
    candidate_group: "cusparse",
    configuration_id: `sell-c${c}-${algorithm}`,
    dtype,
    status: "pass",
    score,
  }));
}

test("SELL C selection matches the FP32/FP64 efficiency geomean and needs all algorithms", () => {
  const input = [
    ...["alg1", "default"].flatMap((algorithm) => [
      ...rowsFor(8, algorithm, "fp32", 100, 10),
      ...rowsFor(8, algorithm, "fp64", 100, 100),
      ...rowsFor(16, algorithm, "fp32", 100, 40),
      ...rowsFor(16, algorithm, "fp64", 100, 40),
      ...rowsFor(32, algorithm, "fp32", 80, 1000),
      ...rowsFor(32, algorithm, "fp64", 80, 1000),
    ]),
  ];
  const winners = sellCWinnerByScope(input, (row) => row.score, () => 100);

  assert.equal(winners.get("A100|suite").c, 16);
  assert.equal(isPublicSpmvSellRow(input.find((row) => row.configuration_id === "sell-c16-alg1"), winners), true);
  assert.equal(isPublicSpmvSellRow(input.find((row) => row.configuration_id === "sell-c8-alg1"), winners), false);
});

test("SELL C parser excludes NROWS and non-SELL configurations", () => {
  assert.equal(sellCConfiguration("sell-c128-default"), 128);
  assert.equal(sellCConfiguration("sell-nrows-default"), null);
  assert.equal(sellCConfiguration("csr-alg1"), null);
});

test("partial legacy scopes keep their existing SELL rows visible", () => {
  const rows = rowsFor(16, "default", "fp32", 100, 20);
  const winners = sellCWinnerByScope(rows, (row) => row.score, () => 100);
  assert.equal(winners.size, 0);
  assert.equal(isPublicSpmvSellRow(rows[0], winners), true);
});

test("failed rows count toward coverage but not performance", () => {
  const input = [
    ...["alg1", "default"].flatMap((algorithm) => ["fp32", "fp64"].flatMap((dtype) => [
      ...rowsFor(4, algorithm, dtype, 99, 20),
      {
        ...rowsFor(4, algorithm, dtype, 1, 0)[0],
        matrix_id: "failed-matrix",
        status: "fail",
      },
    ])),
  ];
  const winners = sellCWinnerByScope(input, (row) => row.score, () => 100);
  assert.equal(winners.get("A100|suite").c, 4);
});

test("a C candidate with missing matrix rows cannot become the public winner", () => {
  const input = [
    ...["alg1", "default"].flatMap((algorithm) => [
      ...rowsFor(4, algorithm, "fp32", 100, 20),
      ...rowsFor(4, algorithm, "fp64", 99, 20),
    ]),
  ];
  const winners = sellCWinnerByScope(input, (row) => row.score, () => 100);

  assert.equal(winners.size, 0);
});
