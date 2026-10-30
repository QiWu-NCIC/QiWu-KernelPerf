import test from "node:test";
import assert from "node:assert/strict";
import { validateFailureReason } from "../scripts/failure-reason.mjs";

test("passing rows keep the failure reason empty", () => {
  assert.doesNotThrow(() => validateFailureReason({ status: "pass", failure_reason: "" }));
  assert.throws(() => validateFailureReason({ status: "pass", failure_reason: "stale error" }));
});

test("failed rows require a bounded reason including explicit historical unknowns", () => {
  for (const status of ["fail", "error"]) {
    assert.doesNotThrow(() => validateFailureReason({ status, failure_reason: "setup: out of memory" }));
    assert.doesNotThrow(() => validateFailureReason({ status, failure_reason: "historical reason unavailable" }));
    assert.throws(() => validateFailureReason({ status, failure_reason: "" }));
    assert.throws(() => validateFailureReason({ status }));
    assert.throws(() => validateFailureReason({ status, failure_reason: "x".repeat(513) }));
  }
});
