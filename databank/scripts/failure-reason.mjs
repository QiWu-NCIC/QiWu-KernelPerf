export function validateFailureReason(row) {
  if (typeof row.failure_reason !== "string") throw new Error("missing failure_reason");
  if (row.failure_reason.length > 512) throw new Error("failure_reason exceeds 512 characters");
  if (row.status === "pass" && row.failure_reason !== "") {
    throw new Error("passing row has a failure_reason");
  }
  if (row.status !== "pass" && !row.failure_reason.trim()) {
    throw new Error("failed row has no failure_reason");
  }
}
