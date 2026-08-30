#!/usr/bin/env bash
set -euo pipefail

API="${API:-http://10.18.96.188:8080}"
SOURCE="${SOURCE:-examples/spmv_template.cu}"
BASE_FORMAT="${BASE_FORMAT:-csr}"

RESPONSE="$(
  jq -n --rawfile source "${SOURCE}" --arg base_format "${BASE_FORMAT}" '{
    generator_id: "curl-spmv-test",
    submitter: "curl-client",
    backends: [
      "rtx5090-workstation",
      "A100-SXM4-80GB"
    ],
    suites: ["spmv"],
    dataset_id: "suitesparse_sample_100",
    operator_ids: ["spmv.csr.fp32"],
    kernels: [{
      name: "csr-spmv-fp32",
      source: $source,
      metadata: {
        operator_id: "spmv.csr.fp32",
        base_format: $base_format
      }
    }],
    priority: 50,
    tags: {
      benchmark: "spmv",
      dataset: "suitesparse_sample_100"
    }
  }' |
  curl -fsS -X POST "${API}/api/v1/jobs" \
    -H 'content-type: application/json' \
    --data-binary @-
)"

printf '%s\n' "${RESPONSE}" | jq .
JOB_ID="$(printf '%s\n' "${RESPONSE}" | jq -r '.job_id')"

while true; do
  JOB="$(curl -fsS "${API}/api/v1/jobs/${JOB_ID}")"
  STATUS="$(printf '%s\n' "${JOB}" | jq -r '.status')"
  echo "${JOB_ID}: ${STATUS}"

  case "${STATUS}" in
    succeeded|failed|cancelled)
      printf '%s\n' "${JOB}" | jq .
      break
      ;;
  esac
  sleep 2
done

curl -fsS "${API}/api/v1/results?job_id=${JOB_ID}" | jq .
