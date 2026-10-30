from kernelperf.benchmark import BenchmarkDriverError, failure_metadata, failure_reason


def test_structured_log_reason_overrides_generic_process_error():
    metadata = failure_metadata(BenchmarkDriverError(
        "spmv-lifecycle", "run", "process exited with returncode=1",
        stdout='log line\n{"status":"error","stage":"setup","error":"CSR capacity exceeded"}',
    ))
    assert metadata["failure_reason"] == "setup: CSR capacity exceeded"


def test_text_log_reason_retains_compiler_diagnostic_without_machine_path():
    reason = failure_reason({
        "error": "compilation failed",
        "failure_stage": "compile",
        "stderr_tail": "/private/build/adapter.cu:42: error: unsupported type\n1 error detected",
    })
    assert "error" in reason
    assert "unsupported type" in reason
    assert reason.startswith("compile:")
    assert "/private/" not in reason


def test_timeout_reason_does_not_publish_executable_path():
    reason = failure_reason({
        "failure_stage": "run",
        "error": "Command '['/private/build/benchmark']' timed out after 1200 seconds",
    })
    assert reason == "run: benchmark command timed out after 1200 seconds"


def test_precision_reason_retains_mismatch_details():
    reason = failure_reason({
        "validation": {"status": "precision-error"},
        "measurement": {"mismatch_count": 3, "first_mismatch_row": 12,
                        "first_mismatch_absolute_error": 0.25,
                        "first_mismatch_bound": 0.01},
    })
    assert "failed_elements=3" in reason
    assert "first_mismatch_row=12" in reason
    assert "first_mismatch_bound=0.01" in reason


def test_invalid_output_reason_retains_nonfinite_counts():
    assert failure_reason({
        "validation_status": "invalid-output",
        "invalid_elements": 5, "nan_elements": 2, "inf_elements": 3,
    }) == "invalid output: invalid_elements=5, NaN=2, Inf=3"


def test_unknown_historical_reason_is_explicit():
    assert failure_reason({}).startswith("historical reason unavailable")


def test_existing_reason_is_bounded_and_secrets_are_redacted():
    reason = failure_reason({"failure_reason": "password=private-token " + "x" * 700})
    assert len(reason) == 512
    assert "private-token" not in reason
