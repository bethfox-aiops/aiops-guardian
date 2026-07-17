#!/usr/bin/env python3
"""
behavioral_policy.py

Behavioral Attestation Phase 5: runtime behavioral verification.

Defines expected-behavior policies per AI workflow and checks a completed
run's evidence (already gathered by Phases 1/2/3/4) against them, flagging
deviations as verification failures -- a distinct signal from the
statistical anomaly detectors, which only ever say "this looks unusual,"
never "this violated a stated expectation."
"""

POLICIES = {
    "retrain_knn": {
        "expected_files": {"knn_model.pkl", "scaler.pkl"},
        "max_gpu_mib": 0,            # KNN is CPU-only; any GPU use is unexpected
        "max_network_connects": 0,   # a retrain run should not open outbound connections
        "min_rows": 100,             # sanity floor -- something is very wrong below this
        "max_rows": 5000,            # RECENT_ROWS is 2000; flag if wildly different
    },
    "retrain_iforest": {
        "expected_files": {"iforest_model.pkl", "iforest_scaler.pkl"},
        "max_gpu_mib": 0,
        "max_network_connects": 0,
        "min_rows": 100,
        "max_rows": 5000,
    },
    "retrain_autoencoder": {
        "expected_files": {"autoencoder_model.keras", "autoencoder_scaler.pkl", "autoencoder_threshold.txt"},
        "max_gpu_mib": None,         # no cap -- GPU use is expected/fine here if CUDA ever gets fixed
        "max_network_connects": 0,
        "min_rows": 500,
        "max_rows": 150000,          # RECENT_ROWS is 100000
    },
}


def verify(workflow_name, *, files_touched, gpu_mib, network_connects, row_count):
    """
    Check a completed workflow run's evidence against its policy.
    Returns {"passed": bool, "violations": [str, ...]}.
    """
    policy = POLICIES.get(workflow_name)
    if policy is None:
        return {"passed": False, "violations": [f"no policy defined for workflow '{workflow_name}'"]}

    violations = []
    files_touched = set(files_touched)

    expected_files = policy.get("expected_files")
    if expected_files is not None:
        missing = expected_files - files_touched
        if missing:
            violations.append(f"expected files not freshly written: {sorted(missing)}")

    max_gpu = policy.get("max_gpu_mib")
    if max_gpu is not None and gpu_mib > max_gpu:
        violations.append(f"GPU usage {gpu_mib:.1f} MiB exceeds policy max {max_gpu} MiB")

    max_conn = policy.get("max_network_connects")
    if max_conn is not None and network_connects > max_conn:
        violations.append(
            f"{network_connects} outbound connect() calls exceeds policy max {max_conn} "
            "(sampled during a 3s eBPF window mid-training, not exhaustive coverage of the full run)"
        )

    min_rows = policy.get("min_rows")
    if min_rows is not None and row_count < min_rows:
        violations.append(f"row_count {row_count} below policy minimum {min_rows}")

    max_rows = policy.get("max_rows")
    if max_rows is not None and row_count > max_rows:
        violations.append(f"row_count {row_count} above policy maximum {max_rows}")

    return {"passed": len(violations) == 0, "violations": violations}
