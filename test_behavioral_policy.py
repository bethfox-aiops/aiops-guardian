"""
test_behavioral_policy.py

Tests for behavioral_policy.py (Phase 5 runtime verification). The key
scenario here (test_row_count_below_minimum_fails) is a direct regression
test for the real defect-demo from 2026-07-17: a deliberately injected
defect (KNN retrain window dropped from 2000 to 20 rows) that Guardian's
own policy verification correctly caught.

Run with: pytest test_behavioral_policy.py -v
"""

from behavioral_policy import verify


def _passing_knn_evidence(**overrides):
    """A baseline set of evidence that should fully satisfy retrain_knn's
    policy -- individual tests override just the field they're checking."""
    evidence = dict(
        files_touched={"knn_model.pkl", "scaler.pkl"},
        gpu_mib=0,
        network_connects=0,
        row_count=2000,
    )
    evidence.update(overrides)
    return evidence


class TestVerify:
    def test_unknown_workflow_fails(self):
        result = verify("not_a_real_workflow", files_touched=set(), gpu_mib=0,
                         network_connects=0, row_count=2000)
        assert result["passed"] is False
        assert "no policy defined for workflow 'not_a_real_workflow'" in result["violations"][0]

    def test_fully_passing_run(self):
        result = verify("retrain_knn", **_passing_knn_evidence())
        assert result["passed"] is True
        assert result["violations"] == []

    def test_row_count_below_minimum_fails(self):
        # Regression test for the real 2026-07-17 defect-demo: KNN's
        # retrain window was deliberately dropped from 2000 to 20 rows,
        # and Guardian's own verification correctly caught it.
        result = verify("retrain_knn", **_passing_knn_evidence(row_count=20))
        assert result["passed"] is False
        assert "row_count 20 below policy minimum 100" in result["violations"]

    def test_row_count_above_maximum_fails(self):
        result = verify("retrain_knn", **_passing_knn_evidence(row_count=999999))
        assert result["passed"] is False
        assert any("above policy maximum" in v for v in result["violations"])

    def test_missing_expected_file_fails(self):
        result = verify("retrain_knn", **_passing_knn_evidence(files_touched={"scaler.pkl"}))
        assert result["passed"] is False
        assert any("knn_model.pkl" in v for v in result["violations"])

    def test_gpu_usage_exceeds_cap_fails(self):
        # retrain_knn is CPU-only -- max_gpu_mib is 0, so ANY GPU use should fail.
        result = verify("retrain_knn", **_passing_knn_evidence(gpu_mib=1))
        assert result["passed"] is False
        assert any("GPU usage" in v for v in result["violations"])

    def test_network_connects_exceed_cap_fails(self):
        result = verify("retrain_knn", **_passing_knn_evidence(network_connects=1))
        assert result["passed"] is False
        assert any("outbound connect()" in v for v in result["violations"])

    def test_autoencoder_has_no_gpu_cap(self):
        # Unlike KNN (max_gpu_mib=0), retrain_autoencoder's policy sets
        # max_gpu_mib=None -- GPU use should NOT be flagged for it.
        result = verify(
            "retrain_autoencoder",
            files_touched={"autoencoder_model.pkl", "autoencoder_scaler.pkl", "autoencoder_threshold.txt"},
            gpu_mib=500,  # would fail retrain_knn's policy, should pass here
            network_connects=0,
            row_count=100000,
        )
        assert result["passed"] is True

    def test_multiple_violations_all_reported(self):
        result = verify("retrain_knn", files_touched=set(), gpu_mib=999,
                         network_connects=5, row_count=1)
        assert result["passed"] is False
        # missing files, GPU cap, network cap, and row_count minimum should ALL fire
        assert len(result["violations"]) == 4
