#!/usr/bin/env python3
"""
retrain_recent_knn.py

Retrain the KNN anomaly detector on the most recent N rows of metrics.csv
so the current system state is the baseline.

Behavioral Attestation Phase 3: instrumented with OpenTelemetry so a
retrain run is a traceable workflow (load_data -> train_model -> save_model)
rather than just isolated log lines. Traces export to the local Tempo
instance.

The whole run flow (load_data / train_model / save_model / verify_behavior,
OTel spans, self-attribution monitoring, model archiving) is shared with
retrain_recent_iforest.py and lives in retrain_common.run_simple_retrain();
this file only supplies the KNN-specific constructor, names, and the
Grafana-annotation-on-policy-failure hook.
"""

from pyod.models.knn import KNN

from grafana_annotate import post_annotation
from retrain_common import run_simple_retrain


def _annotate_failure(violations):
    post_annotation(
        f"retrain_knn policy violation: {'; '.join(violations)}",
        tags=["guardian", "policy-violation", "retrain_knn"],
    )


if __name__ == "__main__":
    run_simple_retrain(
        name="knn",
        detector_label="KNN",
        model_factory=lambda: KNN(n_neighbors=5, method="largest", contamination=0.05),
        model_file="knn_model.pkl",
        scaler_file="scaler.pkl",
        archive_prefix="knn",
        service_name="aiops-watchdog-knn",
        on_fail=_annotate_failure,
    )
