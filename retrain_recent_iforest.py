#!/usr/bin/env python3
"""
retrain_recent_iforest.py

Retrain the Isolation Forest anomaly detector on the most recent N rows of metrics.csv
so the current system state is the baseline.

Behavioral Attestation Phase 3: instrumented with OpenTelemetry so a
retrain run is a traceable workflow (load_data -> train_model -> save_model)
rather than just isolated log lines. Traces export to the local Tempo
instance.

The whole run flow (load_data / train_model / save_model / verify_behavior,
OTel spans, self-attribution monitoring, model archiving) is shared with
retrain_recent_knn.py and lives in retrain_common.run_simple_retrain(); this
file only supplies the IForest-specific constructor and names.
"""

from pyod.models.iforest import IForest

from retrain_common import run_simple_retrain

if __name__ == "__main__":
    run_simple_retrain(
        name="iforest",
        detector_label="Isolation Forest",
        model_factory=lambda: IForest(n_estimators=200, contamination=0.05, random_state=42),
        model_file="iforest_model.pkl",
        scaler_file="iforest_scaler.pkl",
        archive_prefix="iforest",
        service_name="aiops-watchdog-iforest",
    )
