#!/usr/bin/env python3
"""
aiops-watchdog-decisions.py

Exports whether a human decision is currently pending for each anomaly
watchdog, sourced from diagnose_anomaly.build_verdict() -- the exact same
function aiops-approval.py's web panel and generate_report.py's daily
report both call. One implementation of "is a decision needed" shared by
all three consumers (this panel's Prometheus/Grafana/alert path, the
approval panel, and the daily report), not three copies that could quietly
drift out of agreement -- this codebase has already been bitten once by
that exact class of bug (see retrain_common.py's RECENT_ROWS comment).

Exposes on WATCHDOG_PORT (default: 8021):
    aiops_pending_decision{model="knn"|"iforest"|"autoencoder"}  (0 or 1)

Checked every WATCHDOG_INTERVAL seconds (default: 60), not at the other
watchdogs' ~5s cadence -- diagnose_anomaly's checks include a real ~1s CPU
sample and a journalctl scan, too heavy to run that often for a signal
that doesn't change that fast anyway. Ground truth and the suspend/reboot
correlation check are each computed once per cycle and shared across all
three models, same reasoning as aiops-approval.py's get_model_rows().
"""
import os
import time

from prometheus_client import Gauge, start_http_server

import diagnose_anomaly

PORT = int(os.getenv("WATCHDOG_PORT", "8021"))
INTERVAL = float(os.getenv("WATCHDOG_INTERVAL", "60.0"))

pending_decision = Gauge(
    "aiops_pending_decision",
    "1 if diagnose_anomaly.build_verdict() currently recommends a retrain for this model, else 0",
    ["model"],
)


def main():
    start_http_server(PORT)
    print(f"[INFO] aiops-watchdog-decisions running on port {PORT}", flush=True)
    while True:
        try:
            gt = diagnose_anomaly.check_ground_truth()
            corr = diagnose_anomaly.check_suspend_or_reboot_correlation()
            for model in diagnose_anomaly.WATCHDOGS:
                try:
                    v = diagnose_anomaly.build_verdict(model, gt=gt, corr=corr)
                    pending_decision.labels(model=model).set(1 if v.get("recommend_retrain") else 0)
                except Exception as e:
                    print(f"[WARN] build_verdict({model}) failed: {e}", flush=True)
        except Exception as e:
            print(f"[WARN] ground-truth/correlation check failed this cycle: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
