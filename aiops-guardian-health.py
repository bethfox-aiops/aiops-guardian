#!/usr/bin/env python3
"""
aiops-guardian-health.py

Guardian's health/security/AI-risk exporter entrypoint. The three engines
themselves live in guardian_health.py, guardian_security.py, and
guardian_ai_risk.py (shared state/utilities in guardian_common.py); this
file wires them together, computes the cross-cutting aiops_guardian_status
gauge (which needs both health_score and security_score), and runs the
collection loop.
"""

import os
import subprocess  # noqa: F401  (re-exported for test monkeypatching, see test_guardian_health.py)
import time

from prometheus_client import Gauge, start_http_server

from guardian_common import _get_ufw_status_cached, _prev, _ufw_status_cache  # noqa: F401
from guardian_health import compute_health, health_score
from guardian_security import _score_security_base, compute_security, get_ufw_enabled, security_score  # noqa: F401
from guardian_ai_risk import (
    AI_API_KEYS_PRESENT,
    AI_EXPOSED_KEYS,
    AI_GPU_SPIKE,
    AI_LLM_CONNECTIONS,
    AI_MODEL_AGE_DRIFT,
    AI_PROCESSES_RUNNING,
    AI_RISK_SCORE,
    AI_SHADOW_MODELS,
    AI_TOOLS_DETECTED,
    AI_TRAINING_CHANGED,
    AI_WATCHDOG_EXTERNAL,
    _ufw_denies_port_externally,  # noqa: F401
    calculate_ai_risk_score,
    check_gpu_spike_no_known_workload,
    check_model_file_age_drift,
    check_training_data_changed,
    check_watchdog_port_external_access,
    detect_ai_api_keys,
    detect_ai_packages,
    detect_ai_processes,
    get_exposed_api_keys,
    get_outbound_llm_connections,
    get_shadow_model_count,
)

PORT = int(os.environ.get("GUARDIAN_HEALTH_PORT", "8014"))

guardian_status = Gauge("aiops_guardian_status",  "Guardian overall status: 0=healthy, 1=needs attention, 2=critical")


def compute_guardian_status():
    h = health_score._value.get()
    s = security_score._value.get()
    if h < 80:
        guardian_status.set(2)
    elif s < 80:
        guardian_status.set(1)
    else:
        guardian_status.set(0)


def main():
    start_http_server(PORT)
    print(f"[INFO] Guardian health exporter running on port {PORT}")

    iteration = 0
    while True:
        compute_health()
        compute_security(iteration)
        compute_guardian_status()

        count, packages = detect_ai_packages()
        AI_TOOLS_DETECTED.set(count)
        print(f"[AI CHECK] installed_count={count}, packages={packages}")

        proc_count, proc_matches = detect_ai_processes()
        AI_PROCESSES_RUNNING.set(proc_count)
        print(f"[AI PROC] running_count={proc_count}")

        key_count, key_vars = detect_ai_api_keys()
        AI_API_KEYS_PRESENT.set(key_count)
        print(f"[AI API] key_count={key_count}")

        # ── New AI risk checks ──────────────────────────────────────────────
        watchdog_ext     = check_watchdog_port_external_access()
        exposed_keys     = get_exposed_api_keys()
        training_changed = check_training_data_changed()
        model_drift      = check_model_file_age_drift()
        gpu_spike        = check_gpu_spike_no_known_workload()

        # Expensive checks: LLM connections (dig DNS) and shadow model scan run every 20 iterations
        if iteration % 20 == 0:
            llm_conns = get_outbound_llm_connections()
            _prev["llm_conns"] = llm_conns
            shadow = get_shadow_model_count()
            _prev["shadow_model_count"] = shadow
        llm_conns = _prev.get("llm_conns", 0)
        shadow    = _prev["shadow_model_count"]

        AI_WATCHDOG_EXTERNAL.set(watchdog_ext)
        AI_EXPOSED_KEYS.set(exposed_keys)
        AI_LLM_CONNECTIONS.set(llm_conns)
        AI_SHADOW_MODELS.set(shadow)
        AI_TRAINING_CHANGED.set(training_changed)
        AI_MODEL_AGE_DRIFT.set(model_drift)
        AI_GPU_SPIKE.set(gpu_spike)

        if watchdog_ext:
            print("[AI-ALERT] Watchdog ports reachable from non-loopback IP!")
        if exposed_keys:
            print(f"[AI-ALERT] {exposed_keys} API key(s) exposed in files or env!")
        if llm_conns:
            print(f"[AI-ALERT] {llm_conns} outbound LLM API connection(s) detected!")
        if shadow:
            print(f"[AI-ALERT] {shadow} shadow model file(s) outside known directory!")
        if training_changed:
            print("[AI-ALERT] Training data head modified — possible poisoning!")
        if model_drift:
            print("[AI-ALERT] Model file timestamp changed without content change!")
        if gpu_spike:
            print("[AI-ALERT] GPU spike with no recognized training workload!")

        risk, reasons = calculate_ai_risk_score(
            count, proc_count, key_count,
            watchdog_external=watchdog_ext,
            exposed_keys=exposed_keys,
            llm_conns=llm_conns,
            shadow_models=shadow,
            training_changed=training_changed,
            model_age_drift=model_drift,
            gpu_spike=gpu_spike,
        )
        AI_RISK_SCORE.set(risk)
        print(f"[AI RISK] score={risk}, reasons={reasons}")

        iteration += 1
        time.sleep(30)


if __name__ == "__main__":
    main()
