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
    AI_CHECK_LAST_SUCCESS,  # noqa: F401  (set inside safe_check)
    AI_EXPOSED_KEYS,
    AI_GPU_SPIKE,
    AI_LLM_CONNECTIONS,
    AI_MODEL_AGE_DRIFT,
    AI_PROCESSES_RUNNING,
    AI_RISK_COLLECTION_OK,
    AI_RISK_REASON,
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
    safe_check,
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


# Reason-label keys set on the previous cycle, so ones that clear can be
# zeroed instead of lingering as stale non-zero series in Prometheus.
_ai_risk_reason_keys: set = set()


def emit_ai_risk_reasons(factors):
    """Publish ai_risk_reason{reason=<key>} = <points> for each active factor,
    and 0 for any factor that was active last cycle but isn't now."""
    active = {f["key"]: f["points"] for f in factors}
    for key in _ai_risk_reason_keys - active.keys():
        AI_RISK_REASON.labels(reason=key).set(0)
    for key, points in active.items():
        AI_RISK_REASON.labels(reason=key).set(points)
    _ai_risk_reason_keys.clear()
    _ai_risk_reason_keys.update(active)


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

        # ── AI risk checks, each wrapped so one raising doesn't blank the
        #    rest and a failure is distinguishable from a verified zero. ──
        checks_ok = []

        def _check(name, fn, *a):
            val, ok = safe_check(name, fn, *a)
            checks_ok.append(ok)
            return val

        proc_count = _check("ai_processes", lambda: detect_ai_processes()[0])
        AI_PROCESSES_RUNNING.set(proc_count)
        print(f"[AI PROC] third_party_count={proc_count}")

        key_count = _check("api_keys_env", lambda: detect_ai_api_keys()[0])
        AI_API_KEYS_PRESENT.set(key_count)
        print(f"[AI API] key_count={key_count}")

        watchdog_ext     = _check("watchdog_port_external", check_watchdog_port_external_access)
        exposed_keys     = _check("exposed_api_keys", get_exposed_api_keys)
        training_changed = _check("training_data_tamper", check_training_data_changed)
        model_drift      = _check("model_file_age_drift", check_model_file_age_drift)
        gpu_spike        = _check("gpu_spike_no_workload", check_gpu_spike_no_known_workload)

        # Expensive checks (DNS resolution, filesystem walk) run every 20 iterations.
        if iteration % 20 == 0:
            llm_now, llm_ok = safe_check("outbound_llm_connections", get_outbound_llm_connections)
            checks_ok.append(llm_ok)
            if llm_ok:
                _prev["llm_conns"] = llm_now
            shadow_now, shadow_ok = safe_check("shadow_model_scan", get_shadow_model_count)
            checks_ok.append(shadow_ok)
            if shadow_ok:
                _prev["shadow_model_count"] = shadow_now
        llm_conns = _prev.get("llm_conns", 0)
        shadow    = _prev["shadow_model_count"]

        AI_WATCHDOG_EXTERNAL.set(watchdog_ext)
        AI_EXPOSED_KEYS.set(exposed_keys)
        AI_LLM_CONNECTIONS.set(llm_conns)
        AI_SHADOW_MODELS.set(shadow)
        AI_TRAINING_CHANGED.set(training_changed)
        AI_MODEL_AGE_DRIFT.set(model_drift)
        AI_GPU_SPIKE.set(gpu_spike)

        collection_ok = all(checks_ok)
        AI_RISK_COLLECTION_OK.set(1 if collection_ok else 0)
        if not collection_ok:
            print("[AI-ERR] one or more AI-risk sub-checks failed this cycle — score may be understated")

        risk, factors = calculate_ai_risk_score(
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
        emit_ai_risk_reasons(factors)
        print(f"[AI RISK] score={risk}, factors={[(f['key'], f['points']) for f in factors]}")

        iteration += 1
        time.sleep(30)


if __name__ == "__main__":
    main()
