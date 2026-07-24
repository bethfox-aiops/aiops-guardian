#!/usr/bin/env python3
"""
guardian_common.py

Shared constants, change-detection state, and utilities used by both
guardian_security.py and guardian_ai_risk.py: file/string hashing, the
generic subprocess-capture helper, the `_prev` dict that both modules'
checks use to detect changes since the last cycle, and the cached
`sudo ufw status` output (shared because both compute_security()'s
get_ufw_enabled() and guardian_ai_risk's check_watchdog_port_external_access()
need ufw state, and used to each shell out separately before this was
cached).
"""

import hashlib
import subprocess
import time

MODEL_DIR  = "/home/beth/aiops-agents"
DATA_FILE  = "/home/beth/aiops-agents/aiops_data/metrics.csv"
MODEL_FILES = [
    "autoencoder_model.keras",
    "autoencoder_scaler.pkl",
    "autoencoder_threshold.txt",
    "knn_model.pkl",
    "scaler.pkl",
    "iforest_model.pkl",
    "iforest_scaler.pkl",
]

# ─── Change-detection state ──────────────────────────────────────────────────
_prev = {
    "ssh_keys_hash": None,
    "user_count": None,
    "cron_hash": None,
    "systemd_hash": None,
    "suid_count": None,
    "passwd_mtime": None,
    "sudoers_mtime": None,
    "shadow_mtime": None,
    "kernel_module_count": None,
    "model_hashes": {},
    "world_writable": 0,
    "pkg_failures": 0,
    # AI risk state
    "training_data_head_hash": None,
    "model_age_mtimes": {},
    "model_age_hashes": {},
    "shadow_model_count": 0,
}


# ════════════════════════════════════════════════════════════════════════════
# Utility
# ════════════════════════════════════════════════════════════════════════════

def _hash_file(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def _hash_string(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _run(cmd, timeout=15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        return r.stdout
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════════════════
# Cached `sudo ufw status`
# ════════════════════════════════════════════════════════════════════════════

_ufw_status_cache = {"ts": 0.0, "out": ""}

def _get_ufw_status_cached(max_age: float = 25.0) -> str:
    """`sudo ufw status` output, cached briefly so the once-per-cycle callers
    (get_ufw_enabled, _ufw_denies_port_externally x4 ports) share one sudo call."""
    now = time.time()
    if now - _ufw_status_cache["ts"] > max_age:
        try:
            r = subprocess.run(["sudo", "ufw", "status"],
                               capture_output=True, text=True, check=False, timeout=5)
            _ufw_status_cache["out"] = r.stdout
        except Exception:
            _ufw_status_cache["out"] = ""
        _ufw_status_cache["ts"] = now
    return _ufw_status_cache["out"]
