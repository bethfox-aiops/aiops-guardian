#!/usr/bin/env python3
"""
guardian_ai_risk.py

Guardian's AI risk engine: detects installed AI tooling/processes/API keys
plus the newer risk signals (exposed keys, outbound LLM connections, shadow
models, training-data tampering, model file drift, GPU spikes, externally
reachable watchdog ports), combined into the ai_* gauges by
calculate_ai_risk_score().
"""

import glob
import hashlib
import os
import socket
import time
from importlib import metadata

import psutil
from prometheus_client import Gauge

from guardian_common import (
    DATA_FILE,
    MODEL_DIR,
    MODEL_FILES,
    _get_ufw_status_cached,
    _hash_file,
    _prev,
    _run,
)

# ─── AI detection gauges ────────────────────────────────────────────────────
AI_TOOLS_DETECTED    = Gauge("ai_tools_detected",     "Number of AI-related Python packages installed (informational, not a risk penalty)")
AI_PROCESSES_RUNNING = Gauge("ai_processes_running",  "Number of third-party AI runtimes running (Guardian's own processes excluded)")
AI_API_KEYS_PRESENT  = Gauge("ai_api_keys_present",   "Number of AI-related API key environment variables in this process")
AI_RISK_SCORE        = Gauge("ai_risk_score",         "Overall AI risk score (0-100)")

# ─── AI Risk gauges (new) ────────────────────────────────────────────────────
AI_WATCHDOG_EXTERNAL    = Gauge("ai_watchdog_port_external_access", "1 if watchdog ports reachable from non-loopback IP")
AI_EXPOSED_KEYS         = Gauge("ai_exposed_api_keys",              "API keys found on disk (dotfiles, config, systemd units, tracked in git)")
AI_LLM_CONNECTIONS      = Gauge("ai_outbound_llm_connections",      "Active connections to resolved LLM API endpoint IPs")
AI_SHADOW_MODELS        = Gauge("ai_shadow_model_count",            "Model files found outside the known model directory")
AI_TRAINING_CHANGED     = Gauge("ai_training_data_hash_changed",    "1 if training CSV changed in a way that isn't a plain append")
AI_MODEL_AGE_DRIFT      = Gauge("ai_model_file_age_drift",          "1 if a model file mtime changed without its content changing")
AI_GPU_SPIKE            = Gauge("ai_gpu_spike_no_known_workload",   "1 if GPU above 20% with no known training job running")

# ─── Collection-integrity gauges (added 2026-09-10) ─────────────────────────
# Without these, a sub-check that raises and returns 0 is indistinguishable
# from a check that ran and verified "clear" -- the same failure mode the
# logs watchdog fixed with aiops_logs_query_ok.
AI_RISK_COLLECTION_OK  = Gauge("ai_risk_collection_ok",            "1 if every AI-risk sub-check completed this cycle without raising")
AI_CHECK_LAST_SUCCESS  = Gauge("ai_check_last_success_timestamp",  "Unix ts of the last successful run of each AI-risk sub-check", ["check"])
AI_RISK_REASON         = Gauge("ai_risk_reason",                   "Points currently deducted by each named AI-risk factor (0 when inactive)", ["reason"])

# Last-known-good value per check, so a transient failure falls back to the
# previous reading instead of a misleading 0.
_check_last_good: dict = {}


def safe_check(name, fn, *args, default=0):
    """Run an AI-risk sub-check, recording success/failure so a raised
    exception is distinguishable from a verified-clear result.

    Returns (value, ok). On failure, returns the check's last known-good
    value (or `default` if it has never succeeded) and ok=False."""
    try:
        val = fn(*args)
        AI_CHECK_LAST_SUCCESS.labels(check=name).set(time.time())
        _check_last_good[name] = val
        return val, True
    except Exception as e:  # noqa: BLE001 -- deliberately broad; one bad check must not kill the cycle
        print(f"[AI-ERR] sub-check {name!r} raised: {e!r}")
        return _check_last_good.get(name, default), False


# ════════════════════════════════════════════════════════════════════════════
# AI Risk Detection
# ════════════════════════════════════════════════════════════════════════════

def _ufw_denies_port_externally(port: int) -> bool:
    """Return True if ufw has an explicit DENY-from-Anywhere rule for port
    with no competing ALLOW-from-Anywhere rule (i.e. ufw itself blocks it,
    regardless of what the socket is bound to)."""
    out = _get_ufw_status_cached()
    if "Status: active" not in out:
        return False
    deny_anywhere = False
    allow_anywhere = False
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0].split("/")[0] != str(port):
            continue
        # "(v6)" after the port shifts every later column by one, e.g.
        # "8011 (v6)  DENY  Anywhere (v6)  # comment"
        idx = 2 if parts[1] == "(v6)" else 1
        if idx >= len(parts):
            continue
        action = parts[idx]
        source = " ".join(parts[idx + 1:]).split("#")[0].replace("(v6)", "").strip()
        if source.lower() != "anywhere":
            continue
        if action == "DENY":
            deny_anywhere = True
        elif action == "ALLOW":
            allow_anywhere = True
    return deny_anywhere and not allow_anywhere


def check_watchdog_port_external_access() -> int:
    """Return 1 if any watchdog port is reachable via the machine's non-loopback IP."""
    watchdog_ports = [8011, 8012, 8013, 8014]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
    except Exception:
        return 0
    if local_ip.startswith("127."):
        return 0
    for port in watchdog_ports:
        if _ufw_denies_port_externally(port):
            continue
        try:
            with socket.create_connection((local_ip, port), timeout=1):
                return 1
        except Exception:
            pass
    return 0


_API_KEY_FILE_PATTERNS = [
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
    "COHERE_API_KEY", "MISTRAL_API_KEY", "REPLICATE_API_TOKEN",
]

_REPO_DIR = "/home/beth/aiops-agents"


def _git_tracks_secret_file() -> bool:
    """True if a .env / secrets file is actually committed to the repo -- a
    much stronger signal than one merely existing on disk."""
    out = _run(["git", "-C", _REPO_DIR, "ls-files"], timeout=5)
    for line in out.splitlines():
        base = os.path.basename(line).lower()
        if base in (".env", ".env.local", ".env.production", "secrets.env", "credentials"):
            return True
        if base.endswith(".env"):
            return True
    return False


def get_exposed_api_keys() -> int:
    """Count distinct places an AI API key is exposed on this host: shell
    dotfiles, ~/.config, ~/.netrc, systemd unit Environment=/EnvironmentFile=
    lines, and repo .env files -- plus a flag if such a file is committed to
    git. The scan is bounded (fixed path list + shallow globs) so it stays
    cheap enough for the 30s health loop."""
    found = set()

    # Env vars in this process
    for v in _API_KEY_FILE_PATTERNS:
        if os.environ.get(v):
            found.add(f"env:{v}")

    check_files = [
        os.path.expanduser("~/.env"),
        os.path.expanduser("~/.env.local"),
        os.path.expanduser("~/.bashrc"),
        os.path.expanduser("~/.bash_profile"),
        os.path.expanduser("~/.bash_aliases"),
        os.path.expanduser("~/.profile"),
        os.path.expanduser("~/.zshrc"),
        os.path.expanduser("~/.netrc"),
        os.path.expanduser("~/.pam_environment"),
        os.path.expanduser("~/.config/environment.d/99-personal.conf"),
        f"{_REPO_DIR}/.env",
        f"{_REPO_DIR}/.env.local",
    ]
    # Shallow globs: systemd user/system units and environment.d fragments,
    # where an EnvironmentFile= or Environment= line can carry a key.
    check_files += glob.glob(os.path.expanduser("~/.config/systemd/user/*.service"))
    check_files += glob.glob(os.path.expanduser("~/.config/environment.d/*.conf"))
    check_files += glob.glob("/etc/systemd/system/aiops-*.service")

    for path in check_files:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for pattern in _API_KEY_FILE_PATTERNS:
                        if pattern in stripped and ("=" in stripped or " " in stripped):
                            tail = stripped.split(pattern, 1)[1].lstrip("=: \t\"'")
                            if tail and not tail.startswith(("$", "%")):  # not just a var reference
                                found.add(f"file:{path}:{pattern}")
        except Exception:
            pass

    if _git_tracks_secret_file():
        found.add("git:tracked-secret-file")

    return len(found)


# Only hosts that resolve to provider-dedicated or per-zone-anycast IPs.
# Deliberately excluded because their IPs are shared with huge amounts of
# unrelated traffic, which would make this check fire on any Google/AWS
# connection:
#   generativelanguage.googleapis.com / aiplatform.googleapis.com
#       -> shared Google front-end ranges (Gmail, Search, YouTube, ...)
#   bedrock-runtime.*.amazonaws.com
#       -> shared AWS service ranges
# A stronger future version would confirm via TLS SNI rather than IP alone.
_LLM_API_HOSTS = [
    "api.openai.com", "api.anthropic.com", "api.cohere.ai", "api.cohere.com",
    "api.mistral.ai", "api.together.xyz", "api.together.ai", "api.replicate.com",
    "api.perplexity.ai", "api.groq.com", "api.deepseek.com", "api.x.ai",
    "openrouter.ai", "api-inference.huggingface.co", "huggingface.co",
]

# ip -> unix ts last resolved. CDN-fronted APIs rotate IPs, and a live
# connection may sit on an IP that a fresh lookup no longer returns, so we
# keep a rolling union of everything resolved in the last hour rather than
# only the current answer.
_llm_ip_cache: dict = {}
_LLM_IP_TTL = 3600.0


def _known_llm_ips() -> set:
    now = time.time()
    for host in _LLM_API_HOSTS:
        try:
            for res in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
                _llm_ip_cache[res[4][0]] = now
        except OSError:
            pass  # resolution failure for one host shouldn't blank the set
    for ip in [k for k, ts in _llm_ip_cache.items() if now - ts > _LLM_IP_TTL]:
        del _llm_ip_cache[ip]
    return set(_llm_ip_cache)


def get_outbound_llm_connections() -> int:
    """Count established connections whose remote IP resolves to a known LLM
    API host. Replaces the old reverse-DNS match, which failed whenever a
    provider sat behind Cloudflare/Fastly (PTR says 'cloudflare', not the
    provider) -- i.e. almost always."""
    llm_ips = _known_llm_ips()
    if not llm_ips:
        return 0
    count = 0
    culprits = []
    try:
        for c in psutil.net_connections(kind="inet"):
            if (c.status == "ESTABLISHED" and c.raddr
                    and c.raddr.ip in llm_ips and c.raddr.port in (80, 443)):
                count += 1
                culprits.append(c.pid)
    except Exception:
        return 0
    if count:
        print(f"[AI CHECK] {count} outbound LLM connection(s), pids={culprits}")
    return count


_SHADOW_MODEL_EXTS = {".pkl", ".pt", ".pth", ".onnx", ".h5", ".safetensors"}  # .keras excluded (is a dir)
_SHADOW_SKIP_DIRS  = {
    os.path.abspath(MODEL_DIR),
    "/opt/aiops-venv",
    os.path.expanduser("~/aiops-watchdog-k8s"),  # own project, legitimate models
}

def get_shadow_model_count() -> int:
    """Count model files outside the known model directory (suspicious locations first)."""
    count = 0
    # Always scan volatile directories
    for search_dir in ["/tmp", "/dev/shm", "/var/tmp"]:
        if not os.path.isdir(search_dir):
            continue
        try:
            for root, _, files in os.walk(search_dir, followlinks=False):
                count += sum(1 for f in files
                             if os.path.splitext(f)[1].lower() in _SHADOW_MODEL_EXTS)
        except Exception:
            pass
    # Scan home dir, pruning known-safe subtrees
    try:
        for root, dirs, files in os.walk(os.path.expanduser("~"), followlinks=False):
            abs_root = os.path.abspath(root)
            if abs_root in _SHADOW_SKIP_DIRS:
                dirs.clear()
                continue
            dirs[:] = [d for d in dirs if d not in {".cache", ".local", "snap", ".config", ".keras"}]
            count += sum(1 for f in files
                         if os.path.splitext(f)[1].lower() in _SHADOW_MODEL_EXTS)
    except Exception:
        pass
    return count


# Region hashed each cycle: [prev_size - _TAMPER_REGION_BYTES, prev_size - _TAMPER_EOF_SKIP].
# Anchored to the PREVIOUS cycle's EOF (a frozen offset), so a plain append
# -- which only adds bytes past prev_size -- leaves every byte in the region
# untouched and the hash identical. _TAMPER_REGION_BYTES (256 KB ~= 3000
# rows) comfortably spans the last-2000-row window retraining consumes;
# _TAMPER_EOF_SKIP ignores the final few hundred bytes in case a write was
# mid-flight at prev_size.
_TAMPER_REGION_BYTES = 256 * 1024
_TAMPER_EOF_SKIP     = 4 * 1024


def check_training_data_changed() -> int:
    """Return 1 if the training CSV changed in a way a plain append can't
    explain: it shrank (truncation / row deletion), or bytes that were
    already written before last cycle got rewritten (in-place poisoning of
    the recent window). First cycle only seeds state and returns 0."""
    if not os.path.exists(DATA_FILE):
        return 0
    try:
        size = os.path.getsize(DATA_FILE)
    except OSError:
        return 0

    prev_size = _prev["training_data_size"]
    prev_hash = _prev["training_data_region_hash"]

    # Hash the frozen region defined by the PREVIOUS cycle's EOF.
    region_hash = None
    if prev_size is not None and prev_size > _TAMPER_REGION_BYTES + _TAMPER_EOF_SKIP and size >= prev_size:
        start = prev_size - _TAMPER_REGION_BYTES - _TAMPER_EOF_SKIP
        try:
            with open(DATA_FILE, "rb") as f:
                f.seek(start)
                region = f.read(_TAMPER_REGION_BYTES)
            region_hash = hashlib.sha256(region).hexdigest()
        except OSError:
            region_hash = None

    # Roll state forward: next cycle compares against THIS cycle's EOF.
    next_hash = None
    if size > _TAMPER_REGION_BYTES + _TAMPER_EOF_SKIP:
        start = size - _TAMPER_REGION_BYTES - _TAMPER_EOF_SKIP
        try:
            with open(DATA_FILE, "rb") as f:
                f.seek(start)
                next_hash = hashlib.sha256(f.read(_TAMPER_REGION_BYTES)).hexdigest()
        except OSError:
            next_hash = None
    _prev["training_data_size"] = size
    _prev["training_data_region_hash"] = next_hash

    if prev_size is None:
        return 0  # first cycle: seed only
    if size < prev_size:
        print(f"[AI-ALERT] Training CSV shrank ({prev_size} -> {size} bytes) — rows removed/truncated")
        return 1
    if prev_hash is not None and region_hash is not None and region_hash != prev_hash:
        print("[AI-ALERT] Training CSV: already-written bytes were rewritten — not a plain append")
        return 1
    return 0


def check_model_file_age_drift() -> int:
    """Return 1 if any model file's mtime advanced without its content changing (silent touch)."""
    current_mtimes = {}
    current_hashes = {}
    for fname in MODEL_FILES:
        path = os.path.join(MODEL_DIR, fname)
        try:
            current_mtimes[fname] = os.path.getmtime(path)
            current_hashes[fname] = _hash_file(path)
        except Exception:
            pass

    if not _prev["model_age_mtimes"]:
        _prev["model_age_mtimes"] = current_mtimes
        _prev["model_age_hashes"] = current_hashes
        return 0

    drift = 0
    for fname in MODEL_FILES:
        prev_mtime = _prev["model_age_mtimes"].get(fname)
        curr_mtime = current_mtimes.get(fname)
        prev_hash  = _prev["model_age_hashes"].get(fname)
        curr_hash  = current_hashes.get(fname)
        if prev_mtime and curr_mtime and curr_mtime != prev_mtime:
            if curr_hash and curr_hash == prev_hash:
                # Timestamp moved but content identical → silent touch/copy
                drift = 1
                print(f"[AI-ALERT] Model file mtime drifted with no content change: {fname}")

    _prev["model_age_mtimes"] = current_mtimes
    _prev["model_age_hashes"] = current_hashes
    return drift


def check_gpu_spike_no_known_workload() -> int:
    """Return 1 if GPU utilization is above 20% with no recognized training process."""
    out = _run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        timeout=5
    )
    try:
        gpu_util = float(out.strip().splitlines()[0])
    except Exception:
        return 0  # nvidia-smi unavailable or parse failed
    if gpu_util < 20:
        return 0
    # Look for a known legitimate GPU consumer
    known_keywords = ["retrain", "train", "aiops-watchdog", "tensorflow", "torch"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if any(kw in cmdline for kw in known_keywords):
                return 0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return 1


# ════════════════════════════════════════════════════════════════════════════
# AI detection (existing, unchanged)
# ════════════════════════════════════════════════════════════════════════════

AI_PACKAGE_CANDIDATES = [
    "openai", "anthropic", "transformers", "torch", "tensorflow",
    "langchain", "llama_index", "sentence_transformers", "vllm", "ollama", "pyod",
]

def detect_ai_packages():
    installed = []
    for pkg in AI_PACKAGE_CANDIDATES:
        try:
            metadata.version(pkg)
            installed.append(pkg)
        except metadata.PackageNotFoundError:
            pass
        except Exception as e:
            print(f"[WARN] package check failed for {pkg}: {e}")
    return len(installed), installed


# Third-party AI runtimes only. Guardian's own watchdog/retrain scripts used
# to be in this list purely to make the dashboard panel non-zero -- but that
# made ai_processes_running permanently >= 3, which pinned ai_risk_score at
# <=80 forever (same class of baked-in false positive as the priority
# watchdog's phantom ssh check). Removed 2026-09-10.
AI_PROCESS_KEYWORDS = [
    "ollama", "vllm", "llama.cpp", "text-generation-webui", "open-webui",
    "invokeai", "comfyui", "automatic1111", "stable-diffusion",
    "text-generation-inference", "lm-studio", "jan.ai", "gpt4all",
]

# cmdline substrings that mark a process as Guardian's own -- excluded even
# if a keyword matches (e.g. a retrain script importing torch).
_OWN_PROCESS_MARKERS = (
    "aiops-agents/aiops-watchdog", "aiops-agents/aiops-guardian",
    "aiops-agents/retrain_recent", "aiops-agents/retrain_common",
    "aiops-agents/diagnose_anomaly", "aiops-agents/generate_report",
)


def detect_ai_processes():
    matches = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name    = proc.info["name"] or ""
            cmdline = " ".join(proc.info["cmdline"] or [])
            haystack = f"{name} {cmdline}".lower()
            if any(m in haystack for m in _OWN_PROCESS_MARKERS):
                continue
            for kw in AI_PROCESS_KEYWORDS:
                if kw.lower() in haystack:
                    matches.append({"pid": proc.info["pid"], "name": name, "match": kw})
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return len(matches), matches


AI_API_ENV_VARS = [
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
]

def detect_ai_api_keys():
    found = [v for v in AI_API_ENV_VARS if os.environ.get(v)]
    return len(found), found


# Each risk factor: per-unit points and a cap, so "2 shadow models" and
# "40 shadow models" no longer score identically, and no single factor can
# dominate. Binary factors (all-or-nothing conditions) use per_unit == cap.
# `tools` and `processes` are NOT in here on purpose:
#   - installed AI packages are context, not runtime risk -> no penalty
#   - third-party AI runtimes get a small scaled penalty via the
#     "third_party_ai_processes" factor below (Guardian's own excluded
#     upstream in detect_ai_processes)
_RISK_FACTORS = {
    #  key                        per_unit  cap   binary  detail template
    "api_keys_in_env":            (8,       16,   False,  "{n} AI API key(s) in this process's environment"),
    "api_keys_exposed_on_disk":   (15,      30,   False,  "{n} place(s) an AI API key is exposed on disk"),
    "outbound_llm_connections":   (10,      20,   False,  "{n} active outbound LLM API connection(s)"),
    "watchdog_ports_external":    (20,      20,   True,   "Watchdog ports reachable from a non-loopback IP"),
    "shadow_models":              (10,      25,   False,  "{n} model file(s) outside the known model dir"),
    "training_data_tampered":     (25,      25,   True,   "Training CSV changed in a way a plain append can't explain"),
    "model_age_drift":            (15,      15,   True,   "Model file timestamp moved with no content change"),
    "gpu_spike_no_workload":      (12,      12,   True,   "GPU active with no recognized workload"),
    "third_party_ai_processes":   (4,       12,   False,  "{n} third-party AI runtime process(es) running"),
}


def calculate_ai_risk_score(tools, processes, api_keys,
                             watchdog_external=0, exposed_keys=0, llm_conns=0,
                             shadow_models=0, training_changed=0,
                             model_age_drift=0, gpu_spike=0):
    """Return (score, factors).

    score  -- 0..100, where 100 means no risk factor is active (a clean host
              can now actually reach 100; it used to be pinned <=80).
    factors -- list of {"key", "detail", "points"} for every *active* factor,
              most points first. `tools` is accepted for signature stability
              and logged by the caller, but contributes nothing to the score.
    """
    counts = {
        "api_keys_in_env":          api_keys,
        "api_keys_exposed_on_disk": exposed_keys,
        "outbound_llm_connections": llm_conns,
        "watchdog_ports_external":  1 if watchdog_external else 0,
        "shadow_models":            shadow_models,
        "training_data_tampered":   1 if training_changed else 0,
        "model_age_drift":          1 if model_age_drift else 0,
        "gpu_spike_no_workload":    1 if gpu_spike else 0,
        "third_party_ai_processes": processes,
    }

    factors = []
    score = 100
    for key, n in counts.items():
        if n <= 0:
            continue
        per_unit, cap, _binary, template = _RISK_FACTORS[key]
        points = min(per_unit * n, cap)
        score -= points
        detail = template.format(n=n)
        if per_unit * n > cap:
            detail += " (capped)"
        factors.append({"key": key, "detail": detail, "points": points})

    factors.sort(key=lambda f: f["points"], reverse=True)
    return max(score, 0), factors
