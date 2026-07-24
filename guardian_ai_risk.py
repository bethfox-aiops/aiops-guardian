#!/usr/bin/env python3
"""
guardian_ai_risk.py

Guardian's AI risk engine: detects installed AI tooling/processes/API keys
plus the newer risk signals (exposed keys, outbound LLM connections, shadow
models, training-data tampering, model file drift, GPU spikes, externally
reachable watchdog ports), combined into the ai_* gauges by
calculate_ai_risk_score().
"""

import hashlib
import os
import socket
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
AI_TOOLS_DETECTED    = Gauge("ai_tools_detected",     "Number of AI-related Python packages detected")
AI_PROCESSES_RUNNING = Gauge("ai_processes_running",  "Number of AI-related processes currently running")
AI_API_KEYS_PRESENT  = Gauge("ai_api_keys_present",   "Number of AI-related API key environment variables detected")
AI_RISK_SCORE        = Gauge("ai_risk_score",         "Overall AI risk score (0-100)")

# ─── AI Risk gauges (new) ────────────────────────────────────────────────────
AI_WATCHDOG_EXTERNAL    = Gauge("ai_watchdog_port_external_access", "1 if watchdog ports reachable from non-loopback IP")
AI_EXPOSED_KEYS         = Gauge("ai_exposed_api_keys",              "API keys found in env vars or config files")
AI_LLM_CONNECTIONS      = Gauge("ai_outbound_llm_connections",      "Active connections to known LLM API endpoints")
AI_SHADOW_MODELS        = Gauge("ai_shadow_model_count",            "Model files found outside the known model directory")
AI_TRAINING_CHANGED     = Gauge("ai_training_data_hash_changed",    "1 if training data was modified (not just appended)")
AI_MODEL_AGE_DRIFT      = Gauge("ai_model_file_age_drift",          "1 if a model file mtime changed without its content changing")
AI_GPU_SPIKE            = Gauge("ai_gpu_spike_no_known_workload",   "1 if GPU above 20% with no known training job running")


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

def get_exposed_api_keys() -> int:
    """Count API keys found in environment variables and common config files."""
    found = set()
    # Env vars in this process
    for v in _API_KEY_FILE_PATTERNS:
        if os.environ.get(v):
            found.add(f"env:{v}")
    # Common files
    check_files = [
        os.path.expanduser("~/.env"),
        os.path.expanduser("~/.bashrc"),
        os.path.expanduser("~/.bash_profile"),
        os.path.expanduser("~/.profile"),
        "/home/beth/aiops-agents/.env",
    ]
    for path in check_files:
        if not os.path.exists(path):
            continue
        try:
            with open(path, errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for pattern in _API_KEY_FILE_PATTERNS:
                        if pattern in stripped and "=" in stripped:
                            val = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                found.add(f"file:{path}:{pattern}")
        except Exception:
            pass
    return len(found)


_LLM_KEYWORDS = {"openai", "anthropic", "huggingface", "cohere", "mistral",
                  "together", "replicate", "generativelanguage"}

def get_outbound_llm_connections() -> int:
    """Count established HTTPS connections whose reverse DNS matches an LLM provider."""
    count = 0
    try:
        conns = psutil.net_connections(kind="inet")
        remote_ips = {
            c.raddr.ip for c in conns
            if c.status == "ESTABLISHED" and c.raddr and c.raddr.port in (80, 443)
        }
        for ip in list(remote_ips)[:15]:  # cap to avoid slow DNS hangs
            try:
                result = _run(
                    ["dig", "+short", "+time=1", "+tries=1", "-x", ip],
                    timeout=3
                )
                if any(kw in result.lower() for kw in _LLM_KEYWORDS):
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
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


def check_training_data_changed() -> int:
    """Return 1 if the head of the training CSV changed (tampering, not normal appends)."""
    if not os.path.exists(DATA_FILE):
        return 0
    try:
        with open(DATA_FILE, "rb") as f:
            head_hash = hashlib.md5(f.read(102_400)).hexdigest()  # first 100 KB
    except Exception:
        return 0
    if _prev["training_data_head_hash"] is None:
        _prev["training_data_head_hash"] = head_hash
        return 0
    changed = 1 if head_hash != _prev["training_data_head_hash"] else 0
    _prev["training_data_head_hash"] = head_hash
    return changed


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


AI_PROCESS_KEYWORDS = [
    "ollama", "vllm", "llama.cpp", "text-generation-webui", "open-webui",
    "invokeai", "comfyui", "automatic1111", "stable-diffusion", "transformers",
    "langchain", "aiops-watchdog-knn.py", "aiops-watchdog-iforest.py",
    "aiops-watchdog-autoencoder.py",
]

def detect_ai_processes():
    matches = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name    = proc.info["name"] or ""
            cmdline = " ".join(proc.info["cmdline"] or [])
            haystack = f"{name} {cmdline}".lower()
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


def calculate_ai_risk_score(tools, processes, api_keys,
                             watchdog_external=0, exposed_keys=0, llm_conns=0,
                             shadow_models=0, training_changed=0,
                             model_age_drift=0, gpu_spike=0):
    score = 100
    reasons = []
    if tools > 0:
        score -= 10
        reasons.append("AI tools installed")
    if processes > 0:
        score -= 10
        reasons.append("AI processes running")
    if api_keys > 0:
        score -= 20
        reasons.append("AI API keys in environment")
    if exposed_keys > 0:
        score -= 25
        reasons.append(f"{exposed_keys} API key(s) exposed in files/env")
    if llm_conns > 0:
        score -= 20
        reasons.append(f"{llm_conns} outbound LLM API connection(s)")
    if watchdog_external:
        score -= 15
        reasons.append("Watchdog ports externally reachable")
    if shadow_models > 0:
        score -= 20
        reasons.append(f"{shadow_models} shadow model file(s) found")
    if training_changed:
        score -= 25
        reasons.append("Training data head modified")
    if model_age_drift:
        score -= 20
        reasons.append("Model file timestamp drifted without content change")
    if gpu_spike:
        score -= 20
        reasons.append("GPU spike with no known workload")
    return max(score, 0), reasons
