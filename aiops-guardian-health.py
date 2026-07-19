#!/usr/bin/env python3

import hashlib
import os
import shutil
import socket
import subprocess
import time
from importlib import metadata

import psutil
from prometheus_client import Gauge, start_http_server

PORT = int(os.environ.get("GUARDIAN_HEALTH_PORT", "8014"))

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

# ─── Health gauges ───────────────────────────────────────────────────────────
cpu_ok        = Gauge("aiops_health_cpu_ok",      "CPU health: 1=healthy, 0=unhealthy")
mem_ok        = Gauge("aiops_health_mem_ok",      "Memory health: 1=healthy, 0=unhealthy")
disk_ok       = Gauge("aiops_health_disk_ok",     "Disk health: 1=healthy, 0=unhealthy")
inode_ok      = Gauge("aiops_health_inode_ok",    "Inode health: 1=healthy, 0=unhealthy")
service_ok    = Gauge("aiops_health_service_ok",  "Service health aggregate: 1=healthy, 0=unhealthy")
health_score  = Gauge("aiops_health_score",       "Overall system health score (0-100)")
guardian_status = Gauge("aiops_guardian_status",  "Guardian overall status: 0=healthy, 1=needs attention, 2=critical")

# ─── Existing security gauges ────────────────────────────────────────────────
security_issue_code       = Gauge("aiops_security_issue_code",        "Security issue: 0=none, 1=firewall disabled, 2=updates pending, 3=ssh risk, 4=multiple issues, 5=too many open ports")
security_recommendation   = Gauge("aiops_security_recommendation",    "Recommended action: 0=none, 1=enable firewall, 2=apply updates, 3=secure ssh, 4=reduce open ports")
security_updates_pending  = Gauge("aiops_security_updates_pending",   "Count of pending security updates")
security_ufw_enabled      = Gauge("aiops_security_ufw_enabled",       "UFW enabled: 1=yes, 0=no")
security_root_ssh_enabled = Gauge("aiops_security_root_ssh_enabled",  "PermitRootLogin enabled: 1=yes, 0=no")
security_failed_logins_recent = Gauge("aiops_security_failed_logins_recent", "Recent failed login count")
security_score            = Gauge("aiops_security_score",             "Overall security score (0-100)")
open_ports_count          = Gauge("aiops_security_open_ports_count",  "Number of listening network ports")

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

# ─── Identity & Access ───────────────────────────────────────────────────────
g_ssh_keys_changed = Gauge("aiops_security_ssh_keys_changed",       "1 if authorized_keys changed since last check")
g_new_users        = Gauge("aiops_security_new_user_count",         "User accounts added since baseline")
g_sudo_cmds_24h    = Gauge("aiops_security_sudo_commands_24h",      "Sudo commands executed in last 24h")
g_unique_ssh_ips   = Gauge("aiops_security_unique_ssh_source_ips",  "Unique SSH source IPs in last 24h")

# ─── Persistence Detection ───────────────────────────────────────────────────
g_cron_changed     = Gauge("aiops_security_cron_changed",           "1 if cron jobs changed since last check")
g_systemd_changed  = Gauge("aiops_security_systemd_units_changed",  "1 if systemd unit files changed since last check")
g_suid_count       = Gauge("aiops_security_suid_binary_count",      "Number of SUID/SGID binaries")

# ─── Process & Execution Anomalies ──────────────────────────────────────────
g_procs_from_tmp = Gauge("aiops_security_procs_from_tmp",       "Processes running from /tmp or /dev/shm")
g_root_procs     = Gauge("aiops_security_root_process_count",   "Processes running as root")
g_zombie_procs   = Gauge("aiops_security_zombie_process_count", "Zombie processes")

# ─── Network Anomalies ───────────────────────────────────────────────────────
g_outbound_conns    = Gauge("aiops_security_outbound_connections",   "Established outbound connections")
g_promisc_ifaces    = Gauge("aiops_security_promiscuous_interfaces", "Network interfaces in promiscuous mode")
g_high_port_listeners = Gauge("aiops_security_high_port_listeners", "Listeners on ports above 49152")
g_dns_conns         = Gauge("aiops_security_dns_connections",        "Active connections to port 53")

# ─── File Integrity ──────────────────────────────────────────────────────────
g_passwd_changed  = Gauge("aiops_security_passwd_changed",         "1 if /etc/passwd modified since last check")
g_sudoers_changed = Gauge("aiops_security_sudoers_changed",        "1 if /etc/sudoers modified since last check")
g_shadow_changed  = Gauge("aiops_security_shadow_changed",         "1 if /etc/shadow modified since last check")
g_world_writable  = Gauge("aiops_security_world_writable_files",   "World-writable files in sensitive system paths")
g_tmp_file_count  = Gauge("aiops_security_tmp_file_count",         "Number of files in /tmp")
g_tmp_disk_mb     = Gauge("aiops_security_tmp_disk_mb",            "/tmp disk usage in MB")

# ─── Ransomware / Cryptomining ───────────────────────────────────────────────
g_cpu_spike = Gauge("aiops_security_cpu_spike_no_workload", "1 if CPU above 80% with no known workload")

# ─── System Integrity ────────────────────────────────────────────────────────
g_kernel_module_count   = Gauge("aiops_security_kernel_module_count",   "Number of loaded kernel modules")
g_kernel_module_changed = Gauge("aiops_security_kernel_module_changed", "1 if kernel module count changed since last check")
g_ntp_synced            = Gauge("aiops_security_ntp_synced",            "1 if NTP synchronized")
g_log_gap               = Gauge("aiops_security_log_gap_detected",      "1 if gap detected in system logs")
g_pkg_failures          = Gauge("aiops_security_package_integrity_failures", "Packages failing dpkg integrity check")

# ─── AI / Model Specific ────────────────────────────────────────────────────
g_model_changed    = Gauge("aiops_security_model_files_changed",       "1 if AI model files changed outside of a retrain cycle")
g_grafana_failures = Gauge("aiops_security_grafana_auth_failures_24h", "Grafana authentication failures in last 24h")
g_watchdog_conns   = Gauge("aiops_security_watchdog_port_connections",  "Active connections to watchdog ports")

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
# Existing health checks
# ════════════════════════════════════════════════════════════════════════════

def check_service_active(service_name: str) -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", service_name], check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def get_inode_usage_percent(path: str = "/") -> float:
    stats = os.statvfs(path)
    if stats.f_files <= 0:
        return 0.0
    return ((stats.f_files - stats.f_ffree) / stats.f_files) * 100.0


def compute_health():
    cpu  = psutil.cpu_percent(interval=1)
    mem  = psutil.virtual_memory().percent
    disk = shutil.disk_usage("/").used / shutil.disk_usage("/").total * 100.0
    inode = get_inode_usage_percent("/")

    cpu_state   = 1 if cpu   < 85 else 0
    mem_state   = 1 if mem   < 85 else 0
    disk_state  = 1 if disk  < 90 else 0
    inode_state = 1 if inode < 90 else 0

    services = ["prometheus", "grafana-server", "loki"]
    service_state = 1 if all(check_service_active(s) for s in services) else 0

    score = 100
    if cpu_state   == 0: score -= 20
    if mem_state   == 0: score -= 20
    if disk_state  == 0: score -= 20
    if inode_state == 0: score -= 20
    if service_state == 0: score -= 20

    cpu_ok.set(cpu_state)
    mem_ok.set(mem_state)
    disk_ok.set(disk_state)
    inode_ok.set(inode_state)
    service_ok.set(service_state)
    health_score.set(max(score, 0))


# ════════════════════════════════════════════════════════════════════════════
# Identity & Access
# ════════════════════════════════════════════════════════════════════════════

def check_ssh_keys_changed() -> int:
    path = os.path.expanduser("~/.ssh/authorized_keys")
    current = _hash_file(path) if os.path.exists(path) else "missing"
    if _prev["ssh_keys_hash"] is None:
        _prev["ssh_keys_hash"] = current
        return 0
    changed = 1 if current != _prev["ssh_keys_hash"] else 0
    _prev["ssh_keys_hash"] = current
    return changed


def check_new_users() -> int:
    nologin = {"/sbin/nologin", "/bin/false", "/usr/sbin/nologin"}
    try:
        with open("/etc/passwd") as f:
            count = sum(1 for l in f if not l.startswith("#") and l.strip()
                        and l.split(":")[-1].strip() not in nologin)
    except Exception:
        count = 0
    if _prev["user_count"] is None:
        _prev["user_count"] = count
        return 0
    diff = max(0, count - _prev["user_count"])
    _prev["user_count"] = count
    return diff


def get_sudo_commands_24h() -> int:
    out = _run(["journalctl", "_COMM=sudo", "--since", "24 hours ago", "--no-pager", "-q"])
    return sum(1 for l in out.splitlines() if "COMMAND" in l)


def get_unique_ssh_source_ips() -> int:
    out = _run(["journalctl", "_COMM=sshd", "--since", "24 hours ago", "--no-pager", "-q"])
    ips = set()
    for line in out.splitlines():
        if "Accepted" in line and "from" in line:
            parts = line.split("from")
            if len(parts) > 1:
                ip = parts[1].strip().split()[0]
                if ip:
                    ips.add(ip)
    return len(ips)


# ════════════════════════════════════════════════════════════════════════════
# Persistence Detection
# ════════════════════════════════════════════════════════════════════════════

def _hash_cron() -> str:
    paths = []
    for p in ["/etc/crontab"]:
        if os.path.isfile(p):
            paths.append(p)
    for d in ["/etc/cron.d", "/etc/cron.hourly", "/etc/cron.daily",
              "/etc/cron.weekly", "/etc/cron.monthly"]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    paths.append(fp)
    spool = "/var/spool/cron/crontabs"
    if os.path.isdir(spool):
        try:
            for f in sorted(os.listdir(spool)):
                paths.append(os.path.join(spool, f))
        except PermissionError:
            pass
    combined = "|".join(f"{p}:{_hash_file(p)}" for p in paths)
    return _hash_string(combined)


def check_cron_changed() -> int:
    current = _hash_cron()
    if _prev["cron_hash"] is None:
        _prev["cron_hash"] = current
        return 0
    changed = 1 if current != _prev["cron_hash"] else 0
    _prev["cron_hash"] = current
    return changed


def _hash_systemd_units() -> str:
    d = "/etc/systemd/system"
    try:
        parts = []
        for f in sorted(os.listdir(d)):
            if f.endswith(".service"):
                try:
                    mtime = os.path.getmtime(os.path.join(d, f))
                    parts.append(f"{f}:{mtime}")
                except Exception:
                    parts.append(f)
        return _hash_string("|".join(parts))
    except Exception:
        return ""


def check_systemd_units_changed() -> int:
    current = _hash_systemd_units()
    if _prev["systemd_hash"] is None:
        _prev["systemd_hash"] = current
        return 0
    changed = 1 if current != _prev["systemd_hash"] else 0
    _prev["systemd_hash"] = current
    return changed


def get_suid_binary_count() -> int:
    out = _run(
        ["find", "/usr", "/bin", "/sbin", "/usr/local",
         "-perm", "/6000", "-type", "f"],
        timeout=30
    )
    return len([l for l in out.splitlines() if l.strip()])


# ════════════════════════════════════════════════════════════════════════════
# Process & Execution Anomalies
# ════════════════════════════════════════════════════════════════════════════

KNOWN_TMP_PROCS = {"tempo-query"}

def get_procs_from_tmp() -> int:
    """Count processes whose executable lives in /tmp or /dev/shm.
    Only checks exe path and argv[0] — not the full cmdline — to avoid
    false positives from programs that pass /tmp paths as arguments."""
    count = 0
    for proc in psutil.process_iter(["exe", "cmdline", "name"]):
        try:
            name = proc.info.get("name") or ""
            exe  = proc.info.get("exe") or ""
            argv0 = (proc.info.get("cmdline") or [""])[0]
            if name in KNOWN_TMP_PROCS:
                continue
            if "/tmp/" in exe or "/dev/shm/" in exe or "/tmp/" in argv0 or "/dev/shm/" in argv0:
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return count


def get_root_process_count() -> int:
    count = 0
    for proc in psutil.process_iter(["username"]):
        try:
            if proc.info.get("username") == "root":
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return count


def get_zombie_count() -> int:
    count = 0
    for proc in psutil.process_iter(["status"]):
        try:
            if proc.info.get("status") == psutil.STATUS_ZOMBIE:
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return count


# ════════════════════════════════════════════════════════════════════════════
# Network Anomalies
# ════════════════════════════════════════════════════════════════════════════

def get_outbound_connection_count() -> int:
    try:
        conns = psutil.net_connections(kind="inet")
        listeners = {c.laddr.port for c in conns if c.status == psutil.CONN_LISTEN}
        return sum(
            1 for c in conns
            if c.status == "ESTABLISHED" and c.raddr and c.laddr.port not in listeners
        )
    except Exception:
        return 0


def get_promiscuous_interfaces() -> int:
    count = 0
    net_dir = "/sys/class/net"
    try:
        for iface in os.listdir(net_dir):
            flags_path = os.path.join(net_dir, iface, "flags")
            try:
                with open(flags_path) as f:
                    if int(f.read().strip(), 16) & 0x100:  # IFF_PROMISC
                        count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count


def get_high_port_listeners() -> int:
    try:
        conns = psutil.net_connections(kind="inet")
        return sum(1 for c in conns if c.status == psutil.CONN_LISTEN and c.laddr.port > 49152)
    except Exception:
        return 0


def get_dns_connections() -> int:
    try:
        conns = psutil.net_connections(kind="inet")
        return sum(1 for c in conns if c.raddr and c.raddr.port == 53)
    except Exception:
        return 0


# ════════════════════════════════════════════════════════════════════════════
# File Integrity
# ════════════════════════════════════════════════════════════════════════════

def check_critical_file_mtimes() -> tuple:
    results = []
    for key, path in [
        ("passwd_mtime",  "/etc/passwd"),
        ("sudoers_mtime", "/etc/sudoers"),
        ("shadow_mtime",  "/etc/shadow"),
    ]:
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = None
        if _prev[key] is None:
            _prev[key] = mtime
            results.append(0)
        else:
            changed = 1 if mtime != _prev[key] else 0
            _prev[key] = mtime
            results.append(changed)
    return tuple(results)


def get_world_writable_count() -> int:
    out = _run(
        ["find", "/etc", "/bin", "/usr/bin", "/sbin", "/usr/sbin",
         "-perm", "-002", "-type", "f"],
        timeout=30
    )
    return len([l for l in out.splitlines() if l.strip()])


def get_tmp_stats() -> tuple:
    try:
        file_count = len(os.listdir("/tmp"))
    except Exception:
        file_count = 0
    out = _run(["du", "-sm", "/tmp"], timeout=10)
    try:
        disk_mb = float(out.split()[0])
    except Exception:
        disk_mb = 0.0
    return file_count, disk_mb


# ════════════════════════════════════════════════════════════════════════════
# Ransomware / Cryptomining
# ════════════════════════════════════════════════════════════════════════════

def check_cpu_spike_no_workload() -> int:
    cpu = psutil.cpu_percent(interval=0.5)
    if cpu < 80:
        return 0
    known_high_cpu = {"python", "python3", "cc1", "gcc", "make", "ninja"}
    try:
        for proc in psutil.process_iter(["name", "cpu_percent"]):
            try:
                if proc.info["name"] in known_high_cpu and (proc.cpu_percent() or 0) > 40:
                    return 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass
    return 1


# ════════════════════════════════════════════════════════════════════════════
# System Integrity
# ════════════════════════════════════════════════════════════════════════════

def get_kernel_module_count() -> int:
    try:
        with open("/proc/modules") as f:
            return len(f.readlines())
    except Exception:
        return 0


def check_kernel_modules_changed() -> int:
    current = get_kernel_module_count()
    if _prev["kernel_module_count"] is None:
        _prev["kernel_module_count"] = current
        return 0
    changed = 1 if current != _prev["kernel_module_count"] else 0
    _prev["kernel_module_count"] = current
    return changed


def get_ntp_synced() -> int:
    out = _run(["timedatectl", "show", "--property=NTPSynchronized"])
    return 1 if "yes" in out.lower() else 0


def check_log_gap() -> int:
    out = _run(["journalctl", "--since", "5 minutes ago", "--no-pager", "-q", "-n", "1"])
    return 0 if out.strip() else 1


def get_package_integrity_failures() -> int:
    out = _run(["dpkg", "--verify"], timeout=60)
    count = 0
    for line in out.splitlines():
        if not line.strip():
            continue
        if "Permission denied" in line or "missing" in line.lower():
            continue
        # Only flag actual checksum failures (??5??????) or unexpected changes
        if line.startswith("??5") or (not line.startswith("missing") and "??????" not in line):
            count += 1
        elif "??5" in line:
            count += 1
    return count


# ════════════════════════════════════════════════════════════════════════════
# AI / Model Specific
# ════════════════════════════════════════════════════════════════════════════

def check_model_files_changed() -> int:
    current = {f: _hash_file(os.path.join(MODEL_DIR, f)) for f in MODEL_FILES}
    if not _prev["model_hashes"]:
        _prev["model_hashes"] = current
        return 0
    changed = any(current.get(k) != v for k, v in _prev["model_hashes"].items())
    _prev["model_hashes"] = current
    return 1 if changed else 0


def get_grafana_auth_failures() -> int:
    out = _run(["journalctl", "-u", "grafana-server",
                "--since", "24 hours ago", "--no-pager", "-q"])
    ignore = {"token needs to be rotated", "token.rotate", "datasourcenodataa", "smtp not configured"}
    count = 0
    for line in out.lower().splitlines():
        if any(skip in line for skip in ignore):
            continue
        if "invalid username or password" in line or "login failed" in line:
            count += 1
    return count


def get_watchdog_port_connections() -> int:
    watchdog_ports = {8011, 8012, 8013, 8014}
    try:
        conns = psutil.net_connections(kind="inet")
        return sum(1 for c in conns
                   if c.laddr.port in watchdog_ports and c.status == "ESTABLISHED")
    except Exception:
        return 0


# ════════════════════════════════════════════════════════════════════════════
# AI Risk Detection
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
# Existing security checks
# ════════════════════════════════════════════════════════════════════════════

def get_pending_security_updates() -> int:
    out = _run(["bash", "-lc", "apt list --upgradable 2>/dev/null | grep -i security | wc -l"])
    try:
        return int(out.strip())
    except Exception:
        return 0


def get_ufw_enabled() -> int:
    return 1 if "Status: active" in _get_ufw_status_cached() else 0


def get_root_ssh_enabled() -> int:
    try:
        with open("/etc/ssh/sshd_config") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("permitrootlogin"):
                    value = line.split(maxsplit=1)[1].strip().lower()
                    return 1 if value in {"yes", "prohibit-password", "without-password"} else 0
    except Exception:
        pass
    return 0


def get_recent_failed_logins() -> int:
    out = _run(["bash", "-lc",
                "journalctl --since '24 hours ago' | grep -i 'failed password' | wc -l"])
    try:
        return int(out.strip())
    except Exception:
        return 0


def get_open_ports_count() -> int:
    try:
        conns = psutil.net_connections(kind="inet")
        return sum(1 for c in conns if c.status == psutil.CONN_LISTEN)
    except Exception:
        return 0


# ════════════════════════════════════════════════════════════════════════════
# Combined security compute
# ════════════════════════════════════════════════════════════════════════════

def compute_security(iteration: int):
    # ── Base checks (existing) ──────────────────────────────────────────────
    updates      = get_pending_security_updates()
    ufw          = get_ufw_enabled()
    root_ssh     = get_root_ssh_enabled()
    failed_logins = get_recent_failed_logins()
    open_ports   = get_open_ports_count()

    security_updates_pending.set(updates)
    security_ufw_enabled.set(ufw)
    security_root_ssh_enabled.set(root_ssh)
    security_failed_logins_recent.set(failed_logins)
    open_ports_count.set(open_ports)

    base_deduction = 0
    if updates > 0:       base_deduction += min(updates * 2, 30)
    if ufw == 0:          base_deduction += 30
    if root_ssh == 1:     base_deduction += 30
    if failed_logins > 20: base_deduction += 20
    if open_ports > 50:   base_deduction += 20
    elif open_ports > 25: base_deduction += 10

    issue_count = sum([ufw == 0, updates > 0, root_ssh == 1])
    if issue_count > 1:    security_issue_code.set(4)
    elif ufw == 0:         security_issue_code.set(1)
    elif updates > 0:      security_issue_code.set(2)
    elif root_ssh == 1:    security_issue_code.set(3)
    elif open_ports > 50:  security_issue_code.set(5)
    else:                  security_issue_code.set(0)

    if ufw == 0:           security_recommendation.set(1)
    elif updates > 0:      security_recommendation.set(2)
    elif root_ssh == 1:    security_recommendation.set(3)
    elif open_ports > 50:  security_recommendation.set(4)
    else:                  security_recommendation.set(0)

    # ── Identity & Access ───────────────────────────────────────────────────
    ssh_changed  = check_ssh_keys_changed()
    new_users    = check_new_users()
    sudo_cmds    = get_sudo_commands_24h()
    unique_ips   = get_unique_ssh_source_ips()
    g_ssh_keys_changed.set(ssh_changed)
    g_new_users.set(new_users)
    g_sudo_cmds_24h.set(sudo_cmds)
    g_unique_ssh_ips.set(unique_ips)

    # ── Persistence Detection ───────────────────────────────────────────────
    cron_changed    = check_cron_changed()
    systemd_changed = check_systemd_units_changed()
    g_cron_changed.set(cron_changed)
    g_systemd_changed.set(systemd_changed)

    if iteration % 20 == 0:
        suid = get_suid_binary_count()
        _prev["suid_count"] = suid
        g_suid_count.set(suid)
    elif _prev["suid_count"] is not None:
        g_suid_count.set(_prev["suid_count"])

    # ── Process Anomalies ───────────────────────────────────────────────────
    procs_tmp = get_procs_from_tmp()
    root_procs = get_root_process_count()
    zombies    = get_zombie_count()
    g_procs_from_tmp.set(procs_tmp)
    g_root_procs.set(root_procs)
    g_zombie_procs.set(zombies)

    # ── Network Anomalies ───────────────────────────────────────────────────
    outbound    = get_outbound_connection_count()
    promisc     = get_promiscuous_interfaces()
    high_ports  = get_high_port_listeners()
    dns_conns   = get_dns_connections()
    g_outbound_conns.set(outbound)
    g_promisc_ifaces.set(promisc)
    g_high_port_listeners.set(high_ports)
    g_dns_conns.set(dns_conns)

    # ── File Integrity ──────────────────────────────────────────────────────
    passwd_c, sudoers_c, shadow_c = check_critical_file_mtimes()
    g_passwd_changed.set(passwd_c)
    g_sudoers_changed.set(sudoers_c)
    g_shadow_changed.set(shadow_c)

    tmp_count, tmp_mb = get_tmp_stats()
    g_tmp_file_count.set(tmp_count)
    g_tmp_disk_mb.set(tmp_mb)

    if iteration % 20 == 0:
        ww = get_world_writable_count()
        _prev["world_writable"] = ww
        g_world_writable.set(ww)
    else:
        g_world_writable.set(_prev["world_writable"])

    # ── Ransomware / Cryptomining ───────────────────────────────────────────
    cpu_spike = check_cpu_spike_no_workload()
    g_cpu_spike.set(cpu_spike)

    # ── System Integrity ────────────────────────────────────────────────────
    km_count   = get_kernel_module_count()
    km_changed = check_kernel_modules_changed()
    ntp        = get_ntp_synced()
    log_gap    = check_log_gap()
    g_kernel_module_count.set(km_count)
    g_kernel_module_changed.set(km_changed)
    g_ntp_synced.set(ntp)
    g_log_gap.set(log_gap)

    if iteration % 20 == 0:
        pkg = get_package_integrity_failures()
        _prev["pkg_failures"] = pkg
        g_pkg_failures.set(pkg)
    else:
        g_pkg_failures.set(_prev["pkg_failures"])

    # ── AI / Model Specific ─────────────────────────────────────────────────
    model_changed   = check_model_files_changed()
    grafana_fail    = get_grafana_auth_failures()
    watchdog_conn   = get_watchdog_port_connections()
    g_model_changed.set(model_changed)
    g_grafana_failures.set(grafana_fail)
    g_watchdog_conns.set(watchdog_conn)

    # ── Extended deductions ─────────────────────────────────────────────────
    ext_deduction = 0

    if ssh_changed:
        ext_deduction += 25
        print("[SEC-ALERT] SSH authorized_keys changed!")
    if new_users > 0:
        ext_deduction += 25
        print(f"[SEC-ALERT] {new_users} new user account(s) detected!")
    if cron_changed:
        ext_deduction += 15
        print("[SEC-ALERT] Cron configuration changed!")
    if systemd_changed:
        ext_deduction += 15
        print("[SEC-ALERT] Systemd unit files changed!")
    if procs_tmp > 0:
        ext_deduction += min(procs_tmp * 20, 40)
        print(f"[SEC-ALERT] {procs_tmp} process(es) running from /tmp or /dev/shm!")
    if promisc > 0:
        ext_deduction += 30
        print(f"[SEC-ALERT] {promisc} interface(s) in promiscuous mode!")
    if passwd_c:
        ext_deduction += 20
        print("[SEC-ALERT] /etc/passwd modified!")
    if sudoers_c:
        ext_deduction += 25
        print("[SEC-ALERT] /etc/sudoers modified!")
    if shadow_c:
        ext_deduction += 20
        print("[SEC-ALERT] /etc/shadow modified!")
    if km_changed:
        ext_deduction += 20
        print(f"[SEC-ALERT] Kernel module count changed to {km_count}!")
    if ntp == 0:
        ext_deduction += 10
        print("[SEC-ALERT] NTP not synchronized!")
    if log_gap:
        ext_deduction += 15
        print("[SEC-ALERT] Log gap detected — possible log tampering!")
    if model_changed:
        ext_deduction += 25
        print("[SEC-ALERT] AI model files changed outside of a retrain cycle!")
    if cpu_spike:
        ext_deduction += 15
        print("[SEC-ALERT] CPU spike with no known workload — possible cryptomining!")
    if _prev["pkg_failures"] > 0:
        ext_deduction += 15
        print(f"[SEC-ALERT] {_prev['pkg_failures']} package integrity failure(s)!")

    score = max(100 - base_deduction - ext_deduction, 0)
    security_score.set(score)

    print(
        f"[SEC] score={score} | ufw={ufw} updates={updates} root_ssh={root_ssh} "
        f"failed_logins={failed_logins} open_ports={open_ports} | "
        f"ssh_changed={ssh_changed} new_users={new_users} cron_changed={cron_changed} "
        f"systemd_changed={systemd_changed} procs_tmp={procs_tmp} promisc={promisc} "
        f"km_changed={km_changed} ntp={ntp} model_changed={model_changed}"
    )


def compute_guardian_status():
    h = health_score._value.get()
    s = security_score._value.get()
    if h < 80:
        guardian_status.set(2)
    elif s < 80:
        guardian_status.set(1)
    else:
        guardian_status.set(0)


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
    if tools            > 0: score -= 10; reasons.append("AI tools installed")
    if processes        > 0: score -= 10; reasons.append("AI processes running")
    if api_keys         > 0: score -= 20; reasons.append("AI API keys in environment")
    if exposed_keys     > 0: score -= 25; reasons.append(f"{exposed_keys} API key(s) exposed in files/env")
    if llm_conns        > 0: score -= 20; reasons.append(f"{llm_conns} outbound LLM API connection(s)")
    if watchdog_external:    score -= 15; reasons.append("Watchdog ports externally reachable")
    if shadow_models    > 0: score -= 20; reasons.append(f"{shadow_models} shadow model file(s) found")
    if training_changed:     score -= 25; reasons.append("Training data head modified")
    if model_age_drift:      score -= 20; reasons.append("Model file timestamp drifted without content change")
    if gpu_spike:            score -= 20; reasons.append("GPU spike with no known workload")
    return max(score, 0), reasons


# ════════════════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════════════════

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

        if watchdog_ext:   print("[AI-ALERT] Watchdog ports reachable from non-loopback IP!")
        if exposed_keys:   print(f"[AI-ALERT] {exposed_keys} API key(s) exposed in files or env!")
        if llm_conns:      print(f"[AI-ALERT] {llm_conns} outbound LLM API connection(s) detected!")
        if shadow:         print(f"[AI-ALERT] {shadow} shadow model file(s) outside known directory!")
        if training_changed: print("[AI-ALERT] Training data head modified — possible poisoning!")
        if model_drift:    print("[AI-ALERT] Model file timestamp changed without content change!")
        if gpu_spike:      print("[AI-ALERT] GPU spike with no recognized training workload!")

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
