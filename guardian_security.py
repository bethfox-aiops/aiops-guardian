#!/usr/bin/env python3
"""
guardian_security.py

Guardian's security engine: identity & access, persistence, process,
network, file-integrity, ransomware/cryptomining, system-integrity, and
AI/model-specific checks, combined into the aiops_security_* gauges and
aiops_security_score by compute_security().
"""

import os

import psutil
from prometheus_client import Gauge

from guardian_common import (
    MODEL_DIR,
    MODEL_FILES,
    _get_ufw_status_cached,
    _hash_file,
    _hash_string,
    _prev,
    _run,
)

# ─── Existing security gauges ────────────────────────────────────────────────
security_issue_code       = Gauge("aiops_security_issue_code",        "Security issue: 0=none, 1=firewall disabled, 2=updates pending, 3=ssh risk, 4=multiple issues, 5=too many open ports")
security_recommendation   = Gauge("aiops_security_recommendation",    "Recommended action: 0=none, 1=enable firewall, 2=apply updates, 3=secure ssh, 4=reduce open ports")
security_updates_pending  = Gauge("aiops_security_updates_pending",   "Count of pending security updates")
security_ufw_enabled      = Gauge("aiops_security_ufw_enabled",       "UFW enabled: 1=yes, 0=no")
security_root_ssh_enabled = Gauge("aiops_security_root_ssh_enabled",  "PermitRootLogin enabled: 1=yes, 0=no")
security_failed_logins_recent = Gauge("aiops_security_failed_logins_recent", "Recent failed login count")
security_score            = Gauge("aiops_security_score",             "Overall security score (0-100)")
open_ports_count          = Gauge("aiops_security_open_ports_count",  "Number of listening network ports")

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
            count = sum(1 for line in f if not line.startswith("#") and line.strip()
                        and line.split(":")[-1].strip() not in nologin)
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
    return sum(1 for line in out.splitlines() if "COMMAND" in line)


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
    return len([line for line in out.splitlines() if line.strip()])


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
    return len([line for line in out.splitlines() if line.strip()])


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

def _score_security_base(updates: int, ufw: int, root_ssh: int, failed_logins: int, open_ports: int):
    """
    Pure scoring logic for the five base security checks: computes the
    score deduction, issue code, and recommendation code. Split out from
    compute_security() so it's testable without mocking the ~25 other
    checks that function also runs.

    Returns (base_deduction, issue_code, recommendation).
    """
    base_deduction = 0
    if updates > 0:
        base_deduction += min(updates * 2, 30)
    if ufw == 0:
        base_deduction += 30
    if root_ssh == 1:
        base_deduction += 30
    if failed_logins > 20:
        base_deduction += 20
    if open_ports > 50:
        base_deduction += 20
    elif open_ports > 25:
        base_deduction += 10

    issue_count = sum([ufw == 0, updates > 0, root_ssh == 1])
    if issue_count > 1:
        issue_code = 4
    elif ufw == 0:
        issue_code = 1
    elif updates > 0:
        issue_code = 2
    elif root_ssh == 1:
        issue_code = 3
    elif open_ports > 50:
        issue_code = 5
    else:
        issue_code = 0

    if ufw == 0:
        recommendation = 1
    elif updates > 0:
        recommendation = 2
    elif root_ssh == 1:
        recommendation = 3
    elif open_ports > 50:
        recommendation = 4
    else:
        recommendation = 0

    return base_deduction, issue_code, recommendation


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

    base_deduction, issue_code, recommendation = _score_security_base(
        updates, ufw, root_ssh, failed_logins, open_ports
    )
    security_issue_code.set(issue_code)
    security_recommendation.set(recommendation)

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
