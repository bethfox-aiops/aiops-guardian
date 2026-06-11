#!/usr/bin/env python3

import os
import shutil
import subprocess
import time
from prometheus_client import Gauge, start_http_server
import psutil
from importlib import metadata

PORT = int(os.environ.get("GUARDIAN_HEALTH_PORT", "8014"))

cpu_ok = Gauge("aiops_health_cpu_ok", "CPU health: 1=healthy, 0=unhealthy")
mem_ok = Gauge("aiops_health_mem_ok", "Memory health: 1=healthy, 0=unhealthy")
disk_ok = Gauge("aiops_health_disk_ok", "Disk health: 1=healthy, 0=unhealthy")
inode_ok = Gauge("aiops_health_inode_ok", "Inode health: 1=healthy, 0=unhealthy")
service_ok = Gauge("aiops_health_service_ok", "Service health aggregate: 1=healthy, 0=unhealthy")
health_score = Gauge("aiops_health_score", "Overall system health score (0-100)")
guardian_status = Gauge("aiops_guardian_status", "Guardian overall status: 0=healthy, 1=needs attention, 2=critical")
security_issue_code = Gauge("aiops_security_issue_code", "Security issue: 0=none, 1=firewall disabled, 2=updates pending, 3=ssh risk, 4=multiple issues, 5=too many open ports")
security_recommendation = Gauge(
    "aiops_security_recommendation",
    "Recommended action: 0=none, 1=enable firewall, 2=apply updates, 3=secure ssh, 4=reduce open ports")
security_updates_pending = Gauge("aiops_security_updates_pending", "Count of pending security updates")
security_ufw_enabled = Gauge("aiops_security_ufw_enabled", "UFW enabled: 1=yes, 0=no")
security_root_ssh_enabled = Gauge("aiops_security_root_ssh_enabled", "PermitRootLogin enabled: 1=yes, 0=no")
security_failed_logins_recent = Gauge("aiops_security_failed_logins_recent", "Recent failed login count")
security_score = Gauge("aiops_security_score", "Overall security score (0-100)")
AI_TOOLS_DETECTED = Gauge(
    "ai_tools_detected",
    "Number of AI-related Python packages detected"
)
AI_PROCESSES_RUNNING = Gauge(
    "ai_processes_running",
    "Number of AI-related processes currently running"
)
AI_API_KEYS_PRESENT = Gauge(
    "ai_api_keys_present",
    "Number of AI-related API key environment variables detected"
 )
AI_RISK_SCORE = Gauge(
    "ai_risk_score",
    "Overall AI risk score (0-100)"
)
open_ports_count = Gauge(
    "aiops_security_open_ports_count",
    "Number of listening network ports"
)


def check_service_active(service_name: str) -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", service_name],
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_inode_usage_percent(path: str = "/") -> float:
    stats = os.statvfs(path)
    total_inodes = stats.f_files
    free_inodes = stats.f_ffree
    if total_inodes <= 0:
        return 0.0
    used_inodes = total_inodes - free_inodes
    return (used_inodes / total_inodes) * 100.0


def get_pending_security_updates() -> int:
    try:
        result = subprocess.run(
            ["bash", "-lc", "apt list --upgradable 2>/dev/null | grep -i security | wc -l"],
            capture_output=True,
            text=True,
            check=False,
        )
        return int(result.stdout.strip() or "0")
    except Exception:
        return 0


def get_ufw_enabled() -> int:
    try:
        result = subprocess.run(
            ["sudo", "ufw", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        return 1 if "Status: active" in result.stdout else 0
    except Exception:
        return 0


def get_root_ssh_enabled() -> int:
    config_path = "/etc/ssh/sshd_config"
    try:
        if not os.path.exists(config_path):
            return 0
        with open(config_path, "r", encoding="utf-8") as f:
            text = f.read()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("permitrootlogin"):
                value = line.split(maxsplit=1)[1].strip().lower()
                return 1 if value in {"yes", "prohibit-password", "without-password"} else 0
        return 0
    except Exception:
        return 0

def get_recent_failed_logins() -> int:
    try:
        result = subprocess.run(
            ["bash", "-lc", "journalctl --since '24 hours ago' | grep -i 'failed password' | wc -l"],
            capture_output=True,
            text=True,
            check=False,
        )
        return int(result.stdout.strip() or "0")
    except Exception:
        return 0


def get_open_ports_count() -> int:
    try:
        conns = psutil.net_connections(kind="inet")
        return sum(1 for c in conns if c.status == psutil.CONN_LISTEN)
    except Exception:
        return 0


def compute_health():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = shutil.disk_usage("/").used / shutil.disk_usage("/").total * 100.0
    inode = get_inode_usage_percent("/")

    cpu_state = 1 if cpu < 85 else 0
    mem_state = 1 if mem < 85 else 0
    disk_state = 1 if disk < 90 else 0
    inode_state = 1 if inode < 90 else 0

    services = ["prometheus", "grafana-server", "loki"]
    service_states = [check_service_active(s) for s in services]
    service_state = 1 if all(service_states) else 0

    score = 100
    if cpu_state == 0:
        score -= 20
    if mem_state == 0:
        score -= 20
    if disk_state == 0:
        score -= 20
    if inode_state == 0:
        score -= 20
    if service_state == 0:
        score -= 20

    cpu_ok.set(cpu_state)
    mem_ok.set(mem_state)
    disk_ok.set(disk_state)
    inode_ok.set(inode_state)
    service_ok.set(service_state)
    health_score.set(max(score, 0))


def compute_security():
    updates = get_pending_security_updates()
    ufw = get_ufw_enabled()
    root_ssh = get_root_ssh_enabled()
    failed_logins = get_recent_failed_logins()
    open_ports = get_open_ports_count()

    score = 100
    if updates > 0:
        score -= min(updates * 2, 30)
    if ufw == 0:
        score -= 30
    if root_ssh == 1:
        score -= 30
    if failed_logins > 20:
        score -= 20
    if open_ports > 50:
        
        score -= 20
    elif open_ports > 25:
        score -= 10    

    security_updates_pending.set(updates)
    security_ufw_enabled.set(ufw)
    security_root_ssh_enabled.set(root_ssh)
    security_failed_logins_recent.set(failed_logins)
    open_ports_count.set(open_ports)
    security_score.set(max(score, 0))

    issue_count = 0
    if ufw == 0:
        issue_count += 1
    if updates > 0:
        issue_count += 1
    if root_ssh == 1:
        issue_count += 1
    if issue_count > 1:
        security_issue_code.set(4)
    elif ufw == 0:
        security_issue_code.set(1)
    elif updates > 0:
        security_issue_code.set(2)
    elif root_ssh == 1:
        security_issue_code.set(3)
    elif open_ports > 50:
        security_issue_code.set(5)   
    else:
        security_issue_code.set(0)

    if ufw == 0:
        security_recommendation.set(1)
    elif updates > 0:
        security_recommendation.set(2)
    elif root_ssh == 1:
        security_recommendation.set(3)
    elif open_ports > 50:
        security_recommendation.set(4)
    else:
        security_recommendation.set(0)

def compute_guardian_status():
    current_health = health_score._value.get()
    current_security = security_score._value.get()

    if current_health < 80:
        status = 2
    elif current_security < 80:
        status = 1
    else:
        status = 0

    guardian_status.set(status)

AI_PACKAGE_CANDIDATES = [
    "openai",
    "anthropic",
    "transformers",
    "torch",
    "tensorflow",
    "langchain",
    "llama_index",
    "sentence_transformers",
    "vllm",
    "ollama",
    "pyod",
]

def detect_ai_packages():
    installed_packages = []

    for pkg in AI_PACKAGE_CANDIDATES:
        try:
            metadata.version(pkg)
            installed_packages.append(pkg)
        except metadata.PackageNotFoundError:
            pass
        except Exception as e:
            print(f"[WARN] package check failed for {pkg}: {e}")

    return len(installed_packages), installed_packages

AI_PROCESS_KEYWORDS = [
    "ollama",
    "vllm",
    "llama.cpp",
    "text-generation-webui",
    "open-webui",
    "invokeai",
    "comfyui",
    "automatic1111",
    "stable-diffusion",
    "transformers",
    "langchain",
    "aiops-watchdog-knn.py",
    "aiops-watchdog-iforest.py",
    "aiops-watchdog-autoencoder.py",
]

def detect_ai_processes():
    matches = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.info["pid"]
            name = proc.info["name"] or ""
            cmdline = " ".join(proc.info["cmdline"] or [])
            haystack = f"{name} {cmdline}".lower()

            for keyword in AI_PROCESS_KEYWORDS:
                if keyword.lower() in haystack:
                    matches.append({
                        "pid": pid,
                        "name": name,
                        "match": keyword,
                        "cmdline": cmdline,
                    })
                    break

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except Exception as e:
            print(f"[WARN] AI process check failed: {e}")

    return len(matches), matches

AI_API_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
]

def detect_ai_api_keys():
    found_vars = []

    for var_name in AI_API_ENV_VARS:
        if os.environ.get(var_name):
            found_vars.append(var_name)

    return len(found_vars), found_vars


def calculate_ai_risk_score(tools, processes, api_keys):
    score = 100
    reasons = []

    if tools > 0:
        score -= 20
        reasons.append("AI tools installed")

    if processes > 0:
        score -= 30
        reasons.append("AI processes running")

    if api_keys > 0:
        score -= 40
        reasons.append("AI API keys present")

    # Floor at 0
    score = max(score, 0)

    return score, reasons

def main():
    start_http_server(PORT)
    print(f"Guardian health exporter running on port {PORT}")
    while True:
        compute_health()
        compute_security()
        compute_guardian_status()

        count, packages = detect_ai_packages()
        AI_TOOLS_DETECTED.set(count)
        print(f"[AI CHECK] installed_count={count}, installed_packages={packages}")

        proc_count, proc_matches = detect_ai_processes()
        AI_PROCESSES_RUNNING.set(proc_count)
        print(f"[AI PROC CHECK] running_count={proc_count}, matches={proc_matches}")

        api_key_count, api_key_vars = detect_ai_api_keys()
        AI_API_KEYS_PRESENT.set(api_key_count)
        print(f"[AI API CHECK] key_var_count={api_key_count}, key_vars={api_key_vars}")

        risk_score, risk_reasons = calculate_ai_risk_score(count, proc_count, api_key_count)
        AI_RISK_SCORE.set(risk_score)

        print(f"[AI RISK] score={risk_score}, reasons={risk_reasons}")  

        time.sleep(30)



if __name__ == "__main__":
    main()
