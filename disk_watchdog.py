#!/usr/bin/env python3

import shutil
import logging
from datetime import datetime

# === CONFIG ===
THRESHOLD_ALERT = 70   # in percent
THRESHOLD_ACT = 90     # in percent
LOG_FILE = "/tmp/disk_watchdog.log"
CHECK_PATH = "/"       # mount point to monitor

# === SETUP LOGGING ===

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(message)s")

def get_disk_usage(path):
    total, used, free = shutil.disk_usage(path)
    percent_used = (used / total) * 100
    return round(percent_used, 2)

def make_decision(usage):
    if usage < THRESHOLD_ALERT:
        return "WAIT"
    elif usage < THRESHOLD_ACT:
        return "ALERT"
    else:
        return "ACT"
def main():
    usage = get_disk_usage(CHECK_PATH)
    decision = make_decision(usage)
    log_entry = f"Disk usage: {usage}% | Action: {decision}"
    print(log_entry)
    logging.info(log_entry)

if __name__ == "__main__":
    main()


      
