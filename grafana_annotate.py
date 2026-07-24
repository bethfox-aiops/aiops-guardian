#!/usr/bin/env python3
"""
grafana_annotate.py

Posts a Grafana annotation -- a labeled marker on the existing dashboard
timelines -- when something worth flagging happens (e.g. a Behavioral
Attestation policy verification failure). Deliberately fails soft: if
Grafana is unreachable or the token isn't configured, this prints a
warning and moves on rather than breaking whatever called it. Annotating
a dashboard is never worth crashing a retrain run over.

Auth token lives in .grafana_token (gitignored, not committed) as a
Grafana service-account token (starts with "glsa_"), not the admin login
password -- revocable independently of any human's actual credentials.
"""

import json
import os
import urllib.request
import urllib.error

GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://127.0.0.1:3000")
_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".grafana_token")


def _load_token():
    try:
        with open(_TOKEN_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def post_annotation(text, tags=None):
    """
    Post an annotation to Grafana at the current time. Returns True on
    success, False on any failure (missing token, network error, etc.) --
    never raises, since this is best-effort observability, not a critical
    path.
    """
    token = _load_token()
    if not token:
        print(f"[WARN] No .grafana_token found -- skipping annotation: {text}")
        return False

    payload = {
        "text": text,
        "tags": tags or ["guardian"],
    }
    req = urllib.request.Request(
        f"{GRAFANA_URL}/api/annotations",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status in (200, 201):
                return True
            print(f"[WARN] Grafana annotation returned status {resp.status}")
            return False
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"[WARN] Failed to post Grafana annotation: {e}")
        return False


if __name__ == "__main__":
    # Quick manual test: python3 grafana_annotate.py
    ok = post_annotation("Guardian annotation test", tags=["guardian", "test"])
    print("Success" if ok else "Failed")
