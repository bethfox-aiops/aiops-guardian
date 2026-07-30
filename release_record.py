#!/usr/bin/env python3
"""
release_record.py

Behavioral Attestation Phase 6: AI-managed release traceability.

The capstone: combines build provenance (git -- what changed, who/what
requested it, commit hash) with runtime behavioral attestation (Phases
1-5 -- process/eBPF/GPU evidence, OTel trace, policy verification) into
one structured release record. Answers "what changed in the code, and
what changed in observed runtime behavior as a result" -- with evidence,
not inference.
"""

import glob
import hashlib
import json
import os
import subprocess
import datetime

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASES_DIR = os.path.join(REPO_DIR, "releases")

# Sentinel previous_hash for the first record in the chain -- there is no
# real prior record to point at, so this is never a value a genuine
# sha256 hexdigest could collide with (all zeros, right length).
GENESIS_HASH = "0" * 64


def _canonical(obj):
    """Deterministic JSON encoding -- stable key order, no incidental
    whitespace -- so the same record content always hashes the same way."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash_record_content(record_content):
    return hashlib.sha256(_canonical(record_content).encode()).hexdigest()


def _chain_tip():
    """Scan existing release records for the current end of the chain.
    Returns (next_sequence, previous_hash). The chain order lives entirely
    in each record's own "chain" field -- no separate ledger file to keep
    in sync or go stale."""
    tip_sequence = -1
    tip_hash = GENESIS_HASH
    for path in glob.glob(os.path.join(RELEASES_DIR, "*.json")):
        with open(path) as f:
            existing = json.load(f)
        chain = existing.get("chain")
        if chain and chain["sequence"] > tip_sequence:
            tip_sequence = chain["sequence"]
            tip_hash = chain["record_hash"]
    return tip_sequence + 1, tip_hash


def _git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_DIR)
    return result.stdout.strip()


def get_build_provenance():
    """Snapshot of current git state -- the 'what changed' half of provenance."""
    has_parent = _git("rev-list", "--count", "HEAD") != "1"
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_subject": _git("log", "-1", "--format=%s"),
        "commit_author": _git("log", "-1", "--format=%an <%ae>"),
        "commit_date": _git("log", "-1", "--format=%aI"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "files_changed": _git("diff", "--stat", "HEAD~1", "HEAD").splitlines() if has_parent else [],
    }


def record_release(*, release_id, objective, agent, approval, workflow, trace_id, verification):
    """
    Assemble and write a release record combining build provenance with the
    runtime behavioral fingerprint (trace ID + verification result) from a
    workflow run already produced by Phases 1-5.

    Returns (path, record).
    """
    record = {
        "release_id": release_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "objective": objective,
        "agent": agent,
        "approval": approval,
        "build_provenance": get_build_provenance(),
        "deployment": {
            "workflow": workflow,
            "trace_id": trace_id,
            "trace_url": f"http://localhost:3200/api/traces/{trace_id}" if trace_id else None,
        },
        "runtime_behavioral_attestation": verification,
    }

    os.makedirs(RELEASES_DIR, exist_ok=True)
    sequence, previous_hash = _chain_tip()
    record["chain"] = {
        "sequence": sequence,
        "previous_hash": previous_hash,
        "record_hash": _hash_record_content(record),
    }

    path = os.path.join(RELEASES_DIR, f"{release_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    return path, record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record an AI-managed release.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--agent", default="Claude Code (claude-sonnet-5)")
    parser.add_argument("--approval", default="Interactively reviewed and approved by beth in this Claude Code session.")
    parser.add_argument("--passed", action="store_true")
    parser.add_argument("--violation", action="append", default=[])
    args = parser.parse_args()

    path, record = record_release(
        release_id=args.release_id,
        objective=args.objective,
        agent=args.agent,
        approval=args.approval,
        workflow=args.workflow,
        trace_id=args.trace_id,
        verification={"passed": args.passed, "violations": args.violation},
    )
    print(f"[INFO] Release record written: {path}")
    print(json.dumps(record, indent=2))
