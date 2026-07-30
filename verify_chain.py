#!/usr/bin/env python3
"""
verify_chain.py

Walks releases/*.json and confirms the hash chain written by
release_record.py hasn't been broken -- each record's own content still
hashes to its recorded "chain.record_hash", and each record's
"chain.previous_hash" still matches the prior record's hash. This is what
makes the release ledger tamper-evident: editing any past record (or its
position in the chain) after the fact changes a hash somewhere downstream,
and this script is what would actually notice.

Run with: python3 verify_chain.py
Exit code 0 = chain intact, 1 = a break was found.
"""

import glob
import json
import os
import sys

import release_record


def _load_chain():
    records = []
    for path in glob.glob(os.path.join(release_record.RELEASES_DIR, "*.json")):
        with open(path) as f:
            record = json.load(f)
        if "chain" not in record:
            print(f"[SKIP] {os.path.basename(path)} has no chain metadata (pre-dates hash-chaining)")
            continue
        records.append((path, record))
    records.sort(key=lambda pr: pr[1]["chain"]["sequence"])
    return records


def check_chain():
    """Programmatic core, no printing -- re-verifies every record's hash and
    link. Returns (ok, problems, records), where records is the same
    (path, record) list callers like release_report.py need to locate a
    specific release's position in the ledger."""
    records = _load_chain()
    if not records:
        return True, [], records

    expected_previous_hash = release_record.GENESIS_HASH
    problems = []

    for expected_sequence, (path, record) in enumerate(records):
        name = os.path.basename(path)
        chain = record["chain"]

        if chain["sequence"] != expected_sequence:
            problems.append(f"{name}: expected sequence {expected_sequence}, found {chain['sequence']}")

        if chain["previous_hash"] != expected_previous_hash:
            problems.append(
                f"{name}: previous_hash {chain['previous_hash'][:12]}... "
                f"does not match prior record's hash {expected_previous_hash[:12]}..."
            )

        content = {k: v for k, v in record.items() if k != "chain"}
        actual_hash = release_record._hash_record_content(content)
        if actual_hash != chain["record_hash"]:
            problems.append(
                f"{name}: content hash {actual_hash[:12]}... does not match "
                f"recorded hash {chain['record_hash'][:12]}... -- record was likely edited after being written"
            )

        expected_previous_hash = chain["record_hash"]

    return len(problems) == 0, problems, records


def verify_chain():
    ok, problems, records = check_chain()

    if not records:
        print("[INFO] No chained release records found -- nothing to verify.")
        return True

    if not ok:
        print(f"[FAIL] Chain verification found {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return False

    print(f"[PASS] Chain verified intact across {len(records)} record(s), sequence 0-{len(records) - 1}.")
    return True


if __name__ == "__main__":
    sys.exit(0 if verify_chain() else 1)
