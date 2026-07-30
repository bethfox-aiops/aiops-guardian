"""
test_release_chain.py

Tests for the hash-chaining added to release_record.py / verify_chain.py
(Phase 7 item: "genuinely tamper-evident logging"). The key scenario here
(test_edited_record_breaks_chain) is the actual point of this mechanism:
prove that silently editing a past release record after the fact is
something the chain verifier would actually notice, not just something
it's designed to notice in theory.

Run with: pytest test_release_chain.py -v
"""

import json

import release_record
from verify_chain import verify_chain


def _record(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(release_record, "RELEASES_DIR", str(tmp_path))
    kwargs = dict(
        release_id="test-release",
        objective="test",
        agent="test-agent",
        approval="test-approval",
        workflow="test_workflow",
        trace_id="deadbeef",
        verification={"passed": True, "violations": []},
    )
    kwargs.update(overrides)
    return release_record.record_release(**kwargs)


class TestChainWriting:
    def test_first_record_chains_to_genesis(self, tmp_path, monkeypatch):
        _, record = _record(tmp_path, monkeypatch, release_id="first")
        assert record["chain"]["sequence"] == 0
        assert record["chain"]["previous_hash"] == release_record.GENESIS_HASH

    def test_second_record_chains_to_first(self, tmp_path, monkeypatch):
        _, first = _record(tmp_path, monkeypatch, release_id="first")
        _, second = _record(tmp_path, monkeypatch, release_id="second")
        assert second["chain"]["sequence"] == 1
        assert second["chain"]["previous_hash"] == first["chain"]["record_hash"]


class TestVerifyChain:
    def test_untampered_chain_passes(self, tmp_path, monkeypatch):
        _record(tmp_path, monkeypatch, release_id="first")
        _record(tmp_path, monkeypatch, release_id="second")
        assert verify_chain() is True

    def test_edited_record_breaks_chain(self, tmp_path, monkeypatch):
        # This is the actual scenario tamper-evident logging exists for:
        # someone (or something with root, e.g. an over-privileged AI
        # agent per ROADMAP.md Phase 7) edits a past record after the fact.
        _record(tmp_path, monkeypatch, release_id="first")
        _record(tmp_path, monkeypatch, release_id="second")

        path = tmp_path / "first.json"
        record = json.loads(path.read_text())
        record["runtime_behavioral_attestation"]["passed"] = True  # was already True; tamper the violations list instead
        record["runtime_behavioral_attestation"]["violations"] = []
        record["objective"] = "quietly rewritten after the fact"
        path.write_text(json.dumps(record, indent=2))

        assert verify_chain() is False

    def test_empty_releases_dir_passes_trivially(self, tmp_path, monkeypatch):
        monkeypatch.setattr(release_record, "RELEASES_DIR", str(tmp_path))
        assert verify_chain() is True

    def test_pre_chain_record_is_skipped_not_failed(self, tmp_path, monkeypatch):
        # A record written before this mechanism existed (no "chain" key)
        # should be skipped, not treated as a broken link -- the real
        # releases/ dir has two such records from before this feature.
        monkeypatch.setattr(release_record, "RELEASES_DIR", str(tmp_path))
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps({"release_id": "legacy", "objective": "pre-dates chaining"}))
        assert verify_chain() is True
