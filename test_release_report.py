"""
test_release_report.py

Tests for release_report.py -- the Markdown evidence-report renderer built
on top of release_record.py / verify_chain.py. The key scenario here
(test_report_reflects_broken_chain) confirms the report doesn't just
silently render a clean-looking document when the underlying ledger has
actually been tampered with -- that would defeat the whole point of
having a chain-of-custody section.

Run with: pytest test_release_report.py -v
"""

import json

import release_record
from release_report import render_report


def _write_release(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(release_record, "RELEASES_DIR", str(tmp_path))
    kwargs = dict(
        release_id="test-release",
        objective="Test objective for report rendering.",
        agent="Claude Code (claude-sonnet-5)",
        approval="Interactively reviewed and approved by beth.",
        workflow="retrain_knn",
        trace_id="abc123",
        verification={"passed": True, "violations": []},
    )
    kwargs.update(overrides)
    return release_record.record_release(**kwargs)


class TestRenderReport:
    def test_report_contains_core_evidence(self, tmp_path, monkeypatch):
        _write_release(tmp_path, monkeypatch, release_id="r1")
        report = render_report("r1")
        assert "Test objective for report rendering." in report
        assert "retrain_knn" in report
        assert "abc123" in report

    def test_passing_verification_renders_passed(self, tmp_path, monkeypatch):
        _write_release(tmp_path, monkeypatch, release_id="r1", verification={"passed": True, "violations": []})
        report = render_report("r1")
        assert "**PASSED.**" in report

    def test_failing_verification_lists_violations(self, tmp_path, monkeypatch):
        _write_release(
            tmp_path, monkeypatch, release_id="r1",
            verification={"passed": False, "violations": ["row_count 20 below policy minimum 100"]},
        )
        report = render_report("r1")
        assert "**FAILED.**" in report
        assert "row_count 20 below policy minimum 100" in report

    def test_report_confirms_chain_integrity_when_intact(self, tmp_path, monkeypatch):
        _write_release(tmp_path, monkeypatch, release_id="r1")
        _write_release(tmp_path, monkeypatch, release_id="r2")
        report = render_report("r2")
        assert "Chain integrity: VERIFIED." in report
        assert "entry **2 of 2**" in report

    def test_report_reflects_broken_chain(self, tmp_path, monkeypatch):
        # The whole point of the chain-of-custody section: if the ledger
        # has been tampered with, the report has to say so, not render a
        # clean-looking document anyway.
        _write_release(tmp_path, monkeypatch, release_id="r1")
        _write_release(tmp_path, monkeypatch, release_id="r2")

        path = tmp_path / "r1.json"
        record = json.loads(path.read_text())
        record["objective"] = "quietly rewritten after the fact"
        path.write_text(json.dumps(record, indent=2))

        report = render_report("r2")
        assert "Chain integrity: FAILED." in report

    def test_missing_release_id_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(release_record, "RELEASES_DIR", str(tmp_path))
        try:
            render_report("does-not-exist")
            assert False, "expected SystemExit"
        except SystemExit as e:
            assert "does-not-exist" in str(e)
