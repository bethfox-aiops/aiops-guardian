"""
test_guardian_health.py

A first, small test suite for aiops-guardian-health.py, covering the two
functions we already hand-verified during the 2026-07-20 session:
_ufw_denies_port_externally() (where we found and fixed a real IPv6 parsing
bug) and calculate_ai_risk_score().

Run with: pytest test_guardian_health.py -v
"""

import importlib.util
import os
import time

import pytest

# aiops-guardian-health.py isn't a normal importable module (the filename
# has hyphens), so we load it directly from its file path.
_MODULE_PATH = os.path.join(os.path.dirname(__file__), "aiops-guardian-health.py")
_spec = importlib.util.spec_from_file_location("guardian_health", _MODULE_PATH)
gh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gh)


@pytest.fixture
def fake_ufw(monkeypatch):
    """Lets each test set the raw `sudo ufw status` output the code sees,
    without needing real ufw or real sudo access."""
    def _set(status_text):
        monkeypatch.setitem(gh._ufw_status_cache, "out", status_text)
        monkeypatch.setitem(gh._ufw_status_cache, "ts", time.time())
    return _set


class TestUfwDeniesPortExternally:
    def test_inactive_ufw_returns_false(self, fake_ufw):
        fake_ufw("Status: inactive\n")
        assert gh._ufw_denies_port_externally(8011) is False

    def test_deny_from_anywhere_returns_true(self, fake_ufw):
        fake_ufw("""Status: active

To                         Action      From
--                         ------      ----
8011                       DENY        Anywhere
""")
        assert gh._ufw_denies_port_externally(8011) is True

    def test_conflicting_allow_and_deny_returns_false(self, fake_ufw):
        # An ALLOW-from-Anywhere rule alongside the DENY means the port
        # isn't actually fully blocked -- should NOT report as denied.
        fake_ufw("""Status: active

To                         Action      From
--                         ------      ----
8011                       ALLOW       Anywhere
8011                       DENY        Anywhere
""")
        assert gh._ufw_denies_port_externally(8011) is False

    def test_v6_only_deny_returns_true(self, fake_ufw):
        # Regression test for the real bug found/fixed on 2026-07-20: a
        # "(v6)" marker after the port used to shift the action/source
        # columns, silently missing IPv6-only DENY rules.
        fake_ufw("""Status: active

To                         Action      From
--                         ------      ----
8011 (v6)                  DENY        Anywhere (v6)
""")
        assert gh._ufw_denies_port_externally(8011) is True

    def test_port_not_listed_returns_false(self, fake_ufw):
        fake_ufw("""Status: active

To                         Action      From
--                         ------      ----
22                         ALLOW       Anywhere
""")
        assert gh._ufw_denies_port_externally(8011) is False


class TestUfwStatusCaching:
    """Covers _get_ufw_status_cached() -- the fix from 2026-07-20 that made
    several ufw-checking functions share one `sudo ufw status` call instead
    of each shelling out separately. Uses a fake subprocess.run so no real
    ufw/sudo access is needed, and just counts how many times it's called."""

    @pytest.fixture(autouse=True)
    def _reset_cache_and_fake_subprocess(self, monkeypatch):
        # Every test starts with a clean, expired cache so tests can't
        # interfere with each other via the shared module-level dict.
        monkeypatch.setitem(gh._ufw_status_cache, "ts", 0.0)
        monkeypatch.setitem(gh._ufw_status_cache, "out", "")

        self.call_count = 0

        def fake_run(*args, **kwargs):
            self.call_count += 1
            return type("FakeResult", (), {"stdout": "Status: active\n"})()

        monkeypatch.setattr(gh.subprocess, "run", fake_run)

    def test_first_call_shells_out_once(self):
        gh._get_ufw_status_cached()
        assert self.call_count == 1

    def test_second_call_within_window_reuses_cache(self):
        gh._get_ufw_status_cached()
        gh._get_ufw_status_cached()
        gh._get_ufw_status_cached()
        assert self.call_count == 1  # still just the one real call

    def test_full_health_check_cycle_shells_out_once(self):
        # get_ufw_enabled() + 4 port checks == 5 callers sharing one cache.
        gh.get_ufw_enabled()
        for port in [8011, 8012, 8013, 8014]:
            gh._ufw_denies_port_externally(port)
        assert self.call_count == 1

    def test_cache_expires_after_max_age(self):
        gh._get_ufw_status_cached()
        assert self.call_count == 1

        gh._ufw_status_cache["ts"] -= 30  # simulate 30s passing (max_age is 25s)
        gh._get_ufw_status_cached()
        assert self.call_count == 2


class TestScoreSecurityBase:
    """Covers _score_security_base(), pulled out of compute_security() on
    2026-07-23 specifically to make it testable without mocking the ~25
    other checks compute_security() also runs. A few of these tests exist
    because the real behavior is more subtle than it looks at a glance --
    see the comments on each."""

    def test_all_clear_returns_zero(self):
        deduction, issue_code, recommendation = gh._score_security_base(
            updates=0, ufw=1, root_ssh=0, failed_logins=0, open_ports=10)
        assert (deduction, issue_code, recommendation) == (0, 0, 0)

    def test_ufw_disabled_alone(self):
        deduction, issue_code, recommendation = gh._score_security_base(
            updates=0, ufw=0, root_ssh=0, failed_logins=0, open_ports=10)
        assert (deduction, issue_code, recommendation) == (30, 1, 1)

    def test_updates_pending_alone(self):
        deduction, issue_code, recommendation = gh._score_security_base(
            updates=5, ufw=1, root_ssh=0, failed_logins=0, open_ports=10)
        assert (deduction, issue_code, recommendation) == (10, 2, 2)  # min(5*2, 30)

    def test_updates_deduction_caps_at_30(self):
        deduction, _, _ = gh._score_security_base(
            updates=100, ufw=1, root_ssh=0, failed_logins=0, open_ports=10)
        assert deduction == 30  # not 200

    def test_root_ssh_alone(self):
        deduction, issue_code, recommendation = gh._score_security_base(
            updates=0, ufw=1, root_ssh=1, failed_logins=0, open_ports=10)
        assert (deduction, issue_code, recommendation) == (30, 3, 3)

    def test_failed_logins_affects_deduction_but_not_issue_code(self):
        # Subtle: failed_logins isn't part of issue_count at all, so it can
        # add to the deduction while leaving issue_code/recommendation at 0.
        deduction, issue_code, recommendation = gh._score_security_base(
            updates=0, ufw=1, root_ssh=0, failed_logins=25, open_ports=10)
        assert deduction == 20
        assert (issue_code, recommendation) == (0, 0)

    def test_open_ports_between_25_and_50_affects_deduction_but_not_issue_code(self):
        # Subtle: the issue_code/recommendation cascade only checks
        # open_ports > 50, not > 25 -- so this range silently adds to the
        # deduction without ever surfacing as an issue code.
        deduction, issue_code, recommendation = gh._score_security_base(
            updates=0, ufw=1, root_ssh=0, failed_logins=0, open_ports=30)
        assert deduction == 10
        assert (issue_code, recommendation) == (0, 0)

    def test_open_ports_above_50(self):
        deduction, issue_code, recommendation = gh._score_security_base(
            updates=0, ufw=1, root_ssh=0, failed_logins=0, open_ports=60)
        assert (deduction, issue_code, recommendation) == (20, 5, 4)

    def test_multiple_issues_sets_issue_code_4(self):
        # ufw disabled AND updates pending -- issue_count=2, so issue_code
        # becomes the generic "multiple issues" code (4), even though
        # recommendation still follows the priority order below.
        _, issue_code, recommendation = gh._score_security_base(
            updates=5, ufw=0, root_ssh=0, failed_logins=0, open_ports=10)
        assert issue_code == 4
        assert recommendation == 1  # ufw still wins priority for the recommendation

    def test_recommendation_priority_order(self):
        # updates and root_ssh both wrong, no ufw issue -- recommendation
        # should follow the priority order (updates checked before
        # root_ssh), not just "whatever's wrong."
        _, _, recommendation = gh._score_security_base(
            updates=5, ufw=1, root_ssh=1, failed_logins=0, open_ports=10)
        assert recommendation == 2  # updates, not root_ssh


class TestCalculateAiRiskScore:
    """calculate_ai_risk_score() returns (score, factors) where factors is a
    list of {"key", "detail", "points"} for each *active* risk factor. The
    2026-09-10 rewrite made three behavioural changes covered here:
      - `tools` (installed packages) and Guardian's own processes no longer
        cost points, so a clean host can actually reach 100
      - penalties scale with count and cap, instead of a flat if-nonzero
      - factors carry a stable `key` (drives the ai_risk_reason metric)
    """

    def _details(self, factors):
        return [f["detail"] for f in factors]

    def test_no_risk_factors_returns_100(self):
        score, factors = gh.calculate_ai_risk_score(
            tools=0, processes=0, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=0,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        assert score == 100
        assert factors == []

    def test_installed_tools_are_informational_only(self):
        # Regression: installed AI packages used to cost -10 and, with
        # Guardian's own deps always present, permanently pinned the score
        # at <= 80. `tools` must now contribute nothing.
        score, factors = gh.calculate_ai_risk_score(
            tools=9, processes=0, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=0,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        assert score == 100
        assert factors == []

    def test_single_risk_factor_deducts_correctly(self):
        score, factors = gh.calculate_ai_risk_score(
            tools=0, processes=0, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=1,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        assert score == 90  # 1 shadow model * 10 pts/unit
        assert self._details(factors) == ["1 model file(s) outside the known model dir"]
        assert factors[0]["key"] == "shadow_models"

    def test_penalty_scales_with_count(self):
        score, factors = gh.calculate_ai_risk_score(
            tools=0, processes=0, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=2,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        assert score == 80  # 2 * 10, still under the cap

    def test_penalty_caps_and_is_marked(self):
        # shadow_models cap is 25; 4 * 10 = 40 would blow past it.
        score, factors = gh.calculate_ai_risk_score(
            tools=0, processes=0, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=4,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        assert score == 75  # 100 - 25 (capped), not 100 - 40
        assert factors[0]["points"] == 25
        assert "(capped)" in factors[0]["detail"]

    def test_third_party_processes_scale_and_cap(self):
        # processes DO still cost points (third-party AI runtimes), scaled
        # 4/unit, cap 12 -- Guardian's own are filtered out upstream.
        score_one, _ = gh.calculate_ai_risk_score(
            tools=0, processes=1, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=0,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        score_many, _ = gh.calculate_ai_risk_score(
            tools=0, processes=10, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=0,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        assert score_one == 96      # -4
        assert score_many == 88     # -12 (capped), not -40

    def test_factors_sorted_by_points_descending(self):
        _, factors = gh.calculate_ai_risk_score(
            tools=0, processes=1, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=0,
            training_changed=1, model_age_drift=0, gpu_spike=0,
        )
        points = [f["points"] for f in factors]
        assert points == sorted(points, reverse=True)
        assert factors[0]["key"] == "training_data_tampered"  # 25 > 4

    def test_score_never_goes_below_zero(self):
        score, factors = gh.calculate_ai_risk_score(
            tools=1, processes=1, api_keys=1, watchdog_external=1,
            exposed_keys=1, llm_conns=1, shadow_models=1,
            training_changed=1, model_age_drift=1, gpu_spike=1,
        )
        # Deductions here add up to well over 100 -- score must floor at 0.
        assert score == 0
        # 9 active factors: every input except `tools`, which is informational.
        assert len(factors) == 9
        assert all(f["points"] > 0 for f in factors)


class TestCheckTrainingDataChanged:
    """Covers the 2026-09-10 rewrite of check_training_data_changed(). The
    old version hashed the first 100 KB, which never changes under normal
    append-only writes; the new one hashes a byte region frozen at the
    previous cycle's EOF, so a plain append leaves it identical but an
    in-place rewrite of already-written rows (retrain-window poisoning) or
    a truncation is caught."""

    @pytest.fixture
    def csv(self, tmp_path, monkeypatch):
        import guardian_ai_risk as air
        p = tmp_path / "metrics.csv"
        # ~290 KB so prev_size clears the 260 KB region+skip threshold.
        rows = [
            f"2026-09-10T{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}"
            f".000000,{i},1.0,2.0,3.0,4.0,5.0,6.0,0.0,0.0,0.0\n"
            for i in range(7000)
        ]
        p.write_text("".join(rows))
        monkeypatch.setattr(air, "DATA_FILE", str(p))
        monkeypatch.setitem(air._prev, "training_data_size", None)
        monkeypatch.setitem(air._prev, "training_data_region_hash", None)
        return p, air

    def test_first_call_only_seeds(self, csv):
        p, air = csv
        assert air.check_training_data_changed() == 0

    def test_plain_appends_are_not_flagged(self, csv):
        p, air = csv
        air.check_training_data_changed()  # seed
        for batch in range(3):
            with p.open("a") as f:
                for i in range(7000 + batch * 50, 7050 + batch * 50):
                    f.write(f"2026-09-11T00:00:{i % 60:02d}.0,{i},1,2,3,4,5,6,0,0,0\n")
            assert air.check_training_data_changed() == 0

    def test_in_place_rewrite_is_flagged(self, csv):
        p, air = csv
        air.check_training_data_changed()  # seed
        lines = p.read_bytes().split(b"\n")
        idx = len(lines) - 1000  # ~1000 rows from EOF -> inside the frozen region
        lines[idx] = b"9" * len(lines[idx])  # same length -> true in-place edit
        p.write_bytes(b"\n".join(lines))
        assert air.check_training_data_changed() == 1

    def test_truncation_is_flagged(self, csv):
        p, air = csv
        air.check_training_data_changed()  # seed
        data = p.read_bytes()
        p.write_bytes(data[: len(data) // 2])
        assert air.check_training_data_changed() == 1
