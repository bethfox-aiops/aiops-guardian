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


class TestCalculateAiRiskScore:
    def test_no_risk_factors_returns_100(self):
        score, reasons = gh.calculate_ai_risk_score(
            tools=0, processes=0, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=0,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        assert score == 100
        assert reasons == []

    def test_single_risk_factor_deducts_correctly(self):
        score, reasons = gh.calculate_ai_risk_score(
            tools=0, processes=0, api_keys=0, watchdog_external=0,
            exposed_keys=0, llm_conns=0, shadow_models=1,
            training_changed=0, model_age_drift=0, gpu_spike=0,
        )
        assert score == 80  # shadow_models > 0 costs 20 points
        assert reasons == ["1 shadow model file(s) found"]

    def test_score_never_goes_below_zero(self):
        score, reasons = gh.calculate_ai_risk_score(
            tools=1, processes=1, api_keys=1, watchdog_external=1,
            exposed_keys=1, llm_conns=1, shadow_models=1,
            training_changed=1, model_age_drift=1, gpu_spike=1,
        )
        # Deductions here add up to well over 100 -- score must floor at 0,
        # never go negative.
        assert score == 0
        assert len(reasons) == 10
