# Governance Policies

Written 2026-08-04 to close flat Gaps identified in `NIST_AI_RMF_GAP_ANALYSIS.md`
(Stage 2, 2026-07-31). Each section below maps to specific subcategories —
see the gap analysis doc for the full traceability matrix. Same standard as
the rest of Guardian's NIST work: every claim here is either a real,
existing practice being written down for the first time, or a concrete
mechanism built alongside the policy, not an aspirational statement.

## Risk Tolerance Statement

_Closes GOVERN 1.3, MAP 1.5._

Guardian doesn't have a formally documented risk-tolerance statement, but
real risk-tolerance decisions get made constantly. This section makes the
pattern behind those decisions explicit, based on real precedent:

**What gets accepted:**
- Reversible actions, even risky-looking ones (config changes with a diff
  history, model retrains with an archive-before-overwrite step — see
  below).
- Irreversible actions with contained blast radius (e.g. bringing
  Prometheus/Loki/Tempo/Alertmanager configs under version control via
  symlink — a bad edit breaks one service, not the whole system).
- Adding detection/visibility even when the underlying thing being watched
  isn't being fixed yet (e.g. `aiops_security_sudoers_d_changed` was added
  specifically because the sudoers files themselves were judged too risky
  to bring under the same version-control pattern).

**What gets declined:**
- Irreversible actions with uncontained blast radius. Concrete precedent:
  symlinking `/etc/sudoers.d/*` into git was proposed and explicitly
  declined (2026-08-03) — a bad swap could break every passwordless-sudo
  path live services depend on, and `sudo` itself often refuses symlinked
  sudoers files outright. The mitigation (change-detection instead of
  version control) was judged sufficient without taking on that risk.
- New standing privilege for narrow, one-off value. Concrete precedent:
  reading real Intel RAPL energy counters for MEASURE 2.12 below would
  need new sudo scope; a clearly-labeled estimate was used instead rather
  than expand privilege for one metric.
- Certifiable/compliance claims without the substance to back them.
  Concrete precedent: NIST AI RMF was chosen over ISO/IEC 42001 specifically
  because ISO 42001 is a certifiable standard, and referencing it without
  being certified invites an unanswerable "so are you certified?" question.
- Scope expansion for its own sake. `VISION.md`'s own litmus test ("does
  this collect evidence, explain behavior, attribute responsibility, or
  increase trust in autonomous systems?") is itself a risk-tolerance
  filter — features that don't clear it get left for a different project.

**How this gets checked:** informally, by the same person making both the
proposal and the decision (solo project — see GOVERN's overall weakness in
the gap analysis for why this isn't a resourced/independent process). The
value of writing it down is consistency across sessions, not independence
of judgment.

## Model Decommissioning / Phase-Out Process

_Closes GOVERN 1.7._

**Before:** one manual, one-off archive existed (`old/` directory, a single
autoencoder snapshot from 2026-06-11) and nothing since, despite several
retrains in the meantime. No repeatable process, no criteria for when to
archive, no code enforcing it.

**Now:** `retrain_common.py`'s `archive_current_models()` runs automatically
at the start of every retrain script's `save_model` step, before the live
model/scaler/threshold files are overwritten. Whatever's currently live
gets copied to `old/<model>_<date>_<time>/` first. No retention/pruning —
these files are tens of KB each, so keeping every snapshot indefinitely has
no real storage cost and maximizes traceability. Wired into all three
retrain scripts (`retrain_recent_knn.py`, `retrain_recent_iforest.py`,
`retrain_recent.py`), each OTel `save_model` span records `archived_to`
when a snapshot was made.

**Real validation, same day it was built:** while testing this, running
`retrain_recent_iforest.py` for real surfaced a genuine bug — that script
still had the `RECENT_ROWS = 100000` default (the same class of bug already
fixed in the KNN and autoencoder scripts), so it trained on 46,645 rows
instead of a recent 2000-row window, and Phase 5's `behavioral_policy.py`
correctly failed verification (`row_count 46645 above policy maximum
5000`). The archive made recovery trivial: the previous good model was
already snapshotted, so it was restored immediately, the `RECENT_ROWS` bug
was fixed, and a corrected retrain was run and verified clean. The live
`aiops-watchdog-iforest.service` was never at risk — it wasn't restarted
during any of this, so it never loaded the bad model. This is real evidence
the mechanism works, not a hypothetical.

## Third-Party Library Risk Review

_Closes GOVERN 6.1, MAP 4.1._

Checked directly against installed packages (`pip show`), not assumed:

| Library | Version | License | Notes |
|---|---|---|---|
| scikit-learn | 1.7.2 | BSD-3-Clause | Permissive, no copyleft, no patent clause. Large, well-maintained project — low bus-factor risk. |
| TensorFlow | 2.21.0 | Apache-2.0 | Permissive, includes an explicit patent grant. Backed by Google — low bus-factor risk. |
| pyod | 2.0.5 | BSD-2-Clause (verified from the package's own `LICENSE` file, not just its PyPI classifier) | Permissive. Smaller, more single-author-originated project than the other two (Yue Zhao) — comparatively higher bus-factor risk, though it's a widely-used, actively maintained library, not an obscure one. |

**Assessment:** no license/IP risk from any of the three — all permissive
open-source, all compatible with Guardian being a public repo. The
practical risk that exists is operational (a library failing to load or
breaking on an upgrade), not legal — see the failure-contingency section
below for how that's actually handled today.

## Third-Party / Model Load Failure Contingency

_Closes GOVERN 6.2._

Checked what actually happens today, not designed a new theoretical
process:

**Failure mode 1: TensorFlow can't find CUDA libraries.** Real, currently
happening on every autoencoder watchdog/retrain run on this host (no GPU
driver issue — this is a TensorFlow/CUDA library discovery problem,
separate from the actual NVIDIA driver being healthy). Confirmed via
`journalctl`: TensorFlow logs `Could not find cuda drivers on your
machine, GPU will not be used` at startup and continues normally on CPU.
**This is real, working graceful degradation** for this specific failure
mode — not designed, just how TensorFlow already behaves, now documented
as intentional rather than an unexplained log line.

**Failure mode 2: a model file is missing or corrupted, or a library fails
to import.** Checked `aiops-watchdog-{knn,iforest,autoencoder}.py` and
`watchdog_common.py`: `load_model()` is called with no `try`/`except`
around it. A missing or corrupted model file, or an import failure, raises
an unhandled exception and the process exits. Since these run as systemd
services with `Restart=always`, this becomes a crash-loop rather than a
graceful fallback. **No purpose-built handling exists for this case** —
but it isn't silent: `aiops-watchdog-logs.py`'s `aiops_logs_active{unit=...}`
gauge would show the unit as inactive, which feeds
`aiops-watchdog-priority.py`'s tiering. Real, if indirect, detection — not
a designed contingency.

**Honest status:** this subcategory is closed in the sense that current
behavior is now accurately documented (a real gap, not a silently-assumed
"probably fine"). The failure mode 2 gap — no graceful degradation, only
crash-and-detect — is real and could be strengthened later (e.g. catching
the load exception and serving a "model unavailable" state instead of
crash-looping) if it's ever worth the engineering effort for a solo
project. Not built now — flagged, not fixed, and stated as such rather than
overclaimed.
