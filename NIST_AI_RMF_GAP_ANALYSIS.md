# NIST AI RMF Gap Analysis — Guardian

Stage 2 of `ROADMAP.md` Phase 7 item 3: an honest, subcategory-by-subcategory
comparison of Guardian's actual code and documentation against every
category/subcategory in `NIST_AI_RMF_REFERENCE.md` (which holds the verbatim
framework text this document doesn't repeat). Written against real evidence
— specific files, metrics, services, and documented incidents — not
paraphrase or aspiration.

**Scope note:** Guardian is evaluated here as a **solo, single-user personal
system** with its own AI workloads (the KNN/Isolation Forest/Autoencoder
retrain pipelines) as the "AI system" being risk-managed, per the scoping
decision in `NIST_AI_RMF_REFERENCE.md`. Several subcategories assume an
organization — a workforce, an executive team, external stakeholders, a
supply chain — that doesn't exist at this scale. Marking those **Not
applicable** rather than forcing a fit is itself the honest answer, not a
missing answer.

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ **Satisfied** | Real, existing evidence — cited below |
| 🟡 **Partial** | Some real evidence, meaningfully incomplete |
| 🔵 **Planned** | Genuine gap, but already a stated intention somewhere in `ROADMAP.md`/`VISION.md` |
| ❌ **Gap** | No real evidence, no stated plan |
| ⚪ **Not applicable** | Doesn't apply at this system's actual scale |

## Summary rollup

| Function | ✅ | 🟡 | 🔵 | ❌ | ⚪ | Total |
|---|---|---|---|---|---|---|
| GOVERN | 7 | 6 | 0 | 0 | 6 | 19 |
| MAP | 9 | 5 | 0 | 0 | 4 | 18 |
| MEASURE | 8 | 6 | 0 | 0 | 8 | 22 |
| MANAGE | 6 | 5 | 0 | 0 | 2 | 13 |
| **Total** | **30** | **22** | **0** | **0** | **20** | **72** |

**Notable, not massaged:** zero subcategories are marked 🔵 Planned. Every
real capability cited below was built in response to an actual incident or
need (a drift pattern, a discovered sudo gap, a real crash) — nothing here
was built to fill a framework checkbox. That's a genuine strength worth
stating plainly rather than a gap to explain away.

**Updated 2026-08-04:** the 7 flat Gaps from the original 2026-07-31 pass
are now closed — 5 moved to ✅ Satisfied (real docs plus, where it made
sense, real enforcing code: `GOVERNANCE_POLICIES.md`, automated model
archiving), 2 moved to 🟡 Partial rather than being overclaimed as fully
closed (GOVERN 6.2's model-load-failure handling is real for one failure
mode and honestly still absent for another; MEASURE 2.12's energy figure
is a documented estimate, not a true hardware measurement). See
`GOVERNANCE_POLICIES.md` for the full writeup of what changed and why.

**Full fresh pass, 2026-08-05:** all 72 subcategories re-verified against
current real evidence, not assumed unchanged since 7/31 — same rigor as
the original Stage 2 pass. Three subcategories genuinely earned an
upgrade to ✅ this week, each tied to a specific new artifact, not general
drift: **GOVERN 1.4** and **GOVERN 4.1** (`GOVERNANCE_POLICIES.md`'s risk
tolerance statement is a much more precise match for "risk management
process made transparent" than the general project-transparency evidence
used before, and its real declined-for-safety precedent — the
sudoers-symlink decline — is concrete safety-first *practice*, not just
`VISION.md`'s stated intent), and **MAP 4.2** (the same doc's
failure-contingency section finally scopes internal risk controls
specifically to third-party components, closing the exact gap the old
rating named). Several other rows got stronger supporting citations
(reinforcing existing ratings, not changing them) from this week's
infra-config version control, sudoers.d monitoring, and the live
iForest bad-retrain recovery. One stale claim corrected: MEASURE 3.2 used
to say a UPS "doesn't exist yet" — one was bought and installed since,
though Guardian still can't see it directly (NUT integration was
declined to preserve PowerPanel's safety feature), so the rating itself
didn't change, just the note.

**Real finding from doing this properly, not just confirming the 7-gap
subset:** this week's actual operational incidents — the reboot-triggered
and suspend/resume-triggered ML drift discoveries, and the resulting
autoencoder threshold recalibration — exist only in session/assistant
memory, not in any git-tracked repo doc (`OPERATIONS_MANUAL.md`,
`ROADMAP.md`, `CLAUDE.md`). They weren't cited as evidence anywhere in
this document for exactly that reason — private memory isn't verifiable,
repo-tracked evidence. Writing these up into the repo (matching how the
original 4-recurrence promtail-drift saga was documented) would be a
real, valuable follow-up, both for this assessment and for the project's
own operational completeness — not done as part of this pass, since it's
a real chunk of work beyond "re-verify the assessment," but worth
surfacing rather than silently leaving out.

Updated rollup: 30 Satisfied, 22 Partial, 0 Planned, 0 Gap, 20 Not
applicable.

**Stage 6 (targeted refresh), 2026-09-20:** 46 days since the last full
pass -- not a blind re-check of all 72 (most cited evidence is unchanged
code/docs that a full re-run would just reconfirm), but every row touched
below is tied to a real, specific thing that happened since 8/5, verified
against current code/metrics, not assumed. No row's **status** changed in
this pass; several earned materially stronger evidence, and one (MANAGE
4.3) needed an honestly harder note, not a better one.

- **MEASURE 1.2, 2.6, 2.9** — the `guardian_ai_risk.py` scoring rework
  (2026-09-10/11) is real new evidence on three fronts at once: metrics
  genuinely reassessed in response to a found flaw (not on a cadence,
  still 1.2's gap), a new `ai_risk_collection_ok` gauge + `safe_check()`
  wrapper that distinguishes a failed check from a verified-clean one
  (2.6 — the model-file-crash-loop gap it doesn't touch is still open,
  so still 🟡), and `ai_risk_reason{reason=<key>}=<points>` explaining
  *why* the score is what it is, the same pattern `behavioral_policy.py`
  already earned 2.9 credit for.
- **MEASURE 2.7 / GOVERN 4.1** — three more concrete "found a real gap,
  fixed it" instances since 8/5: MD5→SHA-256 across every integrity hash
  (MD5's broken collision resistance was a real, if narrow, weakness),
  the systemd-unit check fixed from filename+mtime (spoofable) to actual
  content, and the Alertmanager finding below. Reinforces both rows'
  existing ✅, doesn't need to raise it further.
- **MEASURE 3.2** — DESKTOP-0AJUKU3 (the same host this row already
  tracks) crashed again, 2026-09-14/15, with new, specific proxy evidence
  this time: `guardian_disk_health.ps1` caught a real Event ID 153 (I/O
  device error) in the 24h before the crash, and disk busy% spiked from
  ~0.5% to 32% in the last sample before it went dark — the same *kind*
  of proxy signal the row already cites, now with a third real occurrence
  behind it.
- **MANAGE 4.1 — sharper, not just reinforced.** This host itself
  suspended for ~65 hours (2026-09-11 22:14 UTC → 2026-09-14 22:04 UTC).
  For that entire window Prometheus, Alertmanager, and every watchdog
  were simply not running — not degraded, not silent-but-alive, off.
  Nothing could have noticed a real incident during that window, because
  the thing that notices things was itself asleep. Found by accident
  (an unrelated question about a Windows host), not by any Guardian
  mechanism. The row's existing "strong on monitoring" framing didn't
  have language for this failure mode at all; it does now.
- **MANAGE 4.3 — the honest finding this stage exists to catch.** The row
  reads "Alertmanager → Slack, reaching the one operator that exists."
  Discovered 2026-09-20: `alertmanager_notifications_failed_total{integration="slack"}`
  showed 234 of 367 send attempts (64%) had been failing since the service
  started (2026-08-26), silently, nothing in the journal. Root cause:
  `.CommonAnnotations` renders empty whenever a group holds more than one
  alert with per-instance text (confirmed live: `WindowsHostUnreachable`
  grouped across both Windows hosts, two distinct summaries, empty common
  text, Slack rejects an empty message) — and that's exactly why
  DESKTOP-0AJUKU3's 11-hour outage on 9/15 never reached Slack. Fixed the
  same day (template now renders per-alert). The mechanism existing is not
  the same claim as the mechanism working, and for three weeks it was the
  former without the latter. Still 🟡 — arguably this is *why* it was
  never ✅ — but the note now says what actually happened instead of what
  was assumed to be happening.
- **MANAGE 2.3** — two more real "unknown risk discovered, responded to
  same day" instances since 8/5: the `ai_risk_score` baseline-pinning bug
  and the Alertmanager finding immediately above. Reinforces existing ✅.
- **Correction to this stage's own first draft:** initially carried
  forward the 2026-08-05 note that the reboot/suspend-triggered ML-drift
  incidents and the autoencoder threshold recalibration were still
  undocumented. Re-checked against `OPERATIONS_MANUAL.md` directly rather
  than trusting the earlier note, and they're not — Chapter 10.1 already
  covers both in real detail (dated 2026-08-03/-04: the GPU-memory
  step-change root cause, the suspend/resume false-positive signature,
  the threshold's move to the 99th percentile). The 8/5 note was already
  stale by the time this stage started; repeating it uncorrected would
  have been exactly the kind of unverified claim this document exists to
  avoid.

Rollup unchanged this stage: 30 Satisfied, 22 Partial, 0 Planned, 0 Gap,
20 Not applicable — every row above kept its letter grade; what changed
is how honestly the notes describe it.

## Three-layer view (for different audiences)

| NIST Outcome (executive) | Guardian Capability (architect) | Technical Implementation (engineer) |
|---|---|---|
| Monitor AI risks over time | Behavioral Attestation | `behavioral_policy.py`, the three ML watchdogs, Prometheus |
| Verify AI behaved as expected | Runtime policy verification | `verify()`, `POLICIES` dict, `test_behavioral_policy.py` |
| Prove evidence wasn't altered | Tamper-evident release ledger | `release_record.py`, `verify_chain.py`, SHA-256 hash chain |
| Communicate AI risk to a human | Alerting + evidence reporting | Alertmanager → Slack, `release_report.py` |
| Human authorizes AI-taken action | Human approval workflow | `aiops-approval.service`, port 8020 |
| Govern the tooling handed to an AI agent | Scoped sudoers, ticket-validated tracing | `trace_suspect.sh`, `ufw_guard.sh`, `.trace_ticket` |

---

## GOVERN

| # | Guardian Capability | Evidence | Status | Notes |
|---|---|---|---|---|
| 1.1 | None | — | ⚪ | No specific AI legal/regulatory regime identified as applicable to a personal lab system. |
| 1.2 | Trustworthy-AI principles stated as policy | `VISION.md`'s litmus test (collects evidence / explains behavior / attributes responsibility / increases trust) | 🟡 | Real, documented — but not systematically mapped to all seven trustworthiness characteristics. |
| 1.3 | Risk tolerance statement | `GOVERNANCE_POLICIES.md`'s Risk Tolerance Statement section | ✅ | Written 2026-08-04, grounded in real precedent (the sudoers-symlink decline, the NIST-over-ISO choice), not abstract principles. |
| 1.4 | Risk management process made transparent, with real outcomes | `GOVERNANCE_POLICIES.md`'s Risk Tolerance Statement (the "what gets accepted / declined" pattern, with real precedent for each) | ✅ | Upgraded 2026-08-05 — the subcategory specifically wants the risk management *process and its outcomes* transparent, and `GOVERNANCE_POLICIES.md` is a much more direct match than general project transparency (`VISION.md`/`ROADMAP.md`, still real supporting evidence). |
| 1.5 | Periodic doc-vs-reality audits | 2026-07-27 and 2026-07-29 audits, documented in `OPERATIONS_MANUAL.md` Ch. 10 | 🟡 | Real, repeatable practice; no fixed schedule or defined roles (solo project). |
| 1.6 | Informal AI-system discovery | `guardian_ai_risk.py`'s `AI_PROCESSES_RUNNING`/`AI_TOOLS` checks | 🟡 | Detects AI activity automatically; not a governed, resourced inventory. |
| 1.7 | Automated model archiving | `retrain_common.py`'s `archive_current_models()`, wired into all 3 retrain scripts; `GOVERNANCE_POLICIES.md` | ✅ | Real mechanism, not just a written policy — validated same day it was built, when it made recovery trivial after a real bad-retrain (see `GOVERNANCE_POLICIES.md`). |
| 2.1 | N/A — solo operator | — | ⚪ | One person; no roles to document. |
| 2.2 | N/A — no other personnel | — | ⚪ | |
| 2.3 | Operator approves AI-driven changes interactively | Every session's approval pattern; `aiops-approval.service` | 🟡 | Real but informal — one person acting as both "leadership" and operator. |
| 3.1 | N/A — no team | — | ⚪ | |
| 3.2 | Human-oversight pattern for AI actions | `aiops-approval.service`; this session's own "ask before risky actions" practice, now also written down in `GOVERNANCE_POLICIES.md`'s risk-tolerance framework | 🟡 | Real mechanism, and now partly written down (2026-08-05) — but `GOVERNANCE_POLICIES.md` is about risk tolerance broadly, not a dedicated human-AI role-boundary policy, so still short of a formal match. |
| 4.1 | Safety-first mindset practiced, not just documented | `GOVERNANCE_POLICIES.md`'s Risk Tolerance Statement (the sudoers-symlink decline, made specifically to protect the infrastructure Guardian's own AI-driven tooling depends on); the automated model-archiving mechanism that made the 2026-08-04 bad-iForest-retrain trivially recoverable; the 2026-09-11 MD5→SHA-256 integrity-hash upgrade, done proactively rather than in response to an exploit | ✅ | Upgraded 2026-08-05 — `VISION.md`'s narrow-scope section was real but design-doc-only; this is the same safety-first thinking demonstrated in actual AI-system decisions and a real caught mistake, not just stated intent. Reinforced 2026-09-20. |
| 4.2 | Risks/impacts documented and communicated publicly | `OPERATIONS_MANUAL.md` Ch. 10 (Known Gaps), `ROADMAP.md`'s self-critique sections, public LinkedIn posts about real incidents | ✅ | Unusually well covered — publicly documented, not just internally. |
| 4.3 | Testing, incident ID, info sharing all real | pytest + GitHub Actions CI; the `trace_suspect.sh`/`ufw_guard.sh` incident writeups | ✅ | |
| 5.1 | N/A — no external stakeholders yet | — | ⚪ | Revisit if `EDGE_ARCHITECTURE.md`'s consulting plan ever gets real customers. |
| 5.2 | N/A — same reason | — | ⚪ | |
| 6.1 | Third-party license review | `GOVERNANCE_POLICIES.md`'s license table, verified against installed package metadata | ✅ | All three (scikit-learn, TensorFlow, pyod) confirmed permissive OSS licenses, no IP/copyleft risk. |
| 6.2 | Documented, partially-handled failure behavior | `GOVERNANCE_POLICIES.md`'s failure-contingency section | 🟡 | TensorFlow/CUDA failure gracefully degrades to CPU (real, already happening); a missing/corrupt model file still crash-loops rather than degrading gracefully — real gap, honestly flagged, not fixed. |

---

## MAP

| # | Guardian Capability | Evidence | Status | Notes |
|---|---|---|---|---|
| 1.1 | Intended purpose/context documented | `VISION.md` — two flagship use cases, deliberately narrow scope | ✅ | |
| 1.2 | N/A — solo project | — | ⚪ | |
| 1.3 | Mission/goals for AI tech documented | `VISION.md`'s north star question, five-engine architecture | ✅ | |
| 1.4 | Business/career value understood | Career-portfolio motivation (tracked in project context, not in Guardian's own docs) | 🟡 | Real, but not written into the repo's own documentation. |
| 1.5 | Risk tolerance statement | `GOVERNANCE_POLICIES.md`'s Risk Tolerance Statement section | ✅ | Same doc as GOVERN 1.3 — explicit "what gets accepted / declined" pattern with real precedent cited for each. |
| 1.6 | Scope/socio-technical implications considered | `VISION.md`'s litmus test | 🟡 | |
| 2.1 | Specific ML tasks/methods defined | KNN, Isolation Forest, Autoencoder — named, documented in `CLAUDE.md`/`OPERATIONS_MANUAL.md` | ✅ | |
| 2.2 | Known limits documented | The KNN sustained-drift pattern (4 recurrences), the chronic `ssh` false-positive in `all_check.service` | 🟡 | Real known-limits documentation; not exhaustive. |
| 2.3 | TEVV practiced | pytest suite, Phase 5 behavioral verification, the real defect-demo, tagged `NIST_AI_RMF_TAGS["MAP 2.3"]` in `behavioral_policy.py` | ✅ | |
| 3.1 | Benefits documented | `VISION.md`'s flagship use cases | ✅ | |
| 3.2 | Costs of AI errors documented | KNN drift's alert-fatigue cost, documented in `OPERATIONS_MANUAL.md` Ch. 10 | 🟡 | |
| 3.3 | Application scope specified | `VISION.md`'s "deliberately narrow scope" section, verbatim | ✅ | This is the clearest, most direct match in the whole document. |
| 3.4 | N/A — solo operator | — | ⚪ | No formal proficiency/certification process for one person. |
| 3.5 | Human oversight process | `aiops-approval.service`; Phase 7 item 1 (scoped agent identity) is literally this subcategory's next step | 🟡 | Real mechanism; not comprehensively documented as a process. |
| 4.1 | Third-party library risk map | `GOVERNANCE_POLICIES.md`'s license/bus-factor table | ✅ | Same review as GOVERN 6.1, viewed from Map's "identify risk" angle — license risk (none) and maintenance/bus-factor risk (pyod somewhat higher than sklearn/TensorFlow) both covered. |
| 4.2 | Internal risk controls scoped to third-party components | `guardian_ai_risk.py` (API-key exposure, shadow-model detection, GPU-spike checks); `GOVERNANCE_POLICIES.md`'s failure-contingency section, which documents real controls (and one honest gap) specifically for third-party library/model failures | ✅ | Upgraded 2026-08-05 — the previous gap was "not scoped specifically to third-party components"; `GOVERNANCE_POLICIES.md` closes exactly that by documenting real third-party failure behavior (TensorFlow/CUDA's graceful degradation, the still-unhandled missing-model-file crash-loop). |
| 5.1 | N/A at this scale | — | ⚪ | Societal impact is genuinely minimal for a personal system. |
| 5.2 | N/A — no external stakeholders | — | ⚪ | |

---

## MEASURE

| # | Guardian Capability | Evidence | Status | Notes |
|---|---|---|---|---|
| 1.1 | Metrics selected for identified risks | `behavioral_policy.py`'s `POLICIES` dict (files/GPU/network/row-count bounds per workflow), tagged `NIST_AI_RMF_TAGS["MEASURE 1.1"]` in the same file | ✅ | |
| 1.2 | Metrics reassessed | `RECENT_ROWS` inconsistency tracking, periodic doc audits; the 2026-09-10/11 `guardian_ai_risk.py` scoring rework, triggered by discovering `ai_risk_score` was structurally pinned below 100 | 🟡 | Happens, but reactively, not on a fixed cadence — this week's rework is a real instance of that same pattern, not a change to it. |
| 1.3 | N/A — solo project | — | ⚪ | No independent reviewer separate from the developer. |
| 2.1 | Test sets/tools documented | `test_behavioral_policy.py`, `test_guardian_health.py`, `test_release_chain.py`, `test_release_report.py` | ✅ | |
| 2.2 | N/A | — | ⚪ | No human-subject evaluation involved. |
| 2.3 | Performance measured under real conditions | Watchdogs score live production data continuously, not a held-out test set | ✅ | |
| 2.4 | Behavior monitored in production | The entire watchdog/Prometheus/Grafana pipeline | ✅ | This is Guardian's core competency. |
| 2.5 | Validity/reliability demonstrated, limits documented | The KNN drift pattern is a real, documented generalizability limit (training snapshot doesn't capture promtail's write-rate variability) | 🟡 | Real evidence of a limit; not a formal validity demonstration. |
| 2.6 | Evaluated for safety; safe failure | `Restart=always`/`on-failure`, explicitly noted in `OPERATIONS_MANUAL.md` as "a crash-loop safety net, not an operations strategy"; `GOVERNANCE_POLICIES.md`'s failure-contingency section, which found one real graceful-degradation case (TensorFlow/CUDA) and one real non-graceful one (missing model file still crash-loops); `ai_risk_collection_ok` + `safe_check()` (2026-09-10), which make a sub-check that raises distinguishable from one that ran and verified clean, instead of both looking like a silent zero | 🟡 | Honest documented gap, not silently omitted — the missing-model-file crash-loop is unchanged, still open, so still 🟡, but the failed-vs-clean distinction is a genuinely new safety mechanism, not just a note. |
| 2.7 | Security/resilience evaluated | This session's `trace_suspect.sh` and `ufw_guard.sh` audits — both confirmed-exploitable gaps, found and closed; the 2026-09-11 MD5→SHA-256 upgrade and systemd-hash content fix; the 2026-09-20 Alertmanager Slack silent-failure discovery (below, MANAGE 4.3) | ✅ | Strong, concrete evidence — now five separate found-and-closed instances, not two. |
| 2.8 | Transparency/accountability risk addressed | `release_record.py` + hash-chaining (`verify_chain.py`) | ✅ | |
| 2.9 | Output explained/interpreted in context | `behavioral_policy.py`'s specific violation messages (e.g., `"row_count 20 below policy minimum 100"`), tagged `NIST_AI_RMF_TAGS["MEASURE 2.9"]` in the same file; `ai_risk_reason{reason=<key>}=<points>` (2026-09-10), which names exactly which factor is deducting how many points from `ai_risk_score` | ✅ | Named explicitly in NIST's own MEASURE 2.9 language — explains *why*, not just *that*. Now demonstrated in a second, independent subsystem. |
| 2.10 | None | — | ⚪ | System handles no third-party personal data today; revisit if `EDGE_ARCHITECTURE.md` ever handles customer data. |
| 2.11 | Deliberately not applicable | — | ⚪ | Guardian's models detect system-metric anomalies (CPU/disk/GPU), not decisions about people — classic demographic-fairness framing doesn't have a clear analog here. Stated explicitly rather than silently skipped. |
| 2.12 | Estimated retrain energy cost | `retrain_common.py`'s `estimate_energy_wh()`, attached to every retrain's OTel span as `energy.estimated_wh` | 🟡 | Real measurement now exists where none did, but it's a documented estimate (CPU% interpolated against this host's published TDP), not a true RAPL measurement — `energy_uj` is root-only on this host (Platypus mitigation) and wasn't judged worth new sudo scope for one metric. |
| 2.13 | TEVV effectiveness informally evaluated | The real defect-demo proving Phase 5 catches an actual injected regression, tagged `NIST_AI_RMF_TAGS["MEASURE 2.13"]` in `behavioral_policy.py` | 🟡 | Proven once, not a repeatable evaluation process. |
| 3.1 | Existing/emergent risks tracked over time | The KNN drift pattern tracked across 4 documented recurrences (2026-07-13, -16, -17, -27) | ✅ | |
| 3.2 | Hard-to-measure risk tracked via proxy signals | The DESKTOP-0AJUKU3 power-quality hypothesis, tracked via disk busy%/event-log proxies — a third real recurrence, 2026-09-14/15, added a specific new data point: `guardian_disk_health.ps1` caught Event ID 153 (I/O device error) in the 24h before the crash, and disk busy% spiked ~0.5%→32% in the last sample before it went unresponsive | 🟡 | Updated 2026-08-05 — a UPS was bought and installed since this was first written, but NUT integration into Guardian was deliberately declined (would have broken PowerPanel's auto-shutdown safety feature), so Guardian itself still only has proxy signals, not direct voltage/power measurement. Status unchanged. Reinforced 2026-09-20 with a third occurrence's real evidence, same proxy-signal limitation. |
| 3.3 | N/A — no external end users | — | ⚪ | |
| 4.1 | N/A — no external domain experts | — | ⚪ | |
| 4.2 | N/A — same | — | ⚪ | |
| 4.3 | N/A — same | — | ⚪ | |

---

## MANAGE

| # | Guardian Capability | Evidence | Status | Notes |
|---|---|---|---|---|
| 1.1 | Go/no-go decisions on AI system changes | Documented decision to hold off on a KNN retrain, betting on a specific cause, later reversed with evidence | ✅ | |
| 1.2 | Risk treatment prioritized by impact/likelihood | `aiops-watchdog-priority.py` — tier + novelty scoring across every risk source | ✅ | Automated, not just manual judgment. |
| 1.3 | Responses to high-priority risks planned/documented | `trace_suspect.sh` and `ufw_guard.sh` fixes — both fully documented, verified in multiple directions before shipping | ✅ | |
| 1.4 | N/A — no downstream acquirers yet | — | ⚪ | Revisit if `EDGE_ARCHITECTURE.md`'s consulting plan gets real customers. |
| 2.1 | Informal resource/tradeoff decisions | Deferring scoped-agent-identity work due to real friction cost | 🟡 | Real judgment calls; not a formal resourcing process. |
| 2.2 | Mechanisms to sustain deployed AI value | The retrain pipeline itself | ✅ | |
| 2.3 | Respond to/recover from previously unknown risk | Both sudoers fixes were exactly this: an unknown gap, discovered, responded to; the 2026-08-04 bad-iForest-retrain (an unknown latent `RECENT_ROWS` bug, surfaced live, recovered from immediately via the freshly-built model archive); the `ai_risk_score` baseline-pinning bug and the Alertmanager Slack silent-failure bug (2026-09-20, see MANAGE 4.3) — both discovered and fixed same-day | ✅ | Textbook real-time example — the recovery mechanism was built the same session it ended up needing to be used. Two more real instances since 8/5. |
| 2.4 | Mechanisms to deactivate underperforming AI | Manual `systemctl stop`/`restart` exists | 🟡 | No automated kill-switch if a model's anomaly rate exceeds a bound — only manual intervention. |
| 3.1 | Third-party risk informally monitored | OS-level pending-updates gauge (`aiops_security_updates_pending`) | 🟡 | Tracks OS packages generally; not AI-library-specific (e.g., no dependency vulnerability scanning for scikit-learn/TensorFlow). |
| 3.2 | Deliberately not applicable | — | ⚪ | Guardian trains its own models from scratch; no pre-trained models in use. |
| 4.1 | Post-deployment monitoring plan | `OPERATIONS_MANUAL.md` Ch. 6 (Operations and Runbook), the alerting pipeline | 🟡 | Strong on monitoring/incident response; "appeal/override" doesn't map cleanly to a personal system. **Sharper gap found 2026-09-20:** the Core host itself suspended for ~65 hours (2026-09-11 22:14 UTC → 2026-09-14 22:04 UTC) — Prometheus, Alertmanager, every watchdog simply weren't running for that entire window, not degraded, off. Nothing could have detected a real incident during it, because the thing that detects incidents was itself asleep, and nothing monitors that. Found by accident, not by any Guardian mechanism. |
| 4.2 | Continual improvement integrated | `ROADMAP.md`'s explicitly-living-document nature; this gap analysis itself | ✅ | |
| 4.3 | Incidents communicated to relevant actors | Alertmanager → Slack, reaching the one operator that exists | 🟡 | **Corrected 2026-09-20, not just reinforced:** for 64% of send attempts (234/367) since Alertmanager started (2026-08-26), this claim was false in practice, silently — `.CommonAnnotations` rendered empty whenever a group held more than one alert with per-instance text (e.g. `WindowsHostUnreachable` across two Windows hosts), Slack rejected the empty message, nothing logged why. This is the actual reason DESKTOP-0AJUKU3's 11-hour outage on 9/15 never reached Slack. Fixed same day the failure was found (template now renders per-alert, verified via a synthetic reproduction of the exact failing shape). Still 🟡 — the mechanism now works, but "existed and looked wired up" and "worked" were different claims for three weeks, and that's worth stating plainly rather than smoothing over now that it's fixed. |

---

## Key findings

- **GOVERN was the honest weak point as of 2026-07-31** (2 of 19 fully satisfied) — still the weakest function after two rounds of real improvement (5/19 after 2026-08-04's gap closure, 7/19 after 2026-08-05's full fresh pass), and most of the rest still assume organizational structure (a workforce, an executive team, a documented supply-chain policy) that a solo project genuinely doesn't have. This isn't a failure to fix; it's what "not applicable at this scale" is for.
- **MEASURE has the strongest, most provable evidence**, even though its raw ✅ count is close to MANAGE's — the KNN defect-catch and the hash-chained release ledger are concrete, demonstrated proof, not just documented intent.
- **MANAGE came out stronger than expected going into this exercise** — `aiops-watchdog-priority.py`'s tier/novelty scoring and the pattern of documented, verified incident responses map onto more of MANAGE's real subcategories than assumed. Zero flat gaps in this function.
- **MAP was the real, actionable opportunity as of 2026-07-31** — and both later passes acted on exactly that: the 2 flat MAP gaps closed to ✅ on 2026-08-04, then a further MAP row (4.2, third-party risk controls) genuinely earned an upgrade on 2026-08-05 — all through documentation of things that were already informally true, no new infrastructure needed, matching the original prediction.
- **No subcategory is marked "Planned."** Everything real here was built to solve an actual problem, not to fill a framework cell — worth stating as a genuine differentiator, not just a coincidence of how this document turned out. Still true after both updates: the 2 items that moved to 🟡 instead of ✅ on 2026-08-04 (GOVERN 6.2, MEASURE 2.12) were deliberately *not* pushed to ✅ by overstating what was actually built, and the 2026-08-05 pass found real new evidence for its 3 upgrades rather than reinterpreting existing evidence more generously.
- **The full fresh pass surfaced one real gap the earlier, narrower passes couldn't have found:** genuine operational learnings from this week (two new ML-drift trigger patterns and their fixes) exist only in session memory, not in any git-tracked doc — see the Summary rollup note for detail. Worth treating as a real follow-up, not a documentation nitpick.
- **Stage 6 (2026-09-20) surfaced the sharpest finding in this document's history:** MANAGE 4.3's evidence — Alertmanager reaching Slack — was factually wrong for three weeks, silently, and the only reason it's known now is that an unrelated 11-hour outage went unnoticed long enough to prompt asking why. Left uncorrected, that's exactly the failure mode this whole framework exists to catch: a control that looks satisfied on paper and isn't, discovered by luck rather than by the system itself. It's now fixed and verified, but the honest record is that it wasn't caught by Guardian noticing — it was caught by a human asking a question the tooling gave no reason to ask.

## What this document does not claim

Not compliance, not certification — NIST AI RMF has neither concept.
The original full 72-subcategory assessment was dated 2026-07-31, against
the codebase as it existed that day. **2026-08-04 update:** the 7
subcategories that were flat Gaps were revisited and closed/partially-closed
(see the Summary rollup note above and `GOVERNANCE_POLICIES.md`) — the
other 65 were not re-checked that day. **2026-08-05 update:** every one of
the 72 subcategories was re-verified against current real evidence, full
rigor, not assumed — the first genuine full re-run since 7/31. Three rows
earned real upgrades; the rest were confirmed accurate or got stronger
supporting evidence without a status change. **2026-09-20 update (Stage
6):** a targeted pass, not a full 72-row re-run — nine rows across GOVERN,
MEASURE, and MANAGE touched with real evidence from the six weeks since
8/5, no status changes, one row (MANAGE 4.3) corrected rather than merely
reinforced. This document is current as of 2026-09-20 for the rows it
touched; the remaining rows were last verified 2026-08-05 and are assumed,
not re-confirmed, this time. The whole document should still be re-run
periodically as Guardian keeps changing, the same way the doc-vs-reality
audits already are — being current today doesn't mean staying current
without another pass later.
