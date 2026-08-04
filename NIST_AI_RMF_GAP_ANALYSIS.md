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
| GOVERN | 5 | 8 | 0 | 0 | 6 | 19 |
| MAP | 8 | 7 | 0 | 0 | 3 | 18 |
| MEASURE | 8 | 7 | 0 | 0 | 7 | 22 |
| MANAGE | 6 | 5 | 0 | 0 | 2 | 13 |
| **Total** | **27** | **27** | **0** | **0** | **18** | **72** |

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
| 1.4 | Transparent, public documentation | `VISION.md`, `ROADMAP.md` — tracked, public repo | 🟡 | Genuinely transparent, but this is a stated philosophy, not a formal "process." |
| 1.5 | Periodic doc-vs-reality audits | 2026-07-27 and 2026-07-29 audits, documented in `OPERATIONS_MANUAL.md` Ch. 10 | 🟡 | Real, repeatable practice; no fixed schedule or defined roles (solo project). |
| 1.6 | Informal AI-system discovery | `guardian_ai_risk.py`'s `AI_PROCESSES_RUNNING`/`AI_TOOLS` checks | 🟡 | Detects AI activity automatically; not a governed, resourced inventory. |
| 1.7 | Automated model archiving | `retrain_common.py`'s `archive_current_models()`, wired into all 3 retrain scripts; `GOVERNANCE_POLICIES.md` | ✅ | Real mechanism, not just a written policy — validated same day it was built, when it made recovery trivial after a real bad-retrain (see `GOVERNANCE_POLICIES.md`). |
| 2.1 | N/A — solo operator | — | ⚪ | One person; no roles to document. |
| 2.2 | N/A — no other personnel | — | ⚪ | |
| 2.3 | Operator approves AI-driven changes interactively | Every session's approval pattern; `aiops-approval.service` | 🟡 | Real but informal — one person acting as both "leadership" and operator. |
| 3.1 | N/A — no team | — | ⚪ | |
| 3.2 | Human-oversight pattern for AI actions | `aiops-approval.service`; this session's own "ask before risky actions" practice | 🟡 | Real mechanism exists; not written down as policy. |
| 4.1 | Safety-first design philosophy documented | `VISION.md`'s "deliberately narrow scope" section | 🟡 | Real, but a design doc, not an organizational policy. |
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
| 2.3 | TEVV practiced | pytest suite, Phase 5 behavioral verification, the real defect-demo | ✅ | |
| 3.1 | Benefits documented | `VISION.md`'s flagship use cases | ✅ | |
| 3.2 | Costs of AI errors documented | KNN drift's alert-fatigue cost, documented in `OPERATIONS_MANUAL.md` Ch. 10 | 🟡 | |
| 3.3 | Application scope specified | `VISION.md`'s "deliberately narrow scope" section, verbatim | ✅ | This is the clearest, most direct match in the whole document. |
| 3.4 | N/A — solo operator | — | ⚪ | No formal proficiency/certification process for one person. |
| 3.5 | Human oversight process | `aiops-approval.service`; Phase 7 item 1 (scoped agent identity) is literally this subcategory's next step | 🟡 | Real mechanism; not comprehensively documented as a process. |
| 4.1 | Third-party library risk map | `GOVERNANCE_POLICIES.md`'s license/bus-factor table | ✅ | Same review as GOVERN 6.1, viewed from Map's "identify risk" angle — license risk (none) and maintenance/bus-factor risk (pyod somewhat higher than sklearn/TensorFlow) both covered. |
| 4.2 | Informal internal risk controls | `guardian_ai_risk.py` (API-key exposure, shadow-model detection, GPU-spike checks) | 🟡 | Real controls exist; not scoped specifically to third-party components. |
| 5.1 | N/A at this scale | — | ⚪ | Societal impact is genuinely minimal for a personal system. |
| 5.2 | N/A — no external stakeholders | — | ⚪ | |

---

## MEASURE

| # | Guardian Capability | Evidence | Status | Notes |
|---|---|---|---|---|
| 1.1 | Metrics selected for identified risks | `behavioral_policy.py`'s `POLICIES` dict (files/GPU/network/row-count bounds per workflow) | ✅ | |
| 1.2 | Metrics reassessed | `RECENT_ROWS` inconsistency tracking, periodic doc audits | 🟡 | Happens, but reactively, not on a fixed cadence. |
| 1.3 | N/A — solo project | — | ⚪ | No independent reviewer separate from the developer. |
| 2.1 | Test sets/tools documented | `test_behavioral_policy.py`, `test_guardian_health.py`, `test_release_chain.py`, `test_release_report.py` | ✅ | |
| 2.2 | N/A | — | ⚪ | No human-subject evaluation involved. |
| 2.3 | Performance measured under real conditions | Watchdogs score live production data continuously, not a held-out test set | ✅ | |
| 2.4 | Behavior monitored in production | The entire watchdog/Prometheus/Grafana pipeline | ✅ | This is Guardian's core competency. |
| 2.5 | Validity/reliability demonstrated, limits documented | The KNN drift pattern is a real, documented generalizability limit (training snapshot doesn't capture promtail's write-rate variability) | 🟡 | Real evidence of a limit; not a formal validity demonstration. |
| 2.6 | Evaluated for safety; safe failure | `Restart=always`/`on-failure`, explicitly noted in `OPERATIONS_MANUAL.md` as "a crash-loop safety net, not an operations strategy" | 🟡 | Honest documented gap, not silently omitted. |
| 2.7 | Security/resilience evaluated | This session's `trace_suspect.sh` and `ufw_guard.sh` audits — both confirmed-exploitable gaps, found and closed | ✅ | Strong, concrete evidence. |
| 2.8 | Transparency/accountability risk addressed | `release_record.py` + hash-chaining (`verify_chain.py`) | ✅ | |
| 2.9 | Output explained/interpreted in context | `behavioral_policy.py`'s specific violation messages (e.g., `"row_count 20 below policy minimum 100"`) | ✅ | Named explicitly in NIST's own MEASURE 2.9 language — explains *why*, not just *that*. |
| 2.10 | None | — | ⚪ | System handles no third-party personal data today; revisit if `EDGE_ARCHITECTURE.md` ever handles customer data. |
| 2.11 | Deliberately not applicable | — | ⚪ | Guardian's models detect system-metric anomalies (CPU/disk/GPU), not decisions about people — classic demographic-fairness framing doesn't have a clear analog here. Stated explicitly rather than silently skipped. |
| 2.12 | Estimated retrain energy cost | `retrain_common.py`'s `estimate_energy_wh()`, attached to every retrain's OTel span as `energy.estimated_wh` | 🟡 | Real measurement now exists where none did, but it's a documented estimate (CPU% interpolated against this host's published TDP), not a true RAPL measurement — `energy_uj` is root-only on this host (Platypus mitigation) and wasn't judged worth new sudo scope for one metric. |
| 2.13 | TEVV effectiveness informally evaluated | The real defect-demo proving Phase 5 catches an actual injected regression | 🟡 | Proven once, not a repeatable evaluation process. |
| 3.1 | Existing/emergent risks tracked over time | The KNN drift pattern tracked across 4 documented recurrences (2026-07-13, -16, -17, -27) | ✅ | |
| 3.2 | Hard-to-measure risk tracked via proxy signals | The DESKTOP-0AJUKU3 power-quality hypothesis, tracked via disk busy%/event-log proxies since a direct measurement (a UPS) doesn't exist yet | 🟡 | |
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
| 2.3 | Respond to/recover from previously unknown risk | Both sudoers fixes were exactly this: an unknown gap, discovered, responded to | ✅ | |
| 2.4 | Mechanisms to deactivate underperforming AI | Manual `systemctl stop`/`restart` exists | 🟡 | No automated kill-switch if a model's anomaly rate exceeds a bound — only manual intervention. |
| 3.1 | Third-party risk informally monitored | OS-level pending-updates gauge (`aiops_security_updates_pending`) | 🟡 | Tracks OS packages generally; not AI-library-specific (e.g., no dependency vulnerability scanning for scikit-learn/TensorFlow). |
| 3.2 | Deliberately not applicable | — | ⚪ | Guardian trains its own models from scratch; no pre-trained models in use. |
| 4.1 | Post-deployment monitoring plan | `OPERATIONS_MANUAL.md` Ch. 6 (Operations and Runbook), the alerting pipeline | 🟡 | Strong on monitoring/incident response; "appeal/override" doesn't map cleanly to a personal system. |
| 4.2 | Continual improvement integrated | `ROADMAP.md`'s explicitly-living-document nature; this gap analysis itself | ✅ | |
| 4.3 | Incidents communicated to relevant actors | Alertmanager → Slack, reaching the one operator that exists | 🟡 | Works for the actors that exist; no "affected communities" yet. |

---

## Key findings

- **GOVERN was the honest weak point as of 2026-07-31** (2 of 19 fully satisfied) — still the weakest function after the 2026-08-04 gap-closure pass (now 5/19), and most of the rest still assume organizational structure (a workforce, an executive team, a documented supply-chain policy) that a solo project genuinely doesn't have. This isn't a failure to fix; it's what "not applicable at this scale" is for.
- **MEASURE has the strongest, most provable evidence**, even though its raw ✅ count is close to MANAGE's — the KNN defect-catch and the hash-chained release ledger are concrete, demonstrated proof, not just documented intent.
- **MANAGE came out stronger than expected going into this exercise** — `aiops-watchdog-priority.py`'s tier/novelty scoring and the pattern of documented, verified incident responses map onto more of MANAGE's real subcategories than assumed. Zero flat gaps in this function.
- **MAP was the real, actionable opportunity as of 2026-07-31** — and the 2026-08-04 pass acted on exactly that: the 2 flat MAP gaps (risk tolerance, third-party library risk mapping) both closed to ✅ through documentation of things that were already informally true, no new infrastructure needed, matching the original prediction.
- **No subcategory is marked "Planned."** Everything real here was built to solve an actual problem, not to fill a framework cell — worth stating as a genuine differentiator, not just a coincidence of how this document turned out. Still true after the 2026-08-04 update: the 2 items that moved to 🟡 instead of ✅ (GOVERN 6.2, MEASURE 2.12) were deliberately *not* pushed to ✅ by overstating what was actually built.

## What this document does not claim

Not compliance, not certification — NIST AI RMF has neither concept.
The full 72-subcategory assessment is a self-assessment against a
voluntary framework, dated 2026-07-31, against the codebase as it existed
that day. **2026-08-04 update:** the 7 subcategories that were flat Gaps
were revisited and closed/partially-closed (see the Summary rollup note
above and `GOVERNANCE_POLICIES.md`) — the other 65 subcategories were not
re-checked on that date and may have drifted since 2026-07-31. The whole
document should be re-run periodically as Guardian changes, the same way
the doc-vs-reality audits already are.
