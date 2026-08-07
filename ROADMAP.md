# Guardian: Behavioral Attestation Roadmap

Phased plan for building the Behavioral Attestation engine described in
[`VISION.md`](VISION.md). Unlike the vision doc, this file is expected to change —
reorder phases, cut scope, swap tools — as reality dictates. Each phase should
leave the system in a working, useful state on its own; none of this requires a
big-bang rewrite.

## Two flagship demos this roadmap serves

Scope is deliberately narrow: **performance attribution for AI workloads on
Linux**, not performance attribution for everything. Everything below builds
toward two concrete demos, in this order:

1. **Performance attribution demo** — inject four distinct problems into the
   autoencoder workload (inefficient Python code, excessive system calls, slow
   storage, GPU starvation) and have Guardian correctly identify each, with
   evidence. Served primarily by Phases 1–2 and 4.
2. **AI-managed release traceability demo** — have an AI coding agent (e.g.
   Claude Code) manage a controlled release of the autoencoder end-to-end,
   with Guardian recording objective → approved commands → files/deps changed →
   commit/artifact → deployment → runtime behavior before/after. Then inject a
   release defect (e.g. excessive worker concurrency) and confirm Guardian
   traces the regression to that exact change. Served primarily by Phases 3, 5,
   and 6.

Treat each phase below as scoped to *these two demos first* — generalize beyond
AI workloads only after both demos work end-to-end.

## Phase 0 — Foundation (done)

What already exists and everything below builds on:
- Telemetry collection (`aiops-watchdog-ml.py` → `metrics.csv`, system-level metrics
  every ~5s: disk, cpu, mem, net, disk I/O, GPU util/mem/temp, inode)
- Three anomaly watchdogs (autoencoder, KNN, isolation forest) exposing Prometheus
  metrics
- Guardian health service: health score, security posture score, AI-risk score
- Prometheus + Grafana for storage/visualization

Gap: everything above answers *"is something wrong,"* never *"what caused it."*
That gap is what the rest of this roadmap closes.

---

## Phase 1 — Process-level attribution

**Goal:** when an anomaly fires, know *which process(es)* were responsible —
the first slice of "evidence-based performance attribution."

- Sample per-process CPU/mem/IO (e.g. via `psutil`) alongside existing system
  metrics, at anomaly-detection resolution.
- On anomaly trigger, snapshot the top-N processes by resource usage at that
  timestamp and attach it to the anomaly record.
- Surface "top suspect process" in Grafana next to the anomaly panel.

No new infrastructure — extends the existing collector and watchdog services.

**Feeds into the performance-attribution demo:** this phase alone should already
be able to distinguish "inefficient Python code" (high CPU in the training
process itself) from "excessive system calls" (high CPU with unusually high
syscall-adjacent activity, refined further in Phase 2) among the four injected
demo problems.

---

## Phase 2 — eBPF / system-activity correlation

**Goal:** attribution below the process level — syscalls, file access, network
connections — for cases where "which process" isn't enough detail.

- Introduce an eBPF tracing layer (candidates: `bpftrace` for quick instrumentation,
  or a packaged framework like Cilium Tetragon / Parca for less hand-rolled
  maintenance — pick based on what's actively maintained when you get here).
- Capture syscall/network/file events scoped to the processes Phase 1 already
  flags as suspects, not system-wide (keeps overhead and noise down).
- Correlate eBPF events into the same anomaly timeline as Phase 1.

This is the heaviest lift in the roadmap — budget the most exploration time here,
and treat the specific tool choice as disposable.

---

## Phase 3 — AI workflow traceability

**Goal:** follow one AI workflow run (e.g. a retrain, a watchdog inference cycle,
an agent task) end-to-end through the system, not just as isolated log lines.

- Instrument AI-related processes (retrain scripts, watchdog inference loops) with
  a trace ID per run (OpenTelemetry is the obvious default given how much tooling
  assumes it, but re-evaluate if something better fits by the time you build this).
- Emit spans for each meaningful step (data load → train/inference → model save →
  service reload) tagged with that trace ID.
- Store/query traces so a single workflow run's full path is retrievable, and link
  it to the process/eBPF evidence from Phases 1–2 that occurred during that run.

---

## Phase 4 — GPU activity correlation

**Goal:** extend attribution and traceability to GPU-bound work specifically,
since GPU abuse/misuse is already a named risk category in the AI-risk scoring.

- Move from system-wide GPU util/mem/temp (current) to per-process GPU accounting
  (e.g. `nvidia-smi --query-compute-apps` or DCGM if available).
- Tie GPU activity to the same trace IDs from Phase 3, so a workflow run's GPU
  footprint is part of its trace, not a separate signal.

---

## Phase 5 — Runtime behavioral verification

**Goal:** move from "attribute what happened" to "verify it matched what was
expected" — the "attestation" half of Behavioral Attestation.

- Define expected-behavior baselines/policies per workflow (e.g. "a retrain run
  should touch these files, use this much GPU, take this long, and not open
  outbound network connections").
- Compare each traced run (Phase 3) against its policy; flag deviations as
  verification failures, distinct from statistical anomalies.
- This is the first phase where the engine asserts something *should* be true,
  rather than only observing what *is*.

---

## Phase 6 — AI-managed release traceability

**Goal:** the capstone — trace a change from code, through deployment, to its
observed runtime behavior, for releases that are themselves AI-managed. This is
where Guardian combines two kinds of provenance that normally stop at different
boundaries: **build provenance** (SLSA-style — agent identity, task/prompt,
files/deps changed, commit, artifact, signing) and **runtime behavioral
attestation** (Guardian's extension — what actually executed, resource/behavior
changes, whether execution matched the approved plan, attributed root cause).

- Tie git commits/releases to deployment events (service restarts, model
  retrains — the kind of event already visible in systemd journal logs).
- Record the AI agent's stated objective and the specific commands it requested
  and had approved (ties into the Governance Engine's Human Approval piece,
  `aiops-approval.service`).
- Attach the pre-/post-deployment behavioral fingerprint (anomaly scores, traces,
  verification results from Phases 1–5) to each release.
- End state: for any release, answer "what changed in the code, and what changed
  in observed runtime behavior as a result" — with evidence, not inference.

**Demo target:** let Claude Code (or another AI coding agent) manage a real,
controlled release of the autoencoder — recording objective, approved commands,
changed files/config, commit/artifact hashes, deployment events, and full
resource/behavior telemetry before and after. Then deliberately introduce a
release defect (e.g. bump worker concurrency too high, or repeatedly reload the
dataset) and confirm Guardian's trace correctly attributes the resulting
regression to that specific change — see the illustrative "Release 2026.07.16.4"
scenario in `VISION.md` for the shape of the expected output.

Note: the KNN retrain done manually on 2026-07-13 (edit script → retrain → restart
→ verify scores normalized) is a small working example of exactly the before/after
pattern this phase should eventually automate and record — it just wasn't
captured as structured evidence at the time.

---

## Phase 7 — Govern the agent, not just the output (proposed 2026-07-20)

**Goal:** close the gap named in `VISION.md`'s "sharper framing" note — Phase 6
proves what an AI-driven change *did*; this phase would prove the AI agent
making changes was *governed while doing it*, which is the distinction a real
audit actually cares about (see the framing that prompted this: an ungoverned
agent holding privileged access isn't automation, it's "an incident with a
countdown").

Concrete, currently-missing pieces, in rough priority order:

1. **A scoped identity for the agent, not the developer's own session.** Right
   now Claude Code (or any AI agent) operates with the full, unscoped `beth`
   user session — same sudo, same git push access, same filesystem reach as
   the human. A real least-privilege model would give the agent its own
   narrower identity (e.g. a separate, tightly-scoped sudoers profile or a
   dedicated service account) with only the access a given task actually
   needs, not blanket inheritance of the human's session.
2. **Govern the tooling engineers hand the agent, not just the agent's
   identity** (clarified 2026-07-27 — this is the actual motivating idea
   behind `~/aiops-guardrail-lab`, item 2 below). Item 1 above is about the
   agent's own privilege boundary; this is a distinct failure mode: **a
   systems engineer can cause an AI agent to do things it shouldn't, by
   accident**, simply by how they build the scripts/wrappers/sudoers entries
   the agent is expected to run — an overly broad shell wrapper, a
   too-permissive sudoers rule, a script that doesn't validate its input, all
   hand an agent more capability than the task in front of it needs, with no
   bad intent required on either the engineer's or the agent's part.

   **First concrete instance found and closed (2026-07-29):**
   `trace_suspect.sh`'s passwordless sudo rule accepted an unrestricted
   wildcard PID argument — exactly the kind of engineer-authored tooling gap
   this item is about, not an agent misbehaving. Confirmed exploitable, not
   just theoretical: `beth` could run `sudo -n trace_suspect.sh 1` and get a
   real root-level eBPF trace of PID 1 (`systemd`), with zero connection to
   Guardian's own detection logic. PID-ownership restriction was considered
   and rejected — `promtail`, this feature's real validating use case, runs
   as root, so "only trace beth-owned PIDs" would have broken it. Fixed
   instead with a ticket mechanism: `ebpf_trace.py` now writes the intended
   PID + timestamp to a short-lived `.trace_ticket` file immediately before
   calling `sudo`, and `trace_suspect.sh` refuses to run unless a fresh,
   matching ticket exists — verified in all four directions (no ticket,
   wrong-PID ticket, stale ticket, valid ticket). Full detail in
   `OPERATIONS_MANUAL.md` Chapter 5.2.

   **Second concrete instance found and closed (2026-07-30), from doing
   exactly that audit:** the other passwordless sudoers entries were
   `systemctl restart prometheus.service`/`loki.service` (fully literal, no
   wildcard, nothing to fix) and `ufw` — which turned out to have **no
   subcommand restriction at all** (`(ALL) NOPASSWD: /usr/sbin/ufw`).
   Confirmed as a real, already-used capability rather than theoretical: this
   session had already used the unrestricted entry twice to add/remove real
   firewall rules, meaning `ufw disable` or `ufw --force reset` could equally
   have been run passwordlessly by anything running as `beth`, removing the
   DENY rules several watchdog ports rely on as their only external defense
   (`OPERATIONS_MANUAL.md` Chapter 3.9/7.3). Narrowing to read-only `status`
   was rejected for the same reason PID-ownership restriction was rejected
   for `trace_suspect.sh` — it would break a real, already-established
   workflow. Fixed with `ufw_guard.sh`, a wrapper that blocks
   `disable`/`reset`/`default` (the only subcommands with zero legitimate use
   here) and passes everything else through; verified in all three block
   cases plus the `status` passthrough, both standalone and through real
   `sudo`. Full detail, including a live-service consequence hit the same
   day, in `OPERATIONS_MANUAL.md` Chapter 5.2.

   Concretely, going forward: continue auditing every new script/wrapper an
   AI agent is expected to invoke (not just the agent's own sudo profile) for
   the same class of gap — over-broad arguments, unscoped file access,
   missing input validation — as an ongoing governance checklist, not a
   one-time fix.
3. **Map `behavioral_policy.py`'s `POLICIES` to a real framework** (ISO 42001
   and/or the NIST AI RMF are the two named in the framing that prompted this)
   instead of the current ad-hoc dict. This is what turns "we check some
   things" into something an auditor would recognize as governance.
   **Related existing work, found 2026-07-27 while auditing docs vs.
   reality:** `~/aiops-guardrail-lab/scripts/guardrail_exporter.py` (untracked,
   outside this repo, running as `guardrail-exporter.service` since 2026-07-19)
   already counts ALLOW/BLOCK/APPROVED/DENIED/INVALID actions and risk tiers
   from a guardrail-decision log — see `VISION.md`'s Governance Engine note.
   It's a real v1, but built independently of `behavioral_policy.py` against
   simulated data, not integrated with it. Merging the two into one coherent
   Guardrails component (one action-classification/logging model, one set of
   Prometheus metrics, ideally framework-mapped per this item) is the concrete
   next step here, not building Guardrails from scratch.

   **Decision (2026-07-31): NIST AI RMF chosen over ISO/IEC 42001 as the
   framework to map against.** Reasoning: NIST AI RMF is free and complete
   (ISO 42001's actual text is paywalled), it's a voluntary self-assessment
   framework with no certification expectation attached (ISO 42001 is a
   certifiable management-system standard — referencing it without being
   certified invites an awkward "so are you certified?" question that has
   no good answer for a solo project), its MEASURE function maps naturally
   onto what Guardian's watchdogs/`behavioral_policy.py` already do, and
   it's the framework actually getting referenced in the AI-safety/security
   community right now. ISO 42001 is still worth a passing mention as a
   secondary nod, not the primary mapping target.

   **Stage 1 (grounding) done, 2026-07-31:** `NIST_AI_RMF_REFERENCE.md`
   captures the actual AI RMF 1.0 Core verbatim (source:
   `NIST.AI.100-1.pdf`, Jan 2023) — all four functions (Govern, Map,
   Measure, Manage), every category and subcategory, plus the seven named
   trustworthiness characteristics. This is reference material only, no
   claims yet about Guardian's own coverage.

   **Stage 2 (gap analysis) done, 2026-07-31:** `NIST_AI_RMF_GAP_ANALYSIS.md`
   — every one of the 72 subcategories across all four functions assessed
   against real evidence (specific files, metrics, services, documented
   incidents), each tagged Satisfied/Partial/Planned/Gap/Not-applicable.
   Headline results: 22 Satisfied, 25 Partial, 0 Planned, 7 Gap, 18 Not
   applicable. GOVERN is honestly the weak function (2/19 Satisfied — most
   of it assumes organizational structure a solo project doesn't have).
   MEASURE has the strongest, most provable evidence (the real KNN
   defect-catch, the hash-chained ledger). MANAGE came out stronger than
   expected (`aiops-watchdog-priority.py`'s tier/novelty scoring covers
   more of it than assumed going in — zero flat gaps in that function).
   MAP is the real, actionable next opportunity — mostly writing down
   context/cost/third-party-risk information that's already informally
   true, not new infrastructure. Adopted a traceability-matrix structure
   (subcategory → Guardian capability → evidence → status) plus a
   three-layer executive/architect/engineer view, both suggested via a
   ChatGPT consultation and verified against the real framework text and
   real code before being trusted — a couple of ChatGPT's example claims
   (an overstated "Govern is meaningfully supported" read, two invented
   capability names) didn't hold up and were corrected rather than
   carried through.

   **Stage 3 (gap closure) done, 2026-08-04:** all 7 flat Gaps addressed —
   5 moved to Satisfied (a risk-tolerance statement, automated model
   archiving in `retrain_common.py` validated same-day by catching a real
   latent `RECENT_ROWS` bug in `retrain_recent_iforest.py`, a verified
   third-party license review), 2 moved to Partial rather than overclaimed
   as fully closed (`GOVERN 6.2`'s model-load-failure handling is real for
   one failure mode and honestly absent for another; `MEASURE 2.12`'s
   retrain-energy figure is a documented CPU%-based estimate, not a true
   RAPL measurement — `energy_uj` is root-only on this host and wasn't
   judged worth new sudo scope for one metric). New `GOVERNANCE_POLICIES.md`
   holds the actual written policies. Updated rollup: 27 Satisfied, 27
   Partial, 0 Planned, 0 Gap, 18 Not applicable.

   **Stage 4 (framework-tag metadata) done, 2026-08-05:** added
   `NIST_AI_RMF_TAGS` directly to `behavioral_policy.py`, tagging the four
   subcategories (`MEASURE 1.1`, `MAP 2.3`, `MEASURE 2.9`, `MEASURE 2.13`)
   already citing that file in the gap analysis doc — so those claims
   trace to real code, not just a separate write-up. Deliberately didn't
   invent new tags beyond what was already asserted elsewhere.

   **Stage 5 (full fresh re-verification) done, 2026-08-05:** re-checked all
   72 subcategories against current real evidence rather than assuming the
   7/31 baseline still held. Three genuine upgrades to Satisfied (`GOVERN
   1.4`, `GOVERN 4.1`, `MAP 4.2`), each tied to a specific new artifact
   (mainly `GOVERNANCE_POLICIES.md`, not general drift). New rollup: 30
   Satisfied, 22 Partial, 0 Planned, 0 Gap, 20 Not applicable — verified
   via a script count against the actual document, not just asserted,
   after catching a real arithmetic error in the first draft of that
   rollup table. Also surfaced a real follow-up: this week's operational
   incidents (reboot- and suspend/resume-triggered ML drift, the
   autoencoder threshold recalibration) exist only in session memory,
   never written into any repo doc — not fixed as part of this pass, worth
   doing separately.

   **Phase 7 item 3 is now fully complete, genuinely current as of
   2026-08-05**, not just 7/31 with a patched subset. ISO 42001 remains a
   secondary mention only, not mapped in the same depth as NIST, by
   deliberate choice. The document should still be re-run periodically as
   Guardian keeps changing — being current today isn't permanent.
4. **Genuinely tamper-evident logging**, not just observable logging. Today's
   evidence trail (systemd journal, OTel/Tempo traces, `release_record.py`
   JSON files) is good observability but nothing is cryptographically signed
   or append-only — a privileged actor (including an AI agent with broad
   sudo) could alter it after the fact.

   **First slice closed (2026-07-30):** `release_record.py`'s `releases/*.json`
   ledger is now hash-chained — each record carries a `chain` block
   (`sequence`, `previous_hash`, `record_hash`); `previous_hash` points at the
   prior record's `record_hash`, so editing any past record after the fact
   changes a hash that every later record's chain depends on. `verify_chain.py`
   walks the whole ledger and confirms it's unbroken; `test_release_chain.py`
   proves this concretely (`test_edited_record_breaks_chain`), not just in
   theory. The two real pre-existing release records were backfilled into the
   chain (order preserved via their original timestamps) rather than starting
   the ledger mid-history. **Scope note:** this only covers the release
   ledger — systemd journal entries and OTel/Tempo traces are still plain
   observable logging, not hash-chained or signed. Extending the same pattern
   to those (or reaching for something heavier, e.g. actual signing) is the
   remaining part of this item if it's worth pursuing further.

   **Follow-on (2026-07-30):** `release_report.py` renders a single release
   record as a self-contained Markdown evidence document — objective, build
   provenance, runtime verification result, and a chain-of-custody section
   stating whether the ledger re-verified intact at report-generation time —
   so the chain's proof is something to actually hand a non-technical reader
   (auditor, lawyer, hiring manager), not just something `verify_chain.py`
   prints to a terminal.

This phase is deliberately not scoped further than that yet — it's a real gap
worth tracking, not a fully-designed plan. Revisit and flesh out phase-by-phase
like the rest of this roadmap once there's appetite to build it.

---

## Sequencing notes

- Phases are ordered by dependency, not necessarily by priority — Phase 1 unlocks
  the most value for the least effort and should come first regardless.
- Phases 2–4 (eBPF, tracing, GPU) can be reordered or run in parallel; they don't
  depend on each other, only on Phase 1's anomaly-timeline concept existing.
- Phases 5–6 both depend on having real trace data (Phase 3) to verify/attach to,
  so they come last regardless of how the middle phases are resequenced.

---

## Note: Agent Behavioral Attribution (not a phase, tracked here for later, added 2026-08-06)

**Goal:** deepen Phase 1's Process Attribution from "which process caused
this" to "which process, and did it descend from a specific AI agent
session" — with an explicit, honest distinction between confirmed lineage
and mere time-correlation. Example of the target output:

- `"Process 18432 modified this file; it descended from this Claude session."`
- `"The file changed while Claude was active, but Guardian cannot establish causation."`

**Not a new engine** — this is a sharpening of the existing Process
Attribution sub-item under Behavioral Attestation, not a sixth top-level
category in `VISION.md`'s five-engine architecture.

**The opportunity:** Claude Code (or any AI coding agent) runs on the same
machine Guardian already monitors — a real, live subject for developing
this against actual agent activity, not manufactured test cases. Matches
the pattern the rest of the attribution work has followed (Phase 1/2's
real findings — promtail, the dpkg/apt-timer spike — came from live
activity, not staged scenarios).

**What already exists to build on:** `ebpf_trace.py`/`trace_suspect.sh`
already does PID-scoped syscall tracing (`openat`/`execve`/`connect`/
`write`) via the sudoers-gated `bpftrace` wrapper — the mechanism for
catching "PID X wrote this file" is already there. `process_attribution.py`
already does top-N process snapshotting. **What's missing:** the ancestry
walk (trace a PID's parent chain back to confirm — or fail to confirm —
it descends from a known AI agent session), and the two-branch honest
output (confirmed vs. correlated-only) instead of a single flat
attribution claim.

**Status: backlog idea, not scoped or started.** No design decisions made
yet on how to identify "this process tree belongs to a Claude Code
session" at the OS level, or how deep the ancestry walk should go. Pick
this up when the user is ready to actually design/build it — don't
assume prior context beyond what's written here.

**Sequencing (user's explicit call, 2026-08-06): finish `diagnose_anomaly.py`'s
wire-the-trigger/approval-gate extension (see the note at the end of the
Phase 7 section, or `project-aiops-autoencoder` memory) before starting
this.** Not a hard technical dependency, just the user's stated priority
order — don't start Agent Behavioral Attribution work ahead of it.

---

## Note: Kubernetes / horizontal scaling (not a phase, tracked here for later)

microk8s was installed to test whether these scripts could run efficiently at
enterprise scale, not just on one machine — worth being clear-eyed about what
the current deployment does and doesn't demonstrate:

- It's single-node (the same box is both the k8s node and the host), so it
  proves the scripts can be containerized/orchestrated. It does **not** exercise
  what "enterprise scale" actually stresses: multi-node scheduling, network
  policy across machines, resource contention under real distributed load. That
  needs more than one node to observe at all.
- Bigger blocker before multi-node is worth trying: the watchdogs assume
  local-filesystem, singleton state — `metrics.csv`, the `.pkl`/`.keras` model
  files, appending writes. Scaling `aiops-watchdog-knn` to multiple replicas
  today would have them fighting over the same local files, not actually
  scaling horizontally. Solving that (shared/networked storage, or making each
  replica stateless with a centralized model) is the real prerequisite —
  multi-node won't tell you anything new until it's addressed.
- Current state: one k8s-deployed replica of the KNN watchdog runs in parallel
  with the systemd-managed instance, independently, both pointed at the same
  local files — a reasonable "learned the orchestration primitives" milestone,
  not yet a scale test.

**Path to an actual enterprise-scale test, in order:**

1. **Make the watchdogs stateless first** (do this before touching multi-node —
   otherwise multi-node just reproduces the same file-contention problem across
   more machines). Move `metrics.csv` writes off local disk (a real DB, or push
   straight to Prometheus remote-write). Move model artifacts (`.pkl`/`.keras`)
   to shared storage (NFS/object storage/a small model registry) that replicas
   *read*, not *own*. This alone is testable and demonstrable on the current
   single node.
2. **Get real node boundaries, cheaply.** microk8s supports joining multiple
   nodes into one cluster (`microk8s add-node`) — a few low-cost cloud VMs
   joined as real nodes is the most realistic option (actual network/resource
   isolation). `kind` with multiple worker containers is a free way to test
   scheduling logic first, but it simulates nodes on one kernel and isn't a
   substitute for real ones.
3. **Test things that actually demonstrate "enterprise-ready," not just "runs
   on more boxes":** scale a stateless watchdog to N replicas under load and
   confirm even distribution; kill a node and confirm pods reschedule
   automatically (cheap, fast, great demo); add a `NetworkPolicy` and confirm
   namespace isolation actually holds.

---

## Note: operational-readiness assessment (skeptical IT Ops manager lens, 2026-07-20)

Bottom line: this would not pass a production readiness review today. The
*design* (Behavioral Attestation's "can I prove what actually happened"
question) is more mature than most production systems get reviewed for — the
gap is that operational hygiene hasn't caught up to the ambition yet. Gaps,
grouped so they're actionable later:

**Reliability**
- No automated test suite at all. 60 `ruff` errors sat in the codebase
  undetected until an explicit lint pass (2026-07-19), including a real logic
  bug (`_ufw_denies_port_externally` missing IPv6-only DENY rules).
- `retrain_recent.py` / `retrain_recent_iforest.py` still have
  `RECENT_ROWS=100000` vs. `retrain_recent_knn.py`'s `2000` — an unnoticed
  cross-script inconsistency, i.e. nothing currently checks sibling scripts
  stay consistent.
- No staging environment — this box is prod, dev, and workstation at once;
  every change is tested live.

**Security**
- Watchdog ports bind to `0.0.0.0` (not `127.0.0.1` as code comments claim);
  `ufw`'s DENY rule is the *only* thing preventing external exposure — single
  point of failure, not defense-in-depth.
- **Closed (2026-07-29):** passwordless sudo for `trace_suspect.sh` used to
  accept an unrestricted wildcard PID argument — confirmed exploitable as a
  real root-level eBPF trace of PID 1, unrelated to Guardian's own detection
  logic. Fixed with a ticket mechanism (`ebpf_trace.py` writes the intended
  PID to a short-lived file immediately before calling `sudo`;
  `trace_suspect.sh` refuses without a fresh, matching ticket) — see Phase 7
  item 2 above for full detail.
- No secrets manager; config lives in plaintext systemd env drop-ins.

**Operational supportability**
- A 156-day-old k8s deployment wasn't identifiable as "known/intentional" vs.
  "stray leftover" without active investigation, even with full session
  context — an on-call engineer with less context would have no chance.
- No runbooks, no documented rollback procedure. `Restart=always` is a crash
  loop with a delay, not an operations strategy.
- **Correction (2026-07-20):** alerting itself does exist —
  `/etc/prometheus/rules/aiops-alerts.yml` defines sustained-anomaly alerts
  for all three watchdogs and they do fire (confirmed `KNNWatchdogSustainedAnomaly`
  actively firing on 2026-07-20, matching the recurring pattern from the
  2026-07-13 report). The real gap is narrower than "no alerting": Alertmanager
  isn't running, so a firing alert never reaches a human — it just sits in
  Prometheus's own UI. Wiring up Alertmanager (even just email/Slack) is a
  small, high-leverage fix relative to how much of the alerting groundwork
  already exists.
- **Closed (2026-07-20):** Alertmanager is now installed (`/usr/local/bin/alertmanager`,
  official tarball release v0.33.1, same install pattern as the existing
  Prometheus binary), running as `alertmanager.service` (enabled on boot),
  routing to a Slack Incoming Webhook. `prometheus.yml`'s `alerting.alertmanagers`
  target now points at `localhost:9093`, and Prometheus confirms it's connected
  (`/api/v1/alertmanagers` shows it active). Verified end-to-end: the real
  `KNNWatchdogSustainedAnomaly` alert was observed landing in Slack with its
  full annotation text, not just a synthetic test. One real bug hit and fixed
  along the way — a stray trailing single-quote in the webhook URL in
  `/etc/alertmanager/alertmanager.yml` (no matching opening quote, so YAML
  read it as a literal trailing character) caused every notify attempt to
  fail with Slack redirecting to its docs 404 page; manual `curl`/`urllib`
  tests against the same URL kept succeeding because the diagnostic script's
  `.strip("'")` silently removed the bad character before testing, which is
  what made the bug hard to spot at first. Remaining gap: no alert
  currently routes anywhere except Slack (no email/PagerDuty-style
  escalation), and there's no alert-fatigue tuning (grouping/inhibition
  rules are still just the initial pass in `alertmanager.yml`).
- Metric *interpretation* (what `ai_risk_score=80` actually means, whether
  it's an emergency) has so far happened in AI chat sessions, not in the
  system itself — see the report-generator idea earlier in this doc as the
  planned fix.

**Scalability claims vs. reality**
- Covered above — k8s deployment is single-node, watchdogs are stateful, so
  "runs on k8s" and "horizontally scalable" are not yet the same claim.

**Credited as genuinely good:**
`ufw` ALLOW-127.0.0.1 + DENY-Anywhere layering (security instinct is right,
even if bind address undermines it); the Behavioral Attestation problem
statement itself is a real, current enterprise gap; `.gitignore` correctly
excludes secrets/models/data with no credentials found in git history
(verified via full history search, not assumed); **the Phase 6 "AI-managed
release traceability" demo target from `VISION.md` already succeeded once**
(`releases/guardian-defect-demo-2026-07-17.json`: a deliberately injected
defect — KNN retrain window dropped 2000→20 — was correctly caught by Phase
5 policy verification, `"passed": false`) — this note's earlier framing of
that as future work was stale by three days; `VISION.md`'s own
"deliberately narrow scope" section already names the over-engineering trap
most solo projects fall into.

**If gated for a real pilot:** minimum bar before touching real infrastructure
would be a test suite, sockets bound correctly (not relying on firewall
alone), and one documented "if this breaks, do X" runbook per service.
