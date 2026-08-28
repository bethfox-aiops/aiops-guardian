# Guardian: Vision

This document captures the long-term architectural direction for Guardian. It is
intentionally written at the vision level, not the implementation level — the
underlying tech (which models, which eBPF tooling, which trace format) is expected
to change over the years; the shape of what Guardian is for should not.

## North star

Guardian's organizing question is:

> **Can I explain and prove what this system actually did?**

Every prior stage of the project — metrics, anomaly detection, security checks,
AI governance — was really a step toward being able to answer that question with
evidence instead of guesswork. Behavioral Attestation is where that question
becomes the explicit point of the project rather than an implicit byproduct.

**Litmus test for new work:** a new Guardian feature belongs if it does at least
one of these:

- Collects evidence
- Explains behavior
- Attributes responsibility
- Increases trust in autonomous systems

If a proposed feature doesn't do any of those, it probably belongs in a different
project, not bolted onto Guardian.

**Sharper framing (added 2026-07-20):** an industry post crossed the user's feed
making the same point from a real-world audit angle — the hard part isn't an AI
agent doing the work, it's proving the agent was *governed* while it did it: least-
privilege access, a documented reason to exist, tamper-evident logging. An auditor
doesn't care that an agent wrote the evidence; they care whether the agent was
controlled. That's a more precise restatement of Guardian's own north star than
"can I explain and prove what this system did" alone — it's specifically "can I
prove the *agents* were governed," not just that behavior was observed. Guardian's
Phase 6 (`release_record.py`) is genuine, working progress toward this (see
`releases/guardian-defect-demo-2026-07-17.json` for a real example: an AI-made
change, evidence collected, a policy violation correctly caught). But be honest
about the gap this framing exposes — Guardian does not yet *govern* the AI agents
that act on it (this project has been built via Claude Code operating with the
user's own full, unscoped session — no separate least-privilege identity, no
mapping to a real framework like ISO 42001 or the NIST AI RMF, and nothing in the
current logging is genuinely tamper-evident, i.e. cryptographically signed or
append-only, as opposed to just "observable"). Closing that gap — not just adding
more observation — is the next real maturity step; see the corresponding items in
`ROADMAP.md`.

## What Guardian actually is

Guardian is not an AI/ML project with some extra features. It's a **systems
intelligence project**, where AI workloads happen to be one of the things it
observes — alongside Linux processes, storage, networking, and everything else a
complex system does. That's why it's called an **AI Systems Intelligence
Platform** rather than "AIOps": AIOps names the tools; Systems Intelligence names
the problem being solved.

## Architecture: five engines

```
Guardian
│
├── Observability Engine
│     Metrics
│     Logs
│     Dashboards
│
├── ML Engine
│     KNN
│     Isolation Forest
│     Autoencoder
│
├── Security Engine
│     File Integrity
│     User Monitoring
│     SSH
│     AI Security
│
├── Governance Engine
│     Guardrails
│     Human Approval
│     Policy
│
└── Behavioral Attestation Engine   ⭐
      Trace Collection
      eBPF
      OpenTelemetry
      GPU Tracing
      Process Attribution
      Performance Root Cause
      Workflow Verification
```

**What exists today, mapped to this structure:**

| Engine | Status |
|---|---|
| Observability | Metrics + dashboards exist (Prometheus/Grafana); structured Logs engine does not yet exist |
| ML | All three models running (KNN, Isolation Forest, Autoencoder watchdogs) |
| Security | Running today via the Guardian security score (file integrity, SSH/user monitoring, AI security checks) |
| Governance | Approval Dashboard (`aiops-approval.service`) provides the Human Approval piece; a first version of Guardrails also exists (`guardrail_exporter.py`, `~/aiops-guardrail-lab`, port 8015) — see note below |
| Behavioral Attestation ⭐ | Built — all six roadmap phases (`ROADMAP.md`) have real, working modules (`process_attribution.py`, `ebpf_trace.py`/`trace_suspect.sh`, `otel_setup.py`, `gpu_attribution.py`, `behavioral_policy.py`, `release_record.py`), wired into the watchdogs and used live (e.g. the KNN/Autoencoder root-cause attribution in the 2026-08-28 daily report). Phase 7 ("govern the agent, not just the output" — see the Sharper framing note above) is the newest, still-open extension, not a gap in Phases 1–6. |

**Guardrails, v1 (added 2026-07-27):** an early guardrail-policy exporter exists
at `~/aiops-guardrail-lab/scripts/guardrail_exporter.py` (`guardrail-exporter.service`,
port 8015, scraped by Prometheus as job `guardrail_lab`) — it counts
ALLOW/BLOCK/APPROVED/DENIED/INVALID actions and LOW/MEDIUM/HIGH/CRITICAL risk
tiers from a guardrail-decision log and exports them as Prometheus gauges
(`guardrail_allow_total`, `guardrail_block_total`, etc.), surfaced on the
"AI Anomaly Detection" Grafana dashboard's "System Guardrails" row. It grew
through three incremental versions (`guardrail_v1.py` → `v3.py`) against
simulated "prod"/"backup" customer data before becoming an exporter, and lives
outside the `aiops-agents` git repo (untracked, no version control of its own)
— treat it as this engine's genuine first version, not a finished Guardrails
system: it counts and classifies actions from a log, it doesn't yet enforce
anything or integrate with `behavioral_policy.py`'s Phase 5 verification.
Formalizing/merging the two (this exporter's action-classification model and
`behavioral_policy.py`'s per-workflow policy checks) into one real Guardrails
component is exactly the kind of work Phase 7 (`ROADMAP.md`) should absorb.

**The motivating idea behind it (clarified 2026-07-27), worth preserving
because it's easy to lose in the exporter/dashboard mechanics above:** the
lab wasn't just a tutorial exercise — it was a deliberate test of *limiting
what a script is allowed to execute* as a form of AI governance, built on a
specific belief: **systems engineers can cause an AI agent to do things it
shouldn't, by accident**, through the tooling/scripts/permissions they hand
it, not only through the agent's own decisions. This is a distinct failure
mode from "the agent needs its own scoped identity" (Phase 7 item 1 below) —
that item is about the *agent's* privilege boundary; this is about the
*human's own tooling design* being a governance surface in its own right. An
engineer who writes a script an AI agent will run doesn't need to intend
over-permissioning for it to happen — an overly broad shell wrapper, a
too-permissive sudoers entry, a script that takes unvalidated input, all hand
an agent more capability than the task in front of it needs. Guardrails, in
this framing, are as much about constraining what *engineers accidentally
build for agents to use* as about constraining the agent itself.

## Pipeline health model

Guardian doesn't just observe target systems — it depends on its own pipeline
staying healthy in order to observe anything at all. That pipeline breaks down
into five layers, each of which can fail silently and independently of the
others:

> **endpoint health → edge health → transport health → ingestion health → analysis health**

- **Endpoint health** — is the collector on the observed host actually running
  and producing fresh data (a textfile-collector script, a watchdog process)?
- **Edge health** — is the intermediate collection point itself up (the Pi,
  `guardian-proto-1`, in the current edge architecture)?
- **Transport health** — is data actually moving between stages (`remote_write`
  from edge to Core, a scrape succeeding)?
- **Ingestion health** — is Core actually receiving and storing what's sent to
  it (Prometheus accepting writes)?
- **Analysis health** — is the layer that turns raw data into a verdict (the
  anomaly watchdogs, the scoring engines) producing correct output from what
  it's given?

**Why this matters, concretely (2026-08-28):** the Pi's `remote_write` to Core
was silently broken for ~12 days — a pure transport-health failure. Endpoint
health (the Windows collector scripts) and edge health (the Pi's own scrape of
those hosts) both looked completely fine the whole time, which is exactly why
nobody noticed: nothing was watching the transport layer specifically. A
monitoring system that only checks "is the data I have right now healthy"
(analysis health, roughly what the daily report and the anomaly watchdogs do)
will never catch a failure at an earlier layer that just results in *no new
data arriving* — that reads as quiet, not broken.

**How to apply:** any future Guardian self-monitoring work should be scoped
against this model explicitly — name which layer(s) it covers, and treat "we
monitor the pipeline" as false until all five layers have real, independent
checks. The Core self-monitoring work in `TODO.md` (data-freshness alerting +
Alertmanager) is transport-health monitoring specifically — it does not, on
its own, cover endpoint or ingestion health, which remain open gaps.

## Deliberately narrow scope

The temptation is to build "performance attribution for everything." Don't.
The sharper, more valuable niche is:

> **Performance attribution for AI workloads on Linux.**

That scope is achievable, demonstrable, and directly plays to the systems
background (Unix, Prometheus, Grafana, Linux internals) that makes Guardian
credible instead of derivative.

## Two flagship use cases

Two concrete use cases anchor Behavioral Attestation and should be built in this
order:

### 1. Performance attribution for AI workloads (build first)

Attribute a performance problem across the full stack: from a Prometheus metric,
to a Linux process, to an OpenTelemetry trace, to a specific stage of a training
job, correlating GPU utilization and system calls into a single evidence chain.

**Demo target:** intentionally introduce four distinct problems into the
autoencoder workload and have Guardian correctly identify each one:

- Inefficient Python code
- Excessive system calls
- Slow storage
- GPU starvation

Correctly diagnosing all four, with evidence, is the bar for "done" on this use
case — more convincing than claiming to monitor many services shallowly.

### 2. AI-managed release traceability (build second)

AI-managed releases can make many coordinated changes quickly — edit files,
change dependencies, rebuild artifacts, change config, deploy, restart services,
run migrations — and when something breaks, the hard problem isn't that AI made
a bad change, it's reconstructing *what changed, why, what ran afterward, and
which change caused the failure.* (Motivating data point: a July 2026 study found
agent-generated systems often had logs, but exposed useful failure-specific
runtime evidence for only a small fraction of injected failures — looking
observable and being diagnosable are not the same thing.)

Guardian's answer is to connect two kinds of provenance that normally stop at
different boundaries:

- **Build provenance** (SLSA-style, stops at the artifact): who/what requested
  the change, which agent made it, what task/prompt/ticket governed the work,
  which files/dependencies changed, which tests ran, which commit produced the
  release, whether the artifact was signed/verified.
- **Runtime behavioral attestation** (Guardian's extension, continues past
  deployment): what actually executed, which services/processes changed
  behavior, whether it accessed unexpected files or opened new network
  connections, whether resource consumption changed, whether the observed
  execution path matched the approved release plan, and which change most
  likely caused a regression.

**Illustrative scenario** (the kind of output this should produce):

```
Release 2026.07.16.4
Requested outcome: Improve autoencoder data-loading performance

AI actions:
- Modified preprocessing.py
- Upgraded pandas dependency
- Changed worker count from 2 to 12
- Rebuilt container image
- Deployed aiops-autoencoder:v17
- Restarted training service

Runtime result:
- Training duration improved 18%
- Memory usage increased 240%
- Swap activity began
- Prometheus latency increased
- Two unrelated exporters missed scrapes

Attributed cause: Excessive worker concurrency introduced by the AI-managed release
Evidence: git diff, AI action log, build provenance, deployment event,
          PID/container correlation, memory-pressure metrics, process/syscall trace
```

**Demo target:** let an AI coding agent (e.g. Claude Code) manage a controlled
release of the autoencoder end-to-end, with Guardian recording: the stated
objective, commands requested and approved, files/config changed, tests/
validation performed, commit and artifact hashes, deployment events, and
process/CPU/memory/disk/network/GPU behavior before and after. Then deliberately
introduce a release defect (e.g. excessive worker concurrency, repeated dataset
reload) and confirm Guardian traces the resulting regression back to that exact
change.

**Target demonstration statement:**

> Guardian provides behavioral attestation for AI-managed software releases,
> connecting AI decisions and code changes to their actual runtime effects.

## Why this shape, not another

This didn't start as a plan to build "an AI Systems Intelligence Platform with
Behavioral Attestation" — that would have been unreconciled ambition on day one.
It emerged in stages, each one exposing the limitation that motivated the next:
metrics → anomaly detection → automation/guardrails → security monitoring → AI
governance → *"can I prove what happened?"* The throughline across every stage
was never chasing a trend for its own sake — every addition had to answer a real
operational question first. Behavioral Attestation is the stage where that
throughline becomes the project's explicit identity rather than an implicit
pattern in hindsight.

## Where implementation specifics are decided

Which correlation techniques, which eBPF framework, how traceability is stored/
queried, and how the two demo use cases get built out phase-by-phase are
deliberately not fixed here — see [`ROADMAP.md`](ROADMAP.md) for the mutable,
expected-to-change execution plan.
