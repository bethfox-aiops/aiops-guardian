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

## Sequencing notes

- Phases are ordered by dependency, not necessarily by priority — Phase 1 unlocks
  the most value for the least effort and should come first regardless.
- Phases 2–4 (eBPF, tracing, GPU) can be reordered or run in parallel; they don't
  depend on each other, only on Phase 1's anomaly-timeline concept existing.
- Phases 5–6 both depend on having real trace data (Phase 3) to verify/attach to,
  so they come last regardless of how the middle phases are resequenced.

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
