# Guardian Edge Architecture

This document captures the planning conversation (2026-07-27, sourced from
ChatGPT) that clarified how Guardian would deploy onto a customer's network
as part of the future consulting business (see `VISION.md`'s long-term
direction and the edge/consulting future-state goal). It is written at the
same level as `VISION.md` — architecture and terminology, not implementation
— because the underlying question ("what physical/software pieces exist and
what does each one do?") needed to be settled before any code gets written.

**Status: planning only.** No Raspberry Pi has been purchased yet and no
code from this document has been written. Nothing here should be built
until the user explicitly picks this back up with hardware in hand.

## The key clarification

The Raspberry Pi is a **collector**, not a per-endpoint device. You do not
need one Pi per monitored machine — one Pi per customer site is the design
point. What each customer *server* needs instead is a small piece of
software, not another appliance.

## Three-tier architecture

```
Windows Server                    Linux Server
  ├─ windows_exporter                ├─ node_exporter
  └─ Guardian endpoint agent         └─ Guardian endpoint agent
              \                          /
               \                        /
                v                      v
          Raspberry Pi — Guardian Edge Collector
            ├─ Scrapes metrics (exporters)
            ├─ Receives evidence (from endpoint agents)
            ├─ Detects initial anomalies (first-pass, local)
            ├─ Buffers data
            └─ Sends it securely to Guardian Core
                            |
                            v
                     Guardian Core
              (Dell / central Linux box — today's
               full repo: Behavioral Attestation,
               correlation, AI analysis, dashboards,
               reporting)
```

### Terminology (this is the part that was confusing)

| Term | Meaning |
|---|---|
| **Endpoint** | The Windows or Linux machine actually being observed |
| **Endpoint agent** | Small software running *on* that machine |
| **Edge Collector** | The Raspberry Pi — one per customer site, collects from all that site's endpoints |
| **Guardian Core** | The existing Dell/laptop — deeper analysis, storage, dashboards, reporting |

Example sizing: a 5-server customer site needs 5 endpoint agents + **1** Pi,
not 5 Pis.

## Why an endpoint agent instead of the Pi reaching out directly

The Pi *could* poll each server remotely (WinRM, SSH, WMI, APIs), but that
concentrates credentials on the Pi and creates firewall/reliability
complications per protocol. A small local agent that collects evidence and
pushes it to the Pi is cleaner and safer, and fits the push-based design
already noted as required in [[project-guardian-edge-consulting]] (a
consultant's Pi usually cannot reach back into a locked-down customer
network — Promtail's existing push-to-Loki model is the template to
follow).

## What the endpoint agent actually needs to be

Not "an agent" in the CrowdStrike/Defender/Tanium sense — something much
narrower:

1. Accept a request from the Edge Collector (e.g. `{"request": "process_snapshot"}`)
2. Run an **approved, read-only** check
3. Collect the results
4. Hash (or sign) the evidence
5. Send it back to the Pi

Realistic estimate: a first version is "a few hundred lines of Python," not
a product. Much of the logic already exists in this repo and would move
from running on the Dell to running locally on each endpoint:
`process_attribution.py`-style process listing, file hashing, security
checks, eBPF-style evidence collection — same ideas, smaller footprint,
different host.

## Future requirement: both platforms needed

The endpoint agent will eventually need to exist for **both Windows and
Linux/Unix** — customer sites have both (see the two-column diagram above).
No language/implementation decision has been made yet; this is just a
stated requirement to design for once endpoint-agent work actually starts
(Milestone 3, not before).

**Related but distinct (2026-07-29):** `windows/guardian_disk_health.ps1`
in this repo is a real PowerShell collector deployed to a Windows host, but
it is **not** an early version of the endpoint agent above — it's a narrow
disk-health diagnostic (SMART/reliability counters + Event Log data) that
writes a local file for `windows_exporter`'s own `textfile` collector to
pick up, built in response to a real drive-failure investigation (see
`OPERATIONS_MANUAL.md` Chapter 4.9). It never accepts a request, runs
arbitrary approved checks, or pushes evidence anywhere — the three things
that would make it a genuine first version of this document's endpoint
agent. Worth knowing it exists as prior art (real PowerShell running
unattended on a Windows host via Scheduled Task) if/when Milestone 3 work
actually starts, but don't conflate the two.

## Rollout plan — deliberately slow, milestone-based

The conversation's own conclusion was to **slow down** rather than design
the whole three-tier platform up front — this mirrors how Behavioral
Attestation itself grew (process attribution → eBPF → traces → evidence
correlation), not designed end-to-end in advance.

**Milestone 1 — prove the deployment path only. DONE (2026-08-14).** No
endpoint agent code yet. Buy the Pi, install Raspberry Pi OS/Ubuntu +
Prometheus (+ optional Grafana) + Guardian Edge, point it at exporters
already running on one Linux box and one Windows box (`node_exporter`,
`windows_exporter`). Single question to answer: *can the Pi collect and
forward to Guardian Core?* Nothing else matters until this works.

Scope was narrowed at execution time to **Windows only** (`DESKTOP-0AJUKU3`'s
`windows_exporter`, not the Linux/`node_exporter` target) — a deliberate
call to keep the first real proof to one target, not a gap. What was
actually built: Prometheus installed on the Pi (`apt install prometheus`,
Debian 13/trixie has it packaged, 2.53.3), one `scrape_configs` job
(`windows-node` → `DESKTOP-0AJUKU3:9182`), one `remote_write` target
pointed at Guardian Core (`http://<core-ip>:9090/api/v1/write`), and a
`global.external_labels: {edge_site: guardian-proto-1}` so pushed series
stay distinguishable from Core's own pre-existing direct scrape of the
same Windows host. On Guardian Core: `--web.enable-remote-write-receiver`
added to Prometheus's existing systemd drop-in (merged into the current
`override.conf` rather than left as a second competing drop-in — an
earlier draft of this change would have silently dropped the already-live
`--web.enable-lifecycle` flag by clearing `ExecStart` a second time;
caught via `systemctl cat` before restarting, not after). Verified live:
all 164 of `windows_exporter`'s default-collector metrics (CPU, memory,
disk, network, services, system) landing in Guardian Core's Prometheus
with the `edge_site` label intact, timestamp advancing across repeated
checks (steady stream, not a one-off), zero errors in the Pi's Prometheus
log. Confirmed reachable via Grafana's existing datasource (same
Prometheus instance every current dashboard already queries) but **no
dashboard/panel built for it yet** — deliberately out of scope for M1,
which only had to prove the forwarding path.

**Security note carried forward from this change:** Guardian Core's port
9090 is `ufw ALLOW IN Anywhere` (not scoped to the Pi's IP, unlike the
narrower pattern used elsewhere in this repo for exactly this reason) —
enabling the remote-write receiver on it means anything that can reach
this laptop's 9090 can now push arbitrary time series into Guardian
Core's Prometheus, not just read from it. Worth scoping to the Pi's IP
(or a real multi-site CIDR once there's more than one edge device) before
this goes beyond a single-Pi prototype — see OPERATIONS_MANUAL.md 10.2.

**Milestone 2 — find the gap. DONE (2026-08-14).** Once M1 works, ask what
evidence is missing that exporters don't provide (processes, event logs,
user/account changes, SSH keys, etc.) — this produces the actual feature
list for the endpoint agent, instead of guessing it up front.

Method: cross-referenced what `windows_exporter` actually exposes on
`DESKTOP-0AJUKU3` (confirmed live via `windows_exporter_collector_success`
— only `cpu, logical_disk, memory, net, os, physical_disk, service,
system, textfile` are enabled; notably **not** `process`) against Guardian
Core's real security/health/AI-risk checks (`guardian_security.py`,
`process_attribution.py`, `guardian_ai_risk.py`, Behavioral Attestation
Phases 1-2 — 36 distinct functions surveyed). Findings, most important
first:

1. **Process-level visibility — the biggest gap.** `get_top_processes`,
   root-process count, zombie count, procs running from `/tmp`: zero
   equivalent for the Windows endpoint right now. `windows_exporter` ships
   an optional `process` collector but it isn't enabled — no per-process
   CPU/mem/handle data reaches Guardian Core at all currently. This is
   Guardian's core "who caused this anomaly" differentiator (Behavioral
   Attestation Phase 1) and it's completely dark for anything monitored
   only via exporters.
2. **Security/integrity checks — entirely absent.** SSH key changes, new
   user accounts, sudo activity, cron/scheduled-task changes, service
   *config* changes (vs. just state, which is covered), SUID-binary- and
   sudoers.d-equivalents, package/patch integrity, failed logins,
   driver/kernel-module changes. 15+ checks with zero exporter coverage —
   real endpoint-agent territory, not fixable by enabling another
   collector flag.
3. **Network/connection-level detail.** `windows_exporter` gives aggregate
   byte/packet counters only — no equivalent of outbound-connection count,
   high-port-listener detection, DNS-connection tracking, or promiscuous-
   interface detection. Would need something like `Get-NetTCPConnection`
   queried locally.
4. **eBPF/syscall tracing (Behavioral Attestation Phase 2) — structurally
   unavailable via any exporter.** The real Windows analog is ETW/Sysmon,
   a fundamentally heavier mechanism than anything else on this list —
   worth treating as its own future scope, not folded into M3.
5. **AI-risk detection** (`detect_ai_packages/processes/api_keys`) — same
   process-visibility gap as #1, plus no installed-software inventory.

**Already covered, not a gap:** CPU/mem/disk/network throughput, service
*state* (already used by the `CRITICAL_SERVICES` check in
`aiops-watchdog-windows.py`), boot time. There's also already a working
precedent for exactly this fill-the-gap pattern: `guardian_disk_health.ps1`,
a small PowerShell script writing a textfile-collector `.prom` file to
cover the SMART/disk-Event-Log gap `windows_exporter` doesn't — the model
to follow for whichever M3 capability gets built first.

**Candidate for M3:** process-level attribution (#1) — Guardian's core
differentiator, has a clean precedent to follow, small enough to be one
real capability rather than a redesign. Not started; this is a
recommendation for the next session, not a decision made.

**Milestone 3 — write exactly one endpoint capability.** Not "the agent" —
one capability (e.g. "return running processes"). Prove the concept before
generalizing.

Guardian ends up as three distinct products/deliverables:

1. **Guardian Endpoint Agent** — small, read-only, evidence collection
2. **Guardian Edge Collector** (Raspberry Pi) — scrapes exporters, receives
   evidence, first-pass anomaly detection, buffers/forwards
3. **Guardian Core** — Behavioral Attestation, evidence correlation, AI
   analysis, dashboards, reporting (this is what already exists today)

## How this relates to existing memory/docs

- [[project-guardian-edge-consulting]] already captured the *why* (consulting
  business future-state) and the push-not-pull / minimal-footprint /
  multi-tenant-labeling constraints — this document is the architectural
  follow-on that resolves "what actually runs where."
- Don't conflate this with near-term scoping: per
  [[user-guardian-motivation]], job-search framing still drives what gets
  built *now*. This document is deliberately parked until hardware exists.
