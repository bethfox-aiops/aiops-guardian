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

**Milestone 1 — prove the deployment path only.** No endpoint agent code
yet. Buy the Pi, install Raspberry Pi OS/Ubuntu + Prometheus (+ optional
Grafana) + Guardian Edge, point it at exporters already running on one
Linux box and one Windows box (`node_exporter`, `windows_exporter`).
Single question to answer: *can the Pi collect and forward to Guardian
Core?* Nothing else matters until this works.

**Milestone 2 — find the gap.** Once M1 works, ask what evidence is
missing that exporters don't provide (processes, event logs, user/account
changes, SSH keys, etc.) — this produces the actual feature list for the
endpoint agent, instead of guessing it up front.

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
