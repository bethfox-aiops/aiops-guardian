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

**Pi-side security hardening, closed 2026-08-16:** `guardian-proto-1` shipped with the Raspberry Pi OS default — unrestricted `pi ALL=(ALL) NOPASSWD: ALL` sudo (from two separate files, the image default plus a cloud-init-generated one) — and `sshd` configured to accept password authentication at the daemon level, with only the account's locked password standing between that and remote password-guessable login. Both closed: `PasswordAuthentication no` set explicitly (deliberate now, not an accident of the account having no password), a real password added to the account afterward as a local-console/`sudo` fallback only (safe once SSH can't use it remotely), and the blanket sudo grant replaced with one scoped rule (`systemctl restart prometheus.service` — the only thing actually needed passwordlessly so far). Full procedure, including the safe ordering to avoid locking yourself out mid-change, now in `PI_SETUP_CHECKLIST.md`. **Matters specifically for the multi-site consulting plan** (not just this prototype): default credentials/sudo on a device meant to eventually sit on a customer's network is a real risk, worth checking at flash time for every future Pi rather than fixing after the fact each time.

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
real capability rather than a redesign.

**Status update (2026-08-26): M3 script written, not yet deployed.**
`windows/guardian_process_attribution.ps1` follows the
`guardian_disk_health.ps1` textfile-collector pattern — top-5-by-CPU and
top-5-by-memory process snapshots via `Get-Counter`, not the built-in
`process` collector (deliberately curated, matching Linux's
`process_attribution.py` top-N approach rather than exposing every
process). See OPERATIONS_MANUAL.md Chapter 4.9 for full detail. Not yet
deployed to either Windows host, no dashboard panel, no Scheduled Task
registered — the script exists and needs real-host verification next.
Prompted by the Pi (`guardian-proto-1`) being SSH-unreachable that day —
this work is independent of Pi access (a locally-run PowerShell script,
nothing Pi-related), so it made sense to pick up while blocked there.

**Milestone 3 — write exactly one endpoint capability.** Not "the agent" —
one capability (e.g. "return running processes"). Prove the concept before
generalizing.

**Milestone 4 — real push, not just pull. Planned (2026-08-26), not started.**
Everything built so far (`guardian_disk_health.ps1`, M3's
`guardian_process_attribution.ps1`) is *pull*: the script writes a local
file, `windows_exporter`'s textfile collector re-exposes it, Prometheus
scrapes it. That's fine for a host Guardian Core can already reach, but it's
the opposite of the endpoint-agent design above (push, because a
consultant's Pi/Core usually *can't* reach into a customer network). M4
is where that actually gets built, using `guardian-proto-1` as the landing
point per the architecture above rather than pushing to Core directly.
Ordered, each step depends on the one before it:

1. **Deploy + verify `guardian_process_attribution.ps1` on a real Windows
   host.** Written 2026-08-26, never run — no PowerShell runtime existed to
   syntax-check it against. Closes out before adding more surface on top.
2. **Design the push payload.** Small schema decision needed before any
   code: JSON over HTTP POST (simple, host-agnostic) vs. the endpoint
   speaking Prometheus `remote_write` itself (heavier, but reuses the
   protocol M1 already proved). Minimal shape: `{host, category,
   timestamp, data}`.
3. **Build the Pi-side receiver.** Nothing listens for a push today — M1
   only set the Pi up to scrape *outward*. A small HTTP service accepting
   evidence and either re-exposing it as textfile-style metrics for the
   Pi's own Prometheus, or converting and forwarding via the same
   `remote_write` path M1 proved. Needs a shared-secret/token check: a new
   inbound listener on a device meant to eventually sit on customer
   networks is real new attack surface, not a detail to skip.
4. **Firewall-scope the new listener.** Narrow `ufw` rule for the
   receiver's port, same discipline as `ufw_guard.sh` elsewhere in this
   project. Good moment to also close the gap M1 already flagged and left
   open: Core's port 9090 is currently unscoped `ALLOW IN Anywhere` and
   accepts remote_write, not just reads.
5. **Windows push script — security/integrity checks first.** #2 on M2's
   gap list, right after process attribution (#1, done in M3). New local
   users (`Get-LocalUser`), failed logons (Security event 4625),
   Administrators-group membership changes. Same 15-minute Scheduled Task
   cadence as the existing scripts, but POSTs to the M4 receiver instead of
   writing a textfile.
6. **Multi-tenant labeling on this path.** Tag whatever the receiver
   forwards with `edge_site=guardian-proto-1` (or a `host`/`source` label),
   matching M1's convention — a from-day-one design constraint per
   [[project-guardian-edge-consulting]], not something to retrofit once a
   second site/Pi exists.
7. **Grafana panel**, once real data is flowing end-to-end. Deliberately
   last, same "prove it before generalizing" discipline as every milestone
   so far.

## External AI/cyber threat observation — evaluation (2026-08-28, analysis only, no code)

User's proposed future Edge capability: have the Pi also observe suspicious
activity attempting to enter the local environment from outside — not just
internal system/AI behavior, which is everything Guardian does today. Explicit
non-goal stated up front: not a firewall/router/IDS/IPS product, just evidence
collection that can later be correlated with existing endpoint and Behavioral
Attestation data (external event → edge evidence → endpoint behavioral change
→ Guardian determines whether the attempt actually had impact). This is the
same "Edge Collector gathers evidence, Core analyzes/correlates/explains"
principle already established above, applied to a new evidence *type*
(network-perimeter) rather than a new engine.

**Fundamental limitation, not a code problem:** sitting where the Pi sits
today (a normal WiFi client, not inline), it can only see (a) traffic
addressed to itself (port scans/auth attempts against the Pi specifically —
free today, e.g. `psad` or iptables/journalctl log parsing) and (b)
broadcast/multicast on the local segment (ARP/DHCP/mDNS — visible regardless
of position). Unicast traffic between *other* devices (an attack against a
Windows host, that host's own outbound traffic) is invisible to the Pi on a
modern switched network without either a SPAN/mirror port on a managed switch
(new hardware, ~$30-60, plus wiring the Pi's onboard Gigabit Ethernet — it's
on WiFi today), a hardware TAP (more correct, unnecessary at home-network
volume), or an inline/gateway position (ruled out — directly conflicts with
the stated "not a router" constraint, makes the Pi a single point of failure
for internet access). Installing an IDS like Suricata on the Pi does not
solve this by itself; it needs traffic positioned in front of it via one of
the above, same requirement either way.

**Lower-effort alternative to direct observation:** consume logs from
something that already has visibility instead of building new observation
capability. DNS query logs (e.g. if Pi-hole or similar ran on the Pi as the
network's resolver) are a strong, low-effort source for malicious-
infrastructure activity (blocklist hits, DGA-domain lookups) with
network-wide coverage, no SPAN hardware needed — trades a new always-on
service + a DNS-config change on the router for much broader visibility than
the self-targeted option alone. Router/gateway log support is unknown for the
current Frontier-provided gateway and would need checking, not assuming.

**AI-specific vs. traditional split:** of the example threat categories
discussed, only "attacks against AI systems/LLM interfaces/model
endpoints/agents/exposed AI services" is genuinely AI-specific — everything
else (port scanning, failed auth, suspicious IPs, malicious DNS activity,
generic exploit indicators) is traditional network/security monitoring, not
novel to this project. Worth being honest that the AI-specific category is
also currently *inapplicable* in this environment: Guardian's own AI-related
ports are explicitly `ufw`-denied from external access (see `CLAUDE.md`'s
security posture note), so there is no externally-exposed AI attack surface
to defend today. This category becomes concretely actionable only if/when
something AI-related is ever exposed externally — it's the more novel,
differentiating part of the idea, but also the part with nothing real to
detect yet.

**Recommended minimal PoC, if this is picked up later (not started, no
milestone number assigned yet — doesn't fit the M1-M4 sequence above, which
is entirely about the pull/push endpoint-agent path, a different problem):**

1. **Pi self-targeted perimeter evidence** — watch for connection attempts
   aimed at the Pi itself (SSH auth failures, port-scan patterns via `psad`
   or equivalent), export via the same textfile-collector/Prometheus pattern
   used everywhere else in this architecture, feed into the same
   attempt-vs-impact correlation idea. Zero new hardware, fits existing
   patterns exactly — proves the correlation concept before any hardware
   spend.
2. **DNS-based threat evidence** (natural second step) — Pi-hole or
   equivalent for network-wide malicious-DNS visibility, still no SPAN
   hardware.
3. **SPAN-port/network-repositioning work** — only worth it once 1-2 have
   proven the correlation is actually valuable; real hardware and rewiring
   investment, not a first step.

See `TODO.md` for the tracked backlog item.

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
