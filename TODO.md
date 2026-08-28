# Guardian TODO

Running list of open work items. Unlike `ROADMAP.md` (Behavioral Attestation
phases) or `EDGE_ARCHITECTURE.md` (Pi/edge milestones), this is a flat,
general backlog — anything goes here regardless of which part of the project
it belongs to.

- [ ] **Core pipeline self-monitoring (transport-health layer).** Scoped
  2026-08-28, not started. See VISION.md's "Pipeline health model" —
  endpoint health → edge health → **transport health** → ingestion health →
  analysis health. This item covers transport health only; endpoint and
  ingestion health remain separate, un-scoped gaps.
  Prompted by a real incident: the Pi's `remote_write` to Core was silently
  broken for ~12 days (stale IP after a DHCP change), caught by accident.
  Plan:
  1. Add an `EdgePipelineStale` alert rule to
     `/etc/prometheus/rules/aiops-alerts.yml` —
     `absent_over_time(up{edge_site="guardian-proto-1"}[30m])`.
  2. Install Alertmanager (systemd service matching Prometheus's pattern),
     wire it into `prometheus.yml`, firewall-scope port 9093 like the
     watchdog ports.
  Existing gap this also fixes: Core already has sustained-anomaly alert
  rules for the three watchdogs, but Alertmanager isn't running, so none of
  them currently reach anyone.
  **Blocked on:** which notification channel? (email needs an SMTP relay;
  ntfy.sh/Pushover are simpler, no account needed for ntfy.sh; or just
  Alertmanager's own web UI, checked manually.) Decide before building step 2.

- [ ] **Route the second Windows machine (`DESKTOP-503POVP`) through the Pi.**
  Added 2026-08-28. Right now only `DESKTOP-0AJUKU3` goes through the edge
  path (Pi scrapes it, `remote_write`s to Core with `edge_site="guardian-proto-1"`)
  — `DESKTOP-503POVP` is only ever scraped directly by Core, with no edge
  involvement at all. Needs a second `static_configs` target added to the
  Pi's `windows-node` scrape job in its `prometheus.yml`, same as
  `DESKTOP-0AJUKU3`'s entry.

- [ ] **External AI/cyber threat observation on the Pi.** Added 2026-08-28,
  evaluated same day (analysis only, no code) — full writeup in
  `EDGE_ARCHITECTURE.md`'s "External AI/cyber threat observation" section.
  Not started. Summary of the evaluation:
  - Fits the existing "Edge Collector gathers evidence, Core
    analyzes/correlates" architecture as a new evidence type, not a new
    engine.
  - Real limitation: the Pi isn't inline with network traffic today, so it
    can only see traffic aimed at itself (free, no new hardware) and local
    broadcast/multicast — seeing traffic between *other* devices needs a
    SPAN port on a managed switch (new hardware + wiring), a TAP, or an
    inline/gateway position (ruled out, conflicts with "not a router").
  - The AI-specific angle (attacks on LLM interfaces/model endpoints) is
    real but currently inapplicable here — no AI-related port is externally
    exposed in this environment today, so there's nothing concrete to
    detect yet on that front specifically.
  - **Recommended first step, no new hardware:** Pi self-targeted evidence
    (SSH auth failures, port-scan patterns via `psad` or equivalent),
    exported via the existing textfile-collector/Prometheus pattern, feeding
    the attempt-vs-impact correlation idea. DNS-based threat evidence
    (Pi-hole or similar) is a reasonable second step; SPAN-port/hardware
    work is a third tier, only after 1-2 prove the correlation is valuable.
