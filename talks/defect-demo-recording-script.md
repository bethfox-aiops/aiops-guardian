# Guardian Demo Recording Script: "Catching a Bad AI-Driven Change"

**Target length:** ~4-5 minutes. Trim narration for the panel; use the full
version for the two focused talks.

**One-line framing to open with:**
> "Most AI demos show an agent doing work. This one shows the system proving
> the agent was governed while it did it."

**Why this scenario:** it's not hypothetical — this exact defect was
deliberately injected and caught for real on 2026-07-17. The recording
reproduces it; it isn't staged from scratch.

---

## Before you hit record: safety setup (not part of the recording)

This touches files a live systemd service (`aiops-watchdog-knn.service`)
depends on. Do this backup *before* recording, and don't narrate it as part
of the take — it's just protecting the live system, not part of the story.

```bash
cd /home/beth/aiops-agents
cp knn_model.pkl /tmp/knn_model.pkl.backup
cp scaler.pkl /tmp/scaler.pkl.backup
```

(The live watchdog service is never restarted during this demo, so it keeps
running on its already-loaded model in memory the whole time regardless —
this backup is belt-and-suspenders for the files on disk.)

---

## Beat 1 — Open (30s)

**Say:**
> "I'm going to deliberately break something Guardian depends on, and show
> you it catches the problem automatically — before it ever reaches
> production. This is Guardian's Behavioral Attestation engine: not just
> 'is something wrong,' but 'can I prove what an AI-driven change actually
> did.'"

**Show:** `VISION.md`'s north star line, or just say it:
> "The organizing question is: can I explain and prove what this system
> actually did?"

---

## Beat 2 — Show the safety net that's about to get tested (45s)

**Run:**
```bash
cat behavioral_policy.py
```

**Say, pointing at the `retrain_knn` policy block:**
> "This is a policy Guardian holds itself to. It says: a KNN model retrain
> needs at least 100 rows of real data to be trustworthy. Anything less,
> and the model shouldn't be trusted — even if the retrain script runs
> successfully with no errors."

---

## Beat 3 — Make the deliberate change (30s)

**Run:**
```bash
grep -n "RECENT_ROWS" retrain_recent_knn.py
```

**Say:**
> "Right now it's set to 2000 rows. I'm going to simulate a bad AI-driven
> change — someone, or an agent, drops that number way too low."

**Edit `retrain_recent_knn.py`** at the `RECENT_ROWS` line the grep above just
showed (don't hardcode a line number here — it's shifted before, e.g. when
the retrain scripts were refactored on 2026-07-24, and the grep already
points at it live on screen):
```python
RECENT_ROWS = 20   # was 2000
```

---

## Beat 4 — Run it for real (30-60s, depends on how long the retrain takes)

**Run:**
```bash
python3 retrain_recent_knn.py
```

**Exact terminal output you'll see** (confirmed from the actual script, not
paraphrased — good to know ahead of time so nothing on screen surprises you
mid-recording):

```
[INFO] Trace ID: <32-character hex trace id>
[INFO] Training on most recent 20 rows.
[INFO] Training KNN anomaly detector...
[INFO] eBPF evidence during training: {...syscall counts...}
[INFO] Anomalies flagged in training data: X / 20
[INFO] Saved knn_model.pkl, scaler.pkl
[VERIFY] FAIL: this run violated its behavioral policy:
  - row_count 20 below policy minimum 100
[INFO] Done. Restart aiops-watchdog-knn to load new model.
```

Total output is short — under 10 lines. Nothing to scroll past, easy for a
viewer to actually read on screen.

**Say while it runs:**
> "This is a real retrain — real data, real model training, instrumented
> with OpenTelemetry so the whole run is traceable end to end. It's not
> going to error out. It'll succeed. That's exactly the problem —
> 'ran without errors' and 'produced something trustworthy' are not the
> same claim."

---

## Beat 5 — The catch (this is the moment — 30s)

**Point at these two lines specifically** (the money shot — consider
pausing/zooming here if editing the recording):

```
[VERIFY] FAIL: this run violated its behavioral policy:
  - row_count 20 below policy minimum 100
```

**Say:**
> "There it is, printed right to the terminal: this run failed its own
> policy. Guardian didn't just watch the retrain succeed — it checked the
> result against a stated policy and caught that it shouldn't be trusted,
> even though nothing 'crashed.'"

---

## Beat 6 — Show the evidence record (45s)

**Run:**
```bash
cat releases/guardian-defect-demo-2026-07-17.json
```

**Say:**
> "This is the actual record from the first time I ran this, three days
> [or however long ago it now is] before recording this. It ties together
> three things that normally live in different places: which AI agent made
> the change and what it was asked to do, the exact git commit and files
> involved, and what actually happened when it ran — the trace, and the
> policy violation. That's the whole idea: not 'trust the agent,' but
> 'here's the evidence.'"

---

## Beat 7 — Put it back (20s, can speed up in editing)

**Run:**
```bash
git checkout -- retrain_recent_knn.py
cp /tmp/knn_model.pkl.backup knn_model.pkl
cp /tmp/scaler.pkl.backup scaler.pkl
```

**Say:**
> "And that's reverted — the live watchdog was never at risk this whole
> time, because the bad model never got deployed. It was caught first."

---

## Beat 8 — Close (20-30s)

**Say:**
> "Most people demoing an AI system show it doing work. The harder and more
> useful question is whether you can prove it was governed while doing it —
> that's what this is built to answer."

(For the two focused talks, this is a good place to bridge into more
architecture detail. For the panel, stop here.)

---

## Notes for trimming to panel length (~90s total)

Keep: Beat 1 (shortened), Beat 5 (the catch), Beat 8 (the close). Cut or
summarize in one sentence: Beats 2-4 and 6-7 — replace with "I dropped a
retrain's data window way too low, on purpose, and Guardian's own
verification caught it before it shipped" as a single spoken sentence over
a screenshot of the `passed: false` output.

## If you want a second, shorter backup clip

The IPv6 firewall-parsing bug fixed on 2026-07-20 is a good secondary
story: found during a routine lint pass, not a security audit; fixed;
verified against real `ufw` rules; and now permanently covered by a
regression test (`test_v6_only_deny_returns_true` in
`test_guardian_health.py`) that runs on every push via GitHub Actions.
Good "even boring maintenance work has a governance angle" beat if a talk
runs long and needs a second example.
