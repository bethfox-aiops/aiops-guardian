<#
.SYNOPSIS
    Guardian process-attribution collector for windows_exporter's textfile
    collector.

.DESCRIPTION
    Milestone 3 of the Guardian Edge architecture (see EDGE_ARCHITECTURE.md):
    windows_exporter's built-in `process` collector isn't enabled on this
    host, so today there is zero per-process CPU/mem visibility reaching
    Guardian Core -- Behavioral Attestation Phase 1's core differentiator
    ("who caused this anomaly") is completely dark for Windows endpoints.
    Deliberately not just flipping on the built-in `process` collector: that
    would expose every process wholesale, a much wider/less-curated surface
    than the top-N-suspects pattern Guardian already uses on Linux
    (process_attribution.py). This script writes a curated top-N-by-CPU and
    top-N-by-memory snapshot instead, same textfile-collector approach as
    guardian_disk_health.ps1 (the established pattern for closing an
    exporter gap without a new service/port).

    CPU% here matches process_attribution.py's convention (psutil default):
    NOT normalized by logical processor count, so a busy multi-threaded
    process can read >100%. `\Process(*)\% Processor Time` is a formatted
    (already-rate-calculated) perf counter, so a single Get-Counter call is
    sufficient here -- no manual before/after delta sampling is needed the
    way it would be reading raw WMI counters, and no state needs to persist
    between runs (this script is stateless, unlike the long-lived Linux
    watchdog process that primes psutil's internal per-pid timer once and
    reuses it every 5s tick).

    Perfmon's Process instance names get a `#N` suffix for duplicates
    (multiple chrome.exe processes show as `chrome`, `chrome#1`, `chrome#2`,
    ...) -- the `name` label below is that raw instance name, not the exact
    image filename process_attribution.py's psutil-based `name()` returns.
    Good enough for "which process", not byte-identical across platforms.

    Scope is deliberately narrow, matching M3's own framing ("write exactly
    one endpoint capability, not the agent"): a periodic top-N snapshot,
    not a live watchdog, and not conditioned on any local anomaly detection
    -- there is no local anomaly evaluation on this host. Correlating a
    resource spike here against Guardian Core's own Windows health/anomaly
    signals is Core's job, not this script's.

.NOTES
    Deploy: save this file locally on the Windows host, then register a
    Scheduled Task to run it every 15 minutes, same cadence as
    guardian_disk_health.ps1 (see OPERATIONS_MANUAL.md for the deployment
    procedure). 15 minutes is coarse for anything CPU/mem-transient -- a
    short-lived spike between runs will be missed. That's an accepted
    tradeoff for a first proof of concept, not a design ceiling; tighten
    the interval (or move to a lightweight always-running collector) once
    this capability's actually proven useful, not before.
#>

$ErrorActionPreference = 'Stop'

$OutputDir = 'C:\Program Files\windows_exporter\textfile_inputs'
$OutputFile = Join-Path $OutputDir 'guardian_process_attribution.prom'
$TempFile = "$OutputFile.tmp"
$TopN = 5

$lines = New-Object System.Collections.Generic.List[string]

function Add-Metric {
    param($Help, $Type, $Name, $Samples)
    $lines.Add("# HELP $Name $Help")
    $lines.Add("# TYPE $Name $Type")
    foreach ($s in $Samples) { $lines.Add($s) }
}

$cpuLines = @()
$memLines = @()

try {
    $counterPaths = '\Process(*)\% Processor Time', '\Process(*)\ID Process', '\Process(*)\Working Set'
    $samples = (Get-Counter -Counter $counterPaths -ErrorAction Stop).CounterSamples

    $totalMemBytes = (Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory

    $procs = @()
    foreach ($grp in ($samples | Group-Object InstanceName)) {
        if ($grp.Name -in @('_total', 'idle')) { continue }

        $idSample = $grp.Group | Where-Object { $_.Path -like '*\id process' } | Select-Object -First 1
        if (-not $idSample) { continue }
        $procPid = [int]$idSample.CookedValue
        if ($procPid -eq 0) { continue }

        $cpuSample = $grp.Group | Where-Object { $_.Path -like '*\% processor time' } | Select-Object -First 1
        $wsSample = $grp.Group | Where-Object { $_.Path -like '*\working set' } | Select-Object -First 1

        $cpuPct = if ($cpuSample) { [math]::Round($cpuSample.CookedValue, 2) } else { 0.0 }
        $memPct = if ($wsSample -and $totalMemBytes -gt 0) {
            [math]::Round(($wsSample.CookedValue / $totalMemBytes) * 100, 2)
        } else { 0.0 }

        $procs += [PSCustomObject]@{
            Pid  = $procPid
            Name = ($grp.Name -replace '"', '')
            Cpu  = $cpuPct
            Mem  = $memPct
        }
    }

    $topCpu = $procs | Sort-Object -Property Cpu -Descending | Select-Object -First $TopN
    foreach ($p in $topCpu) {
        $cpuLines += "windows_top_process_cpu_percent{pid=`"$($p.Pid)`",name=`"$($p.Name)`"} $($p.Cpu)"
    }

    $topMem = $procs | Sort-Object -Property Mem -Descending | Select-Object -First $TopN
    foreach ($p in $topMem) {
        $memLines += "windows_top_process_mem_percent{pid=`"$($p.Pid)`",name=`"$($p.Name)`"} $($p.Mem)"
    }
} catch {
    # Get-Counter / Get-CimInstance unavailable or failed on this host --
    # skip this cycle's snapshot rather than crash the scheduled task. The
    # freshness gauge below still updates so a run that hit this branch is
    # distinguishable from the task not running at all.
}

if ($cpuLines.Count -gt 0) {
    Add-Metric -Help "Top $TopN processes by CPU percent (single-core-relative, matches psutil's default -- not divided by logical processor count, so >100% is possible on a busy multi-threaded process)." -Type gauge -Name windows_top_process_cpu_percent -Samples $cpuLines
}
if ($memLines.Count -gt 0) {
    Add-Metric -Help "Top $TopN processes by memory percent (working set as a percent of total physical memory)." -Type gauge -Name windows_top_process_mem_percent -Samples $memLines
}

# --- Freshness gauge (same rationale as guardian_disk_health.ps1) ---
$epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
Add-Metric -Help "Unix timestamp of the last successful run of this collector." -Type gauge -Name windows_process_attribution_collector_last_run_timestamp_seconds -Samples @("windows_process_attribution_collector_last_run_timestamp_seconds $epoch")

# --- Write atomically (same pattern as guardian_disk_health.ps1) ---
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
$lines -join "`n" | Out-File -FilePath $TempFile -Encoding ascii -Force
Move-Item -Path $TempFile -Destination $OutputFile -Force
