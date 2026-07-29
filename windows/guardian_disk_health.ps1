<#
.SYNOPSIS
    Guardian disk-health collector for windows_exporter's textfile collector.

.DESCRIPTION
    windows_exporter's built-in collectors (cpu, logical_disk, memory, net,
    os, physical_disk, service, system) expose performance counters, not
    SMART/reliability data -- there is no signal today for wear, read/write
    errors, or disk/controller-level Event Log entries, all of which are the
    actual textbook indicators of a failing drive. This script closes that
    gap without a new service or port: it writes a standard Prometheus
    textfile-collector .prom file that windows_exporter's existing
    `textfile` collector picks up and re-exposes on its own /metrics
    endpoint, so it flows through the pipeline already scraped by
    Prometheus and already on the Windows dashboard (see
    OPERATIONS_MANUAL.md Chapter 3.3.1, "Guardian Metrics - Cross Platform
    - Windows").

    Deliberately narrow scope: this is a disk-health diagnostic, not the
    general "Guardian Endpoint Agent" described in EDGE_ARCHITECTURE.md
    (which would run arbitrary approved checks and push evidence to a
    Raspberry Pi Edge Collector). No Pi exists yet. This script only ever
    writes a local file for windows_exporter to read -- it never makes an
    outbound connection itself.

.NOTES
    Deploy: save this file locally on the Windows host, then register a
    Scheduled Task to run it every 15 minutes (matching this project's
    other lightweight polling intervals, e.g. disk_watchdog.timer on the
    Linux side). See the deployment notes in OPERATIONS_MANUAL.md.
#>

$ErrorActionPreference = 'Stop'

$OutputDir = 'C:\Program Files\windows_exporter\textfile_inputs'
$OutputFile = Join-Path $OutputDir 'guardian_disk_health.prom'
$TempFile = "$OutputFile.tmp"

$lines = New-Object System.Collections.Generic.List[string]

function Add-Metric {
    param($Help, $Type, $Name, $Samples)
    $lines.Add("# HELP $Name $Help")
    $lines.Add("# TYPE $Name $Type")
    foreach ($s in $Samples) { $lines.Add($s) }
}

# --- SMART / reliability counters ---
# Not every disk/controller/driver stack reports every field (or any field
# at all) -- older SATA controllers in particular often expose nothing via
# Get-StorageReliabilityCounter. Each metric is only emitted for disks that
# actually report that specific value, rather than emitting a fake zero.
$wear = @(); $readErr = @(); $writeErr = @(); $readErrUncorr = @(); $writeErrUncorr = @(); $temp = @()

try {
    $disks = Get-PhysicalDisk
    foreach ($disk in $disks) {
        $labelId = $disk.DeviceId
        $labelName = ($disk.FriendlyName -replace '"', '').Trim()
        try {
            $counter = Get-StorageReliabilityCounter -PhysicalDisk $disk -ErrorAction Stop
        } catch {
            continue
        }
        if ($null -ne $counter.Wear) {
            $wear += "windows_disk_reliability_wear_percent{disk=`"$labelId`",device=`"$labelName`"} $($counter.Wear)"
        }
        if ($null -ne $counter.ReadErrorsTotal) {
            $readErr += "windows_disk_reliability_read_errors_total{disk=`"$labelId`",device=`"$labelName`"} $($counter.ReadErrorsTotal)"
        }
        if ($null -ne $counter.WriteErrorsTotal) {
            $writeErr += "windows_disk_reliability_write_errors_total{disk=`"$labelId`",device=`"$labelName`"} $($counter.WriteErrorsTotal)"
        }
        if ($null -ne $counter.ReadErrorsUncorrected) {
            $readErrUncorr += "windows_disk_reliability_read_errors_uncorrected_total{disk=`"$labelId`",device=`"$labelName`"} $($counter.ReadErrorsUncorrected)"
        }
        if ($null -ne $counter.WriteErrorsUncorrected) {
            $writeErrUncorr += "windows_disk_reliability_write_errors_uncorrected_total{disk=`"$labelId`",device=`"$labelName`"} $($counter.WriteErrorsUncorrected)"
        }
        if ($null -ne $counter.Temperature) {
            $temp += "windows_disk_reliability_temperature_celsius{disk=`"$labelId`",device=`"$labelName`"} $($counter.Temperature)"
        }
    }
} catch {
    # Get-PhysicalDisk/Get-StorageReliabilityCounter unavailable on this host
    # (older Windows, non-Storage-Spaces-capable driver stack, etc.) -- skip
    # this section rather than fail the whole collector; the event-log
    # section below still runs and still provides real signal.
}

if ($wear.Count -gt 0) { Add-Metric -Help "SSD wear indicator, percent of rated life used (0-100). Absent for disks/controllers that don't report it." -Type gauge -Name windows_disk_reliability_wear_percent -Samples $wear }
if ($readErr.Count -gt 0) { Add-Metric -Help "Total read errors (corrected + uncorrected) reported by the storage reliability counter." -Type gauge -Name windows_disk_reliability_read_errors_total -Samples $readErr }
if ($writeErr.Count -gt 0) { Add-Metric -Help "Total write errors (corrected + uncorrected) reported by the storage reliability counter." -Type gauge -Name windows_disk_reliability_write_errors_total -Samples $writeErr }
if ($readErrUncorr.Count -gt 0) { Add-Metric -Help "Uncorrected read errors -- the more serious subset of read errors." -Type gauge -Name windows_disk_reliability_read_errors_uncorrected_total -Samples $readErrUncorr }
if ($writeErrUncorr.Count -gt 0) { Add-Metric -Help "Uncorrected write errors -- the more serious subset of write errors." -Type gauge -Name windows_disk_reliability_write_errors_uncorrected_total -Samples $writeErrUncorr }
if ($temp.Count -gt 0) { Add-Metric -Help "Disk temperature in Celsius, if reported by hardware." -Type gauge -Name windows_disk_reliability_temperature_celsius -Samples $temp }

# --- Disk/controller-related Event Log entries, rolling 24h window ---
# 7   = bad block encountered on the disk
# 11  = the driver detected a controller error
# 15  = the disk is not ready for access yet
# 51  = delayed write failure (a classic, well-known failing-drive signal)
# 153 = an IO operation was retried / IO device error
$eventIds = 7, 11, 15, 51, 153
$eventCounts = @()
foreach ($id in $eventIds) {
    try {
        $count = (Get-WinEvent -FilterHashtable @{ LogName = 'System'; Id = $id; StartTime = (Get-Date).AddHours(-24) } -ErrorAction SilentlyContinue | Measure-Object).Count
    } catch {
        $count = 0
    }
    $eventCounts += "windows_disk_event_count{event_id=`"$id`"} $count"
}
Add-Metric -Help "Disk/controller-related System event log entries in the last 24h, by Event ID (7=bad block, 11=controller error, 15=disk not ready, 51=delayed write failure, 153=IO device error)." -Type gauge -Name windows_disk_event_count -Samples $eventCounts

# --- Freshness gauge ---
# Standard practice for the textfile collector: without this, a stuck or
# failing scheduled task looks identical to a healthy one -- the last-known
# .prom file just keeps getting re-scraped forever with no way to tell it's
# stale. Compare against time() in Prometheus/Grafana to detect that case.
$epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
Add-Metric -Help "Unix timestamp of the last successful run of this collector." -Type gauge -Name windows_disk_health_collector_last_run_timestamp_seconds -Samples @("windows_disk_health_collector_last_run_timestamp_seconds $epoch")

# --- Write atomically ---
# windows_exporter's textfile collector reads whatever is in the directory
# on its own schedule -- writing the real filename directly risks it
# scraping a half-written file mid-write. Write to a temp file, then move
# it into place (a rename, not a copy, so it's atomic on the same volume).
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
$lines -join "`n" | Out-File -FilePath $TempFile -Encoding ascii -Force
Move-Item -Path $TempFile -Destination $OutputFile -Force
