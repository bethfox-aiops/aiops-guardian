#!/bin/bash
#
# trace_suspect.sh
#
# Behavioral Attestation Phase 2: scoped eBPF trace of a single suspect
# process. Invoked via sudo (see /etc/sudoers.d/aiops-trace) by the
# unprivileged watchdog services when Phase 1 process attribution flags a
# suspect PID, so we can see what that process actually did at the
# syscall/file/network level during the anomaly window.
#
# Usage: sudo trace_suspect.sh <pid>
#
# Duration is fixed here, not caller-controlled, to bound the cost/risk of
# this privileged action regardless of what calls it.

set -euo pipefail

DURATION=3
TICKET_MAX_AGE=2
TICKET_FILE="/home/beth/aiops-agents/.trace_ticket"
PID="${1:-}"

if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PID must be numeric" >&2
    exit 1
fi

if [ ! -d "/proc/$PID" ]; then
    echo "ERROR: no such process: $PID" >&2
    exit 1
fi

# The sudoers NOPASSWD rule for this script has no way to scope *which* PID
# is legitimate to trace (see ebpf_trace.py's TICKET_FILE comment for why
# PID ownership can't be used to scope it either -- promtail, this feature's
# original real-world catch, runs as root). Require a fresh ticket written
# by ebpf_trace.py immediately before it called sudo, naming this exact PID,
# so a direct "sudo trace_suspect.sh <pid>" invocation from anywhere else on
# the box -- bypassing Guardian's own attribution logic entirely -- fails
# here instead of silently getting root-level eBPF access to any process.
if [ ! -f "$TICKET_FILE" ]; then
    echo "ERROR: no trace ticket -- refusing untracked trace request" >&2
    exit 1
fi

read -r TICKET_PID TICKET_TIME < "$TICKET_FILE"

if [ "$TICKET_PID" != "$PID" ]; then
    echo "ERROR: ticket is for a different PID -- refusing" >&2
    exit 1
fi

NOW=$(date +%s)
TICKET_AGE=$(awk -v now="$NOW" -v t="$TICKET_TIME" 'BEGIN { print now - t }')
if (( $(awk -v age="$TICKET_AGE" -v max="$TICKET_MAX_AGE" 'BEGIN { print (age > max) }') )); then
    echo "ERROR: trace ticket expired -- refusing" >&2
    exit 1
fi

exec timeout "$DURATION" bpftrace -e '
tracepoint:syscalls:sys_enter_openat /pid == '"$PID"'/ { printf("OPEN %d %s\n", pid, str(args->filename)); }
tracepoint:syscalls:sys_enter_execve /pid == '"$PID"'/ { printf("EXEC %d %s\n", pid, str(args->filename)); }
tracepoint:syscalls:sys_enter_connect /pid == '"$PID"'/ { printf("CONNECT %d\n", pid); }
tracepoint:syscalls:sys_enter_write /pid == '"$PID"'/ { printf("WRITE %d fd=%d len=%d\n", pid, args->fd, args->count); }
'
