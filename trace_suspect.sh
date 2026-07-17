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
PID="${1:-}"

if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PID must be numeric" >&2
    exit 1
fi

if [ ! -d "/proc/$PID" ]; then
    echo "ERROR: no such process: $PID" >&2
    exit 1
fi

exec timeout "$DURATION" bpftrace -e '
tracepoint:syscalls:sys_enter_openat /pid == '"$PID"'/ { printf("OPEN %d %s\n", pid, str(args->filename)); }
tracepoint:syscalls:sys_enter_execve /pid == '"$PID"'/ { printf("EXEC %d %s\n", pid, str(args->filename)); }
tracepoint:syscalls:sys_enter_connect /pid == '"$PID"'/ { printf("CONNECT %d\n", pid); }
tracepoint:syscalls:sys_enter_write /pid == '"$PID"'/ { printf("WRITE %d fd=%d len=%d\n", pid, args->fd, args->count); }
'
