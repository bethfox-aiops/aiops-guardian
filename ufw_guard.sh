#!/bin/bash
# ufw_guard.sh
#
# Wrapper the passwordless sudo entry for ufw should point at, instead of
# the raw /usr/sbin/ufw binary directly. Same governance pattern as
# trace_suspect.sh: sudoers scopes which *script* runs as root, this
# script scopes what it's actually allowed to do.
#
# The gap this closes: NOPASSWD access to raw ufw had no restriction on
# subcommand at all, so anything running as beth (an AI agent session,
# or anything else) could passwordlessly run `ufw disable` or
# `ufw --force reset` -- wiping out the exact DENY rules OPERATIONS_MANUAL.md
# Chapter 3.9/7.3 documents as the *only* thing keeping the watchdog ports
# (8011-8014, 8016-8018) non-external. Real add/delete-rule usage (this
# session used passwordless ufw for the temporary file-transfer rule and
# the 8016-8018 fix) is preserved -- only the specific subcommands with no
# legitimate use case here are blocked.
set -euo pipefail

UFW=/usr/sbin/ufw

# Deliberately a denylist, not an allowlist: blocks the specific
# subcommands with zero legitimate use in this project's real workflow
# (verified via a grep of every automated ufw caller plus this session's
# own history) rather than trying to enumerate every valid one. Known
# tradeoff: a denylist fails open against a subcommand not thought of here
# -- worth revisiting as an allowlist if ufw's own subcommand set changes
# or this ever runs somewhere less trusted than a single-user box.
DENYLIST=(disable reset default)

subcommand=""
for arg in "$@"; do
    case "$arg" in
        --*) continue ;;  # skip global flags like --force, --dry-run
        *) subcommand="$arg"; break ;;
    esac
done

for blocked in "${DENYLIST[@]}"; do
    if [ "$subcommand" = "$blocked" ]; then
        echo "ERROR: 'ufw $subcommand' is blocked via passwordless sudo -- disabling the firewall," \
             "resetting all rules, or changing the default policy requires an interactive" \
             "'sudo ufw ...' with a password, not this route." >&2
        exit 1
    fi
done

exec "$UFW" "$@"
