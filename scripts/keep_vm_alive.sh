#!/usr/bin/env bash

set -u

interval_seconds="${KEEPALIVE_INTERVAL_SECONDS:-300}"

while true; do
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if command -v cancel_shutdown >/dev/null 2>&1; then
        cancel_shutdown >/dev/null 2>&1 || true
        printf '%s shutdown cancellation requested; ' "$timestamp"
    else
        printf '%s cancel_shutdown is unavailable; ' "$timestamp"
    fi

    if command -v time_left >/dev/null 2>&1; then
        time_left 2>/dev/null || true
    else
        printf 'time_left is unavailable\n'
    fi

    sleep "$interval_seconds"
done
