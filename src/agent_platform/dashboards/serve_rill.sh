#!/bin/sh
# Run Rill so that one Ctrl+C stops it at once.
#
# Rill's own graceful shutdown waits ~20s on its background reconcilers. The project is
# re-ingested from the Parquet export on every start (`--reset`), so there is no state to
# protect: on Ctrl+C (or TERM) Rill is stopped outright and the shell returns.
"$@" &
pid=$!
trap 'kill -9 "$pid" 2>/dev/null; exit 0' INT TERM
wait "$pid"
