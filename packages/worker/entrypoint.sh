#!/bin/bash
# Entrypoint: start Cloud SQL Auth Proxy if CLOUD_SQL_CONNECTION is set,
# then run the worker. Proxy runs in background and dies with the container.
set -e

if [ -n "$CLOUD_SQL_CONNECTION" ]; then
    echo "Starting Cloud SQL Auth Proxy → $CLOUD_SQL_CONNECTION"
    /cloud-sql-proxy --port=5432 --address=127.0.0.1 --private-ip "$CLOUD_SQL_CONNECTION" &
    sleep 3  # wait for proxy to be ready
    echo "Proxy started"
fi

exec python -X faulthandler worker.py
