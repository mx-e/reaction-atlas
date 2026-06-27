#!/bin/bash
set -e
if [ -n "$CLOUD_SQL_CONNECTION" ]; then
    echo "Starting Cloud SQL Auth Proxy → $CLOUD_SQL_CONNECTION"
    /cloud-sql-proxy --port=5432 --address=127.0.0.1 --private-ip "$CLOUD_SQL_CONNECTION" &
    sleep 3
    echo "Proxy started"
fi
exec python3 worker.py
