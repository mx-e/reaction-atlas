"""
Test spot preemption handling end-to-end.

Starts a worker subprocess, waits for it to claim PES work,
sends SIGTERM, and verifies the work item was released back to 'pending'.

Runs inside the worker Docker container to use existing deps.

Usage:
    docker compose up -d db
    docker compose run --rm -e PES_MD_STEPS=500 -e PES_MAX_ITERATIONS=3 \
        worker python /app/test_preemption.py
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

# Use the worker's DB module directly since we're in the same environment
sys.path.insert(0, "/app")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://crn:crn@db:5432/crn_cloud")


def get_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def query_work_queue(conn):
    """Return all PES work queue items."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, compound_id, status, worker_id "
            "FROM pes_work_queue ORDER BY id"
        )
        rows = cur.fetchall()
    conn.commit()
    return rows


def ensure_pending_work(conn):
    """Make sure there's at least one pending PES work item."""
    rows = query_work_queue(conn)
    pending = [r for r in rows if r[2] == "pending"]
    if pending:
        print(f"Found {len(pending)} pending work items")
        return

    print("No pending work — inserting test items...")
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pes_work_queue (compound_id, minimum_id, status)
            SELECT c.id, m.id, 'pending'
            FROM compounds c
            JOIN minima m ON m.compound_id = c.id
            WHERE c.n_atoms >= 4
              AND NOT EXISTS (SELECT 1 FROM pes_work_queue q WHERE q.minimum_id = m.id)
            LIMIT 5
            RETURNING id, compound_id, status
        """)
        inserted = cur.fetchall()
        conn.commit()
        print(f"Inserted {len(inserted)} pending work items")
        if not inserted:
            print("ERROR: No suitable compounds/minima to create work items")
            sys.exit(1)


def wait_for_status(conn, target_status, timeout=180):
    """Wait until at least one work item has the target status."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = query_work_queue(conn)
        matched = [r for r in rows if r[2] == target_status]
        if matched:
            return matched[0]
        time.sleep(2)
    return None


def main():
    print("=" * 60)
    print("SPOT PREEMPTION TEST")
    print("=" * 60)

    # 1. Connect to DB
    print("\n[1] Connecting to database...")
    conn = get_conn()
    print("Connected.")

    # 2. Ensure pending work exists
    print("\n[2] Checking work queue...")
    ensure_pending_work(conn)

    rows = query_work_queue(conn)
    print(f"Queue state ({len(rows)} items):")
    for r in rows:
        print(f"  id={r[0]} compound={r[1]} status={r[2]} worker={r[3]}")

    # 3. Start worker subprocess
    print("\n[3] Starting worker subprocess...")
    env = {**os.environ, "DATABASE_URL": DATABASE_URL, "PYTHONPATH": "/app"}
    worker = subprocess.Popen(
        [sys.executable, "worker.py"],
        cwd="/app",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"Worker PID: {worker.pid}")

    # 4. Wait for work to become in_progress
    print("\n[4] Waiting for worker to claim PES work...")
    claimed = wait_for_status(conn, "in_progress", timeout=180)

    if claimed is None:
        print("TIMEOUT: No work became in_progress within 180s")
        print("\nWorker output:")
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        output = worker.stdout.read() or ""
        for line in output.split("\n")[-40:]:
            print(f"  {line}")
        sys.exit(1)

    work_id = claimed[0]
    print(f"Worker claimed work_id={work_id} compound_id={claimed[1]}")

    # 5. Wait a bit to let computation start
    print("\n[5] Letting computation run for 8 seconds...")
    time.sleep(8)

    # Verify still in_progress
    rows = query_work_queue(conn)
    item = next((r for r in rows if r[0] == work_id), None)
    if item and item[2] == "in_progress":
        print(f"Confirmed: work_id={work_id} still in_progress")
    elif item and item[2] == "completed":
        print(f"Work completed before we could send SIGTERM (computation was fast)")
        print("Try setting PES_MD_STEPS=2000 for a slower job")
        worker.wait(timeout=30)
        sys.exit(0)
    else:
        print(f"Unexpected state: {item}")

    # 6. Send SIGTERM
    print(f"\n[6] Sending SIGTERM to worker PID {worker.pid}...")
    t_signal = time.time()
    worker.send_signal(signal.SIGTERM)

    # 7. Wait for exit
    print("Waiting for worker to exit...")
    try:
        worker.wait(timeout=30)
        dt = time.time() - t_signal
        print(f"Worker exited (code={worker.returncode}) in {dt:.1f}s")
    except subprocess.TimeoutExpired:
        print("Worker did not exit in 30s — killing")
        worker.kill()
        worker.wait()

    # 8. Print tail of worker output
    output = worker.stdout.read() or ""
    lines = output.strip().split("\n")
    print(f"\nWorker output (last 25 lines of {len(lines)}):")
    for line in lines[-25:]:
        print(f"  {line}")

    # 9. Verify work was released
    print(f"\n[7] Checking work queue after SIGTERM...")
    conn.close()
    conn = get_conn()
    rows = query_work_queue(conn)

    print(f"Queue state ({len(rows)} items):")
    for r in rows:
        marker = " <-- TARGET" if r[0] == work_id else ""
        print(f"  id={r[0]} compound={r[1]} status={r[2]} worker={r[3]}{marker}")

    item = next((r for r in rows if r[0] == work_id), None)
    if item is None:
        print(f"\nFAIL: work_id={work_id} disappeared from queue")
        sys.exit(1)

    status, worker_field = item[2], item[3]

    print(f"\n{'=' * 60}")
    if status == "pending" and worker_field is None:
        print("PASS: Work released back to 'pending', worker_id cleared")
        print("Another worker can immediately pick this up.")
    elif status == "pending":
        print(f"PARTIAL PASS: Status is 'pending' but worker_id='{worker_field}' not cleared")
    elif status == "completed":
        print("INFO: Work was completed before SIGTERM took effect")
    elif status == "in_progress":
        print("FAIL: Work is still 'in_progress' — SIGTERM handler did not release it")
        print(f"It will be reclaimed after WORK_TIMEOUT_S ({3600}s) by another worker")
        sys.exit(1)
    else:
        print(f"FAIL: Unexpected status '{status}'")
        sys.exit(1)
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
