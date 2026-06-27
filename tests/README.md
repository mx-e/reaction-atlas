# Verification Suite

## Quick Start (macOS with Docker)

```bash
# 1. Start PostgreSQL
docker compose up -d db

# 2. Install test deps
pip install -r tests/requirements.txt

# 3. Run DB + serialization + work queue tests
python tests/verify_all.py

# 4. (Optional) Also test API server
docker compose up -d db api
python tests/verify_all.py --api

# 5. Tear down
docker compose down -v
```

## Integration Test

```bash
# Full integration: concurrent workers + DB invariants + API verification
docker compose up -d db api
python tests/test_integration.py
```

Simulates 5 concurrent workers competing for PES work and generating
reactions in parallel via threads. Verifies:
- No duplicate SMILES, minima, or reactions (concurrent INSERT safety)
- Work queue fully drained (FOR UPDATE SKIP LOCKED correctness)
- No orphan compounds or dangling graph edges
- API returns correct data for the generated graph
- Stats accumulated correctly across workers

## What's tested

| Suite | Tests | What it verifies |
|-------|-------|-----------------|
| Serialization | 4 | numpy ↔ bytea round-trips, trajectory npz compression |
| Schema | 3 | Table creation, indexes, unique constraints on real PostgreSQL |
| CRUD | 4 | Compounds, reactions, graph edges, annotations with relationships |
| Work Queue | 2 | `FOR UPDATE SKIP LOCKED` atomic claiming, concurrent worker simulation |
| DB Layer | 5 | `lib/db.py` end-to-end: compound lifecycle, reactions, work queue, stats accumulation |
| API | 11 | All major endpoints with seeded test data |

## Environment Variables

- `DATABASE_URL` — PostgreSQL connection (default: `postgresql://crn:crn@localhost:5432/crn_cloud`)
- `API_URL` — API server URL (default: `http://localhost:8080`)
