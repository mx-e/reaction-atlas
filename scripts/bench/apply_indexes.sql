-- Hot-path indexes for PES work queue.
--
-- Background: at 50+ workers the claim_pes_work query scans
-- pes_work_queue twice: once to find candidate rows (pending OR stale
-- in_progress) ordered by id, and once inside the NOT EXISTS subquery
-- to find "is this compound already being worked?". Neither had a
-- supporting index before these.
--
-- All indexes are partial / narrow to keep write overhead low; the
-- queue is write-heavy.

-- 1. Supports the NOT EXISTS subquery in claim_pes_work.
--    Probes (compound_id, status='in_progress', claimed_at >= X).
--    Also supports get_pes_backlog_in_progress count.
CREATE INDEX IF NOT EXISTS idx_pes_work_in_progress
  ON pes_work_queue (compound_id, claimed_at)
  WHERE status = 'in_progress';

-- 2. Supports the outer claim_pes_work_gated_scan ORDER BY id with
--    WHERE status IN ('pending', 'in_progress'). The existing
--    idx_pes_work_pending only covers status='pending' alone.
CREATE INDEX IF NOT EXISTS idx_pes_work_active
  ON pes_work_queue (id)
  WHERE status IN ('pending', 'in_progress');

-- Gather stats so the planner picks the new indexes.
ANALYZE pes_work_queue;

-- 3. Supports DFT claim's ORDER BY barrier_forward ASC over the active
--    (non-manual-equilibrium) subset of reactions. Without this PG seq-scans
--    the whole reactions table (1 GB, 2944 rows today) to find the sort key
--    for every DFT claim — ~2.2 ms per call at current scale, grows linearly.
CREATE INDEX IF NOT EXISTS idx_reactions_barrier_fwd_active
  ON reactions (barrier_forward)
  WHERE discovery_method IS DISTINCT FROM 'manual_equilibrium';

-- 4+5. Partial active indexes on the DFT/CREST queues — mirror the PES fix
--      so the claim queries don't degrade as completed rows accumulate.
--      Claims ORDER BY id and filter on status.
CREATE INDEX IF NOT EXISTS idx_dft_work_active
  ON dft_work_queue (id)
  WHERE status IN ('pending', 'in_progress');

CREATE INDEX IF NOT EXISTS idx_crest_work_active
  ON crest_work_queue (id)
  WHERE status IN ('pending', 'in_progress');

ANALYZE reactions;
ANALYZE dft_work_queue;
ANALYZE crest_work_queue;
