"""Post-exploration analysis functions.

Adapted from the original to work with DB-backed ReactionGraph.
Functions that previously accessed graph.graph (NetworkX) now query the DB.
"""

from loguru import logger


def print_energy_validation_statistics(graph):
    """Print statistics about energy validation for MIN <-> TS transitions."""
    stats = graph.energy_validation_stats
    logger.info("\nEnergy Validation Statistics:")
    logger.info("-" * 50)
    logger.info(f"MIN -> TS valid: {stats.get('min_to_ts_valid', 0)}")
    logger.info(f"MIN -> TS invalid: {stats.get('min_to_ts_invalid', 0)}")
    logger.info(f"TS -> MIN valid: {stats.get('ts_to_min_valid', 0)}")
    logger.info(f"TS -> MIN invalid: {stats.get('ts_to_min_invalid', 0)}")
    logger.info(f"Threshold violations: {stats.get('threshold_violations', 0)}")
    logger.info(f"Branches terminated: {stats.get('branches_terminated', 0)}")

    total = sum(stats.get(k, 0) for k in [
        "min_to_ts_valid", "min_to_ts_invalid", "ts_to_min_valid", "ts_to_min_invalid",
    ])
    if total > 0:
        valid = stats.get("min_to_ts_valid", 0) + stats.get("ts_to_min_valid", 0)
        logger.info(f"Overall validation success rate: {valid / total * 100:.1f}%")


def print_decomposition_statistics(graph):
    """Print statistics about decomposition validation."""
    stats = graph.decomposition_stats
    logger.info("\nDecomposition Validation Statistics:")
    logger.info(f"Total attempts: {stats.get('attempted', 0)}")
    logger.info(f"Single component: {stats.get('single_component', 0)}")
    logger.info(f"Successful: {stats.get('successful', 0)}")
    logger.info(f"Rejected: {stats.get('rejected_invalid_fragments', 0)}")

    if stats.get("attempted", 0) > 0:
        logger.info(f"Success rate: {stats['successful'] / stats['attempted'] * 100:.1f}%")


def analyze_decompositions(graph):
    """Analyze multi-product decompositions by querying the DB."""
    from packages.db.models import Reaction, ReactionProduct
    from sqlalchemy import func

    with graph.db.session() as session:
        # Count products per reaction
        product_counts = (
            session.query(ReactionProduct.reaction_id, func.count(ReactionProduct.id))
            .group_by(ReactionProduct.reaction_id)
            .all()
        )
        multi = sum(1 for _, count in product_counts if count > 1)
        single = sum(1 for _, count in product_counts if count == 1)
        logger.info(f"\nDecomposition Analysis: {single} single-product, {multi} multi-product reactions")


def print_exploration_statistics(graph):
    """Print comprehensive exploration statistics from DB."""
    from packages.db.models import ExplorationStats, BatchLog

    with graph.db.session() as session:
        stats_row = session.query(ExplorationStats).filter(ExplorationStats.id == 1).first()
        s = stats_row.stats_json if stats_row else {}

        batch_logs = session.query(BatchLog).all()
        batch_log = [b.summary_json for b in batch_logs]

    logger.info("\n" + "=" * 60)
    logger.info("EXPLORATION STATISTICS")
    logger.info("=" * 60)

    # Pipeline funnel
    submitted = s.get("pipeline_contexts_submitted", 0)
    funnel = [
        ("Contexts submitted", s.get("pipeline_contexts_submitted", 0)),
        ("Survived denoising", s.get("pipeline_survived_denoising", 0)),
        ("Passed fwd barrier", s.get("pipeline_passed_fwd_barrier", 0)),
        ("Passed IRC", s.get("pipeline_passed_irc", 0)),
        ("Passed fragmentation", s.get("pipeline_passed_fragmentation", 0)),
        ("Passed bwd barrier", s.get("pipeline_passed_bwd_barrier", 0)),
        ("Added to graph", s.get("pipeline_added_to_graph", 0)),
    ]
    logger.info("\nPipeline Funnel (generative):")
    for label, count in funnel:
        pct = f" ({count / submitted * 100:.1f}%)" if submitted > 0 else ""
        logger.info(f"  {label:.<30s} {count:>6d}{pct}")

    # Context type breakdown
    logger.info("\nContext Type Breakdown:")
    logger.info(f"  Merge submitted:  {s.get('merge_submitted', 0):>6d}   valid: {s.get('merge_valid', 0):>6d}")
    logger.info(f"  Single submitted: {s.get('single_submitted', 0):>6d}   valid: {s.get('single_valid', 0):>6d}")

    # IRC breakdown
    logger.info("\nIRC Breakdown:")
    for label, key in [
        ("Hessian pass", "irc_hessian_pass"),
        ("Hessian fail", "irc_hessian_fail"),
        ("Relax fwd fail", "irc_relax_fwd_fail"),
        ("Relax bwd fail", "irc_relax_bwd_fail"),
        ("Endpoints neither match", "irc_endpoints_neither_match"),
        ("Endpoints both match", "irc_endpoints_both_match"),
        ("Endpoints one match", "irc_endpoints_one_match"),
        ("Greedy fallback used", "irc_greedy_fallback_used"),
        ("Greedy accepted", "irc_greedy_accepted"),
    ]:
        logger.info(f"  {label:.<30s} {s.get(key, 0):>6d}")

    # Discovery methods
    logger.info("\nDiscovery Methods:")
    logger.info(f"  Generative:       {s.get('reactions_generative', 0):>6d}")
    logger.info(f"  PES exploration:  {s.get('reactions_pes_exploration', 0):>6d}")

    # PES exploration
    logger.info("\nPES Exploration:")
    logger.info(f"  Compounds explored:         {s.get('compounds_explored', 0):>6d}")
    logger.info(f"  Intramol TSs discovered:    {s.get('total_intramol_ts_discovered', 0):>6d}")
    logger.info(f"  Escaped reactions (total):  {s.get('total_escaped_reactions', 0):>6d}")
    logger.info(f"  Escaped reactions (valid):  {s.get('total_escaped_valid', 0):>6d}")
    total_pes_time = s.get("total_pes_time_s", 0.0)
    logger.info(f"  Total PES wall time:        {total_pes_time:>9.1f}s")

    # Noise histogram
    noise_hist = s.get("noise_histogram", {})
    if noise_hist:
        BIN_WIDTH = 50
        bins = {}
        for level, entry in noise_hist.items():
            bin_lo = (int(level) // BIN_WIDTH) * BIN_WIDTH
            if bin_lo not in bins:
                bins[bin_lo] = {"batches": 0, "reactions": 0}
            bins[bin_lo]["batches"] += entry["batches"]
            bins[bin_lo]["reactions"] += entry["reactions"]
        logger.info("\nNoise Level Histogram:")
        logger.info(f"  {'Range':>10s}  {'Batches':>8s}  {'Reactions':>10s}  {'React/Batch':>12s}")
        for bin_lo in sorted(bins.keys()):
            entry = bins[bin_lo]
            batches = entry["batches"]
            reactions = entry["reactions"]
            rpb = f"{reactions / batches:.2f}" if batches > 0 else "N/A"
            label = f"{bin_lo}-{bin_lo + BIN_WIDTH - 1}"
            logger.info(f"  {label:>10s}  {batches:>8d}  {reactions:>10d}  {rpb:>12s}")

    # Wall time breakdown
    batch_wall = sum(b.get("wall_time_s", 0.0) for b in batch_log) if batch_log else 0.0
    pes_wall = s.get("total_pes_time_s", 0.0)
    total_wall = batch_wall + pes_wall

    if total_wall > 0:
        steps = [
            ("PES: MD", s.get("total_pes_md_s", 0.0)),
            ("PES: PRFO", s.get("total_pes_prfo_s", 0.0)),
            ("PES: TS validation", s.get("total_pes_validation_s", 0.0)),
            ("PES: Escaped rxns", s.get("total_pes_escaped_s", 0.0)),
            ("PES: NEB dedup", s.get("total_pes_neb_dedup_s", 0.0)),
            ("Gen: Diffusion", s.get("total_gen_diffusion_s", 0.0)),
            ("Gen: Energy/filter", s.get("total_gen_energy_s", 0.0)),
            ("Gen: IRC", s.get("total_gen_irc_s", 0.0)),
            ("Gen: Fragmentation", s.get("total_gen_fragmentation_s", 0.0)),
            ("Gen: Graph add", s.get("total_gen_graph_add_s", 0.0)),
        ]
        tracked = sum(v for _, v in steps)
        steps.append(("Other", max(0.0, total_wall - tracked)))

        h, m = divmod(int(total_wall), 3600)
        m = m // 60
        logger.info(f"\nWall Time Breakdown (total {h}h {m}min):")
        for label, val in steps:
            pct = val / total_wall * 100
            sh, sm = divmod(int(val), 3600)
            sm = sm // 60
            logger.info(f"  {label:.<25s} {sh:>2d}h {sm:>2d}min  ({pct:>5.1f}%)")

    logger.info("=" * 60)


def print_edge_energies(graph):
    """Print edge energy statistics from DB."""
    from packages.db.models import Reaction, ReactionProduct, Compound
    from sqlalchemy import func

    with graph.db.session() as session:
        # Only two floats per row needed — don't hydrate full ORM rows
        # (Reaction has four LargeBinary columns that would otherwise
        # pull tens of GB for no reason).
        rows = session.query(Reaction.barrier_forward, Reaction.barrier_backward).all()
        logger.info(f"\nEdge Energy Summary: {len(rows)} reactions")

        barriers_fwd = [r[0] for r in rows]
        barriers_bwd = [r[1] for r in rows]

        if barriers_fwd:
            import numpy as np
            logger.info(f"  Forward barriers:  min={min(barriers_fwd):.3f}  "
                       f"max={max(barriers_fwd):.3f}  mean={np.mean(barriers_fwd):.3f} eV")
            logger.info(f"  Backward barriers: min={min(barriers_bwd):.3f}  "
                       f"max={max(barriers_bwd):.3f}  mean={np.mean(barriers_bwd):.3f} eV")

        # Count multi-product reactions
        multi = (
            session.query(ReactionProduct.reaction_id)
            .group_by(ReactionProduct.reaction_id)
            .having(func.count(ReactionProduct.id) > 1)
            .count()
        )
        logger.info(f"  Multi-product reactions: {multi}")
