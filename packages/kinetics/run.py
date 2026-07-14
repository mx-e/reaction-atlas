"""CLI: solve the kinetics ODE for a populated reaction-network database.

Connects to a PostgreSQL database — a live exploration DB (see
docs/reproducing.md) or one restored from the Zenodo dump (see docs/data.md) —
builds the mass-action ODE model from the reaction graph, integrates it, and
prints the resulting steady-state distribution. This is the same code path the
background solver runs inside the (separately hosted) API service; here it is
exposed as a one-shot command.

    export DATABASE_URL=postgresql://crn:crn@localhost:5432/crn_cloud
    uv run --extra db python -m packages.kinetics.run
    uv run --extra db python -m packages.kinetics.run --temperature 500 --experiment main

For a self-contained, database-free demonstration of the same solver on a small
shipped network, see demo/kinetics/run_demo.py.
"""
import argparse
import sys

from packages.db.connection import get_session
from packages.kinetics.build import build_snapshot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--database-url", default=None,
                    help="SQLAlchemy URL; defaults to the DATABASE_URL environment variable.")
    ap.add_argument("--temperature", type=float, default=500.0,
                    help="Eyring temperature in Kelvin (default: 500).")
    ap.add_argument("--experiment", default=None,
                    help="Restrict to reactions tagged with this experiment (default: all).")
    ap.add_argument("--t-max", type=float, default=1e8,
                    help="Integration horizon in seconds (default: 1e8).")
    ap.add_argument("--no-prefer-dft", action="store_true",
                    help="Prefer ML in-box barriers over separated PBE0 barriers.")
    ap.add_argument("--top", type=int, default=20,
                    help="How many steady-state species to print (default: 20).")
    args = ap.parse_args(argv)

    session = get_session(args.database_url)
    try:
        snap = build_snapshot(
            session,
            temperature=args.temperature,
            prefer_dft=not args.no_prefer_dft,
            t_max=args.t_max,
            experiment=args.experiment,
        )
    finally:
        session.close()

    if snap is None:
        print("Model has 0 usable reactions — nothing to solve "
              "(is the database populated?).", file=sys.stderr)
        return 1

    scope = args.experiment or "all experiments"
    print(f"Kinetics solve ({scope}, T={args.temperature:.0f} K):")
    print(f"  species                 : {snap.n_species}")
    print(f"  reactions               : {snap.n_reactions} "
          f"({snap.n_manual_equilibria} equilibria, {snap.n_reactions_dft} DFT)")
    print(f"  solver wall time        : {snap.solve_wall_time_s:.2f}s")

    ranked = sorted(snap.steady_state_distribution.items(),
                    key=lambda kv: kv[1], reverse=True)
    print(f"  steady-state top {min(args.top, len(ranked))} of {len(ranked)} active species:")
    for smi, w in ranked[:args.top]:
        print(f"    {w:10.4f}   {smi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
