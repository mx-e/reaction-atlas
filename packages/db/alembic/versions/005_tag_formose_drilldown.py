"""Tag the formose-drilldown subset.

Data-only migration (no schema changes). Adds 'formose-drilldown' to the
`experiments` arrays of compounds, minima, intra_transition_states, reactions,
and graph_edges that participate in the user-curated formose subgraph.

Subgraph membership is sourced from the `annotations` table, which carries
free-form 'compound_tags' rows whose `label` field is a comma-separated
string. A compound is in the subgraph if any of its tags equals one of:
   - core_formose
   - core_formose_tautomer
   - extended_formose
   - formaldehyde_pool

Note: 'core_formose' is a strict prefix of 'core_formose_tautomer', so we
split the comma-separated label and exact-match each piece — using LIKE
would over-match.

Cascade rules:
  - all minima of a tagged compound → tagged
  - all intra_transition_states of a tagged compound → tagged
  - reactions where every reactant AND every product compound is tagged
  - graph_edges whose reaction is tagged
  - is_seed=True compounds (buffer chemistry — water/HCHO/etc. needed for
    kinetics convergence) and their derived minima/TSs
  - manual_equilibrium reactions (water autoionization etc.) + their
    graph_edges, when all participants are now tagged

Idempotent: every UPDATE has a `NOT (:tag = ANY(experiments))` guard so
re-running this migration is a no-op.

Revision ID: 005
Revises: 004
Create Date: 2026-05-06
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text


revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TAG = "formose-drilldown"
# Tag values in the DB are namespaced under auto:group:* in the
# comma-separated `label` field of compound_tags annotations.
_LABEL_AUTO_GROUP = (
    "auto:group:core_formose",
    "auto:group:core_formose_tautomer",
    "auto:group:extended_formose",
    "auto:group:formaldehyde_pool",
)


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Compounds: tag every compound whose annotations.label (split on
    #    comma + trimmed) contains one of the four membership tags.
    #    IN clause unrolled to four explicit bindparams since
    #    sqlalchemy.text() does not expand tuple parameters by default.
    bind.execute(
        text("""
            UPDATE compounds c
               SET experiments = c.experiments || ARRAY[:tag]
              FROM annotations a
             WHERE a.entity_type = 'compound_tags'
               AND a.entity_key = c.smiles
               AND EXISTS (
                   SELECT 1
                     FROM regexp_split_to_table(a.label, ',\\s*') AS t(tag)
                    WHERE trim(t.tag) IN (:l1, :l2, :l3, :l4)
               )
               AND NOT (:tag = ANY(c.experiments))
        """),
        {
            "tag": _TAG,
            "l1": _LABEL_AUTO_GROUP[0],
            "l2": _LABEL_AUTO_GROUP[1],
            "l3": _LABEL_AUTO_GROUP[2],
            "l4": _LABEL_AUTO_GROUP[3],
        },
    )

    # 2. is_seed=True compounds — buffer / fragment chemistry needed for
    #    kinetics convergence (water, HCHO, hydroxide, hydronium, etc.).
    bind.execute(
        text("""
            UPDATE compounds
               SET experiments = experiments || ARRAY[:tag]
             WHERE is_seed = true
               AND NOT (:tag = ANY(experiments))
        """),
        {"tag": _TAG},
    )

    # 3. Minima of tagged compounds.
    bind.execute(
        text("""
            UPDATE minima m
               SET experiments = m.experiments || ARRAY[:tag]
              FROM compounds c
             WHERE c.id = m.compound_id
               AND :tag = ANY(c.experiments)
               AND NOT (:tag = ANY(m.experiments))
        """),
        {"tag": _TAG},
    )

    # 4. Intra-TSs of tagged compounds.
    bind.execute(
        text("""
            UPDATE intra_transition_states its
               SET experiments = its.experiments || ARRAY[:tag]
              FROM compounds c
             WHERE c.id = its.compound_id
               AND :tag = ANY(c.experiments)
               AND NOT (:tag = ANY(its.experiments))
        """),
        {"tag": _TAG},
    )

    # 5. Reactions where EVERY reactant AND EVERY product compound is tagged,
    #    AND has at least one reactant + at least one product (defensive
    #    against mid-write rows).
    bind.execute(
        text("""
            UPDATE reactions r
               SET experiments = r.experiments || ARRAY[:tag]
             WHERE NOT (:tag = ANY(r.experiments))
               AND EXISTS (
                   SELECT 1 FROM reaction_reactants WHERE reaction_id = r.id
               )
               AND EXISTS (
                   SELECT 1 FROM reaction_products WHERE reaction_id = r.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM reaction_reactants rr
                     JOIN compounds c ON c.id = rr.compound_id
                    WHERE rr.reaction_id = r.id
                      AND NOT (:tag = ANY(c.experiments))
               )
               AND NOT EXISTS (
                   SELECT 1 FROM reaction_products rp
                     JOIN compounds c ON c.id = rp.compound_id
                    WHERE rp.reaction_id = r.id
                      AND NOT (:tag = ANY(c.experiments))
               )
        """),
        {"tag": _TAG},
    )

    # 6. Manual-equilibrium reactions: same gate (all participants tagged).
    #    Step 5 already covers them since the WHERE doesn't exclude
    #    manual_equilibrium — kept here as a sanity comment, no extra
    #    SQL needed.

    # 7. Graph edges that point to a tagged reaction.
    bind.execute(
        text("""
            UPDATE graph_edges ge
               SET experiments = ge.experiments || ARRAY[:tag]
              FROM reactions r
             WHERE r.id = ge.reaction_id
               AND :tag = ANY(r.experiments)
               AND NOT (:tag = ANY(ge.experiments))
        """),
        {"tag": _TAG},
    )


def downgrade() -> None:
    """Strip 'formose-drilldown' from every experiments array."""
    bind = op.get_bind()
    for tbl in (
        "compounds", "minima", "intra_transition_states",
        "reactions", "graph_edges", "annotations", "saved_layouts",
    ):
        bind.execute(
            text(
                f"UPDATE {tbl} SET experiments = array_remove(experiments, :tag) "
                f"WHERE :tag = ANY(experiments)"
            ),
            {"tag": _TAG},
        )
