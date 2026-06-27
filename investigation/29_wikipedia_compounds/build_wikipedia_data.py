"""Generate packages/frontend/src/lib/wikipediaData.ts from wiki_wikidata_v3.csv.

The CSV is curated upstream in
  /home/sgugler/Research/dyanamical_system_tests/36_isomer_enumeration/
and contains every Wikidata chemical compound up to ~C4O4 (810 rows), with a
boolean `on_enwiki` flag and an `enwiki_title` for those that have an English
Wikipedia article (294 rows).

Output:
  WIKIPEDIA_CANON_SMILES — Set<string> of every entry's RDKit-canonical
                          (no-stereo) SMILES.  Used by the "wikipedia" color
                          mode and the auto:source:wikipedia tag.

  WIKIPEDIA_NAMES        — canon_smiles -> display name.  Prefers
                          enwiki_title over wikidata_name.

  WIKIPEDIA_HAS_ARTICLE  — Set<string> of canon_smiles for entries with an
                          actual enwiki article (used by the labeling script
                          to decide what to write into the compounds
                          annotation table).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

CSV = Path("/home/sgugler/Research/dyanamical_system_tests/36_isomer_enumeration/wiki_wikidata_v3.csv")
TS_OUT = Path(__file__).resolve().parents[2] / "packages/frontend/src/lib/wikipediaData.ts"


def main():
    rows = list(csv.DictReader(CSV.open()))
    canon_set: set[str] = set()
    names: dict[str, str] = {}
    has_article: set[str] = set()

    for r in rows:
        canon = r["canon_smiles"]
        if not canon:
            continue
        canon_set.add(canon)
        on_enwiki = r["on_enwiki"] == "True"
        title = r["enwiki_title"].strip()
        wd_name = r["wikidata_name"].strip()
        if on_enwiki and title:
            has_article.add(canon)
            names[canon] = title
        elif wd_name:
            names[canon] = wd_name

    canon_sorted = sorted(canon_set)
    has_article_sorted = sorted(has_article)
    names_pairs = sorted(names.items())

    print(f"Source: {CSV}")
    print(f"  rows:                     {len(rows)}")
    print(f"  unique canonical SMILES:  {len(canon_sorted)}")
    print(f"  with enwiki article:      {len(has_article_sorted)}")
    print(f"  with display name:        {len(names_pairs)}")
    print(f"Writing -> {TS_OUT}")

    body = []
    body.append("// AUTO-GENERATED from investigation/29_wikipedia_compounds/build_wikipedia_data.py")
    body.append(f"// Source: {CSV.name}  ({len(canon_sorted)} compounds, {len(has_article_sorted)} on enwiki)")
    body.append("// Re-run the build script when the upstream CSV changes.")
    body.append("")
    body.append("/** Every Wikidata-listed compound (RDKit canonical, no stereo). */")
    body.append("export const WIKIPEDIA_CANON_SMILES: ReadonlySet<string> = new Set([")
    for s in canon_sorted:
        body.append(f"  {json.dumps(s)},")
    body.append("]);")
    body.append("")
    body.append("/** Subset of the above that has an actual English-Wikipedia article. */")
    body.append("export const WIKIPEDIA_HAS_ARTICLE: ReadonlySet<string> = new Set([")
    for s in has_article_sorted:
        body.append(f"  {json.dumps(s)},")
    body.append("]);")
    body.append("")
    body.append("/** canon_smiles -> display name (enwiki_title preferred, wikidata_name fallback). */")
    body.append("export const WIKIPEDIA_NAMES: Readonly<Record<string, string>> = {")
    for s, name in names_pairs:
        body.append(f"  {json.dumps(s)}: {json.dumps(name)},")
    body.append("};")
    body.append("")

    TS_OUT.write_text("\n".join(body))
    print("done.")


if __name__ == "__main__":
    main()
