"""Auto-label compounds in the live DB with their English-Wikipedia name.

Strategy:
  1. Read wiki_wikidata_v3.csv -> {canon_smiles: enwiki_title} for the 294
     entries that have an actual enwiki article.
  2. Fetch /api/reaction-graph -> list of compounds with smiles + smiles_no_stereo.
  3. Fetch /api/annotations -> existing compound labels (entity_type="compounds").
  4. For each compound whose stereo-stripped SMILES matches a wikipedia entry:
       - no existing label   -> PUT the wikipedia title.
       - same label already  -> skip silently.
       - different label set -> print a WARNING and skip (do not overwrite).
  5. Note: matching is stereo-agnostic, so D-glucose and L-glucose both
     match the single "Glucose" wikipedia entry.  We label both.

Run:
  python push_wikipedia_labels.py --dry-run   # plan only
  python push_wikipedia_labels.py             # write
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx

API = "https://crn-cloud-api-yij7eukkba-uc.a.run.app"
CSV_PATH = Path("/home/sgugler/Research/dyanamical_system_tests/36_isomer_enumeration/wiki_wikidata_v3.csv")
CONCURRENCY = 6
RETRIES = 4


def load_wiki_names() -> dict[str, str]:
    """canon_smiles -> enwiki_title for entries with on_enwiki=True."""
    out: dict[str, str] = {}
    for r in csv.DictReader(CSV_PATH.open()):
        if r["on_enwiki"] != "True":
            continue
        title = r["enwiki_title"].strip()
        canon = r["canon_smiles"].strip()
        if title and canon:
            out[canon] = title
    return out


async def fetch_compounds(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get(f"{API}/api/reaction-graph", params={"level": "reaction"}, timeout=120)
    r.raise_for_status()
    nodes = r.json().get("nodes", [])
    return [n for n in nodes if n.get("type") == "compound"]


async def fetch_existing_labels(client: httpx.AsyncClient) -> dict[str, str]:
    r = await client.get(f"{API}/api/annotations", timeout=60)
    r.raise_for_status()
    cs = r.json().get("compounds", {}) or {}
    return {smi: (row.get("label") or "").strip() for smi, row in cs.items()}


async def put_label(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    smiles: str,
    label: str,
    notes: str | None,
    counters: dict,
):
    async with sem:
        url = f"{API}/api/annotations/compounds/{quote(smiles, safe='')}"
        for attempt in range(RETRIES):
            try:
                r = await client.put(url, json={"label": label, "notes": notes})
                if 200 <= r.status_code < 300:
                    counters["ok"] += 1
                    return
                if r.status_code < 500:
                    counters["client_err"] += 1
                    counters[f"err_{r.status_code}"] = counters.get(f"err_{r.status_code}", 0) + 1
                    return
            except Exception as e:
                counters["last_exc"] = repr(e)[:120]
            await asyncio.sleep(2 ** attempt)
        counters["failed"] += 1


async def main():
    global API
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--api", default=API)
    args = ap.parse_args()
    API = args.api

    wiki = load_wiki_names()
    print(f"Loaded {len(wiki)} wikipedia titles from {CSV_PATH.name}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        compounds, existing = await asyncio.gather(
            fetch_compounds(client),
            fetch_existing_labels(client),
        )
        print(f"  {len(compounds)} compounds in DB")
        print(f"  {len(existing)} existing compound annotations")

        # Plan: list of (smiles, new_label, existing_notes) to write
        to_write: list[tuple[str, str, str | None]] = []
        warnings: list[tuple[str, str, str]] = []  # (smi, existing_label, wiki_title)
        already_correct = 0
        unmatched = 0

        # Need notes from existing rows so we don't blank them when we PUT.
        existing_full: dict[str, dict] = {}
        r = await client.get(f"{API}/api/annotations", timeout=60)
        r.raise_for_status()
        for smi, row in (r.json().get("compounds", {}) or {}).items():
            existing_full[smi] = row

        for c in compounds:
            smi = c.get("smiles") or ""
            ns = c.get("smiles_no_stereo") or ""
            if not smi or not ns:
                continue
            title = wiki.get(ns)
            if not title:
                unmatched += 1
                continue
            cur = (existing.get(smi) or "").strip()
            if not cur:
                notes = (existing_full.get(smi, {}) or {}).get("notes")
                to_write.append((smi, title, notes))
            elif cur == title:
                already_correct += 1
            else:
                warnings.append((smi, cur, title))

        print()
        print(f"matched & to write:        {len(to_write)}")
        print(f"matched, label already OK: {already_correct}")
        print(f"matched, label DIFFERS:    {len(warnings)}  (will not overwrite)")
        print(f"compounds w/o wiki match:  {unmatched}")

        if warnings:
            print()
            print("=== WARNINGS: existing label differs from wikipedia title ===")
            for smi, cur, title in warnings:
                print(f"  smiles: {smi!r}")
                print(f"    existing: {cur!r}")
                print(f"    wiki:     {title!r}")

        if args.dry_run:
            print()
            print("Dry run — no writes performed.  First 12 to-write entries:")
            for smi, title, _ in to_write[:12]:
                print(f"  {smi!r:50s} -> {title!r}")
            return

        if not to_write:
            print("Nothing to write.")
            return

        print()
        print(f"Writing {len(to_write)} labels (concurrency={CONCURRENCY}) ...")
        sem = asyncio.Semaphore(CONCURRENCY)
        counters = {"ok": 0, "client_err": 0, "failed": 0}
        t0 = time.monotonic()
        tasks = [
            asyncio.create_task(put_label(client, sem, smi, lbl, notes, counters))
            for smi, lbl, notes in to_write
        ]
        for i, t in enumerate(asyncio.as_completed(tasks)):
            await t
            done = i + 1
            if done % 50 == 0 or done == len(tasks):
                rate = done / max(time.monotonic() - t0, 0.001)
                print(f"  {done}/{len(to_write)}  ok={counters['ok']}  "
                      f"client_err={counters['client_err']}  "
                      f"failed={counters['failed']}  ({rate:.1f}/s)")

        print()
        print(f"Final: ok={counters['ok']}  "
              f"client_err={counters['client_err']}  "
              f"failed={counters['failed']}")
        for k, v in sorted(counters.items()):
            if k.startswith("err_") or k == "last_exc":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
