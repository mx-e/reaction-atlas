#!/usr/bin/env bash
#
# Produce the single ZIP of "all required content" for Nature peer review:
# source code + the small demo dataset + README + docs + license.
#
# It archives the *committed* state of the repository via `git archive`, so it
# automatically respects .gitignore (no .venv, caches, secrets, or large local
# dumps) and exactly matches what a reviewer would clone. Commit your changes
# first.
#
# The full published reaction-network database and the seed inputs are released
# separately on Zenodo (see docs/data.md) and are intentionally NOT bundled —
# they are tens of GB. The committed generative-proposer checkpoint
# (packages/worker/models/ts_best_model) and the 8 KB kinetics-demo network ARE
# included.
#
# Usage:  ./scripts/make_submission_zip.sh [output.zip]

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT="${1:-reaction-atlas-submission.zip}"
REF="${2:-HEAD}"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "WARNING: you have uncommitted changes; the zip archives committed state ($REF) only." >&2
fi

git archive --format=zip --prefix=reaction-atlas/ -o "$OUT" "$REF"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))."
echo "Top-level contents:"
unzip -l "$OUT" | awk '{print $4}' | grep -E '^reaction-atlas/[^/]+/?$' | sort -u
