#!/usr/bin/env python3.11
"""Download benign *technical* prompts used to measure false positives.

The four injection datasets contain few ordinary coding / security / DevOps
questions, so a classifier trained on them can flag prompts such as "Write a
Python function that validates a credit card with Luhn". This script samples
benign prompts from three public instruction datasets:

* ``codealpaca``  - sahil2801/CodeAlpaca-20k (CC-BY-4.0): coding tasks
* ``cyber``       - Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset
                    (Apache-2.0): defensive-security questions (``user`` turn only)
* ``dolly``       - databricks/databricks-dolly-15k (CC-BY-SA-3.0): general
                    questions, often with a pasted reference passage

Rows are written to ``benchmarks/data_benign/<short>.jsonl`` in the same format
as ``download_datasets.py`` (label is always 0). The sample is deterministic:
pages are taken at evenly spaced offsets, and each row's split ("train" or
"test") is fixed by a hash of its text, so re-running gives the same files.

These files are kept out of ``benchmarks/data/`` on purpose: the training
script reads every file there, and the benign set must first be used to measure
the *current* model before any of it is used for training.

Usage::

    python3.11 benchmarks/download_benign.py [--per-source 2000] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_datasets import PAGE_SIZE, PAGE_SLEEP_SECONDS, fetch_page  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data_benign"


def _join(*parts: object) -> str:
    return "\n\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


SOURCES = (
    {
        "short": "codealpaca",
        "dataset": "sahil2801/CodeAlpaca-20k",
        "rows": 20022,
        "extract": lambda r: _join(r.get("instruction"), r.get("input")),
    },
    {
        "short": "cyber",
        "dataset": "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        "rows": 53201,
        "extract": lambda r: _join(r.get("user")),
    },
    {
        "short": "dolly",
        "dataset": "databricks/databricks-dolly-15k",
        "rows": 15011,
        "extract": lambda r: _join(r.get("instruction"), r.get("context")),
    },
)


def split_for(text: str) -> str:
    """Stable 50/50 split by text hash (independent of sampling order)."""
    return "test" if zlib.crc32(text.encode("utf-8")) % 2 else "train"


def _fetch_patiently(dataset: str, offset: int, rounds: int = 4) -> dict:
    """fetch_page already backs off ~15 s; on a sustained 429 wait longer."""
    for round_no in range(1, rounds + 1):
        try:
            return fetch_page(dataset, "train", offset)
        except RuntimeError:
            if round_no == rounds:
                raise
            print(f"  rate-limited; waiting {30 * round_no}s", file=sys.stderr)
            time.sleep(30 * round_no)
    raise AssertionError("unreachable")


def download(spec: dict, per_source: int) -> list[dict]:
    pages = max(1, -(-per_source // PAGE_SIZE))
    step = max(PAGE_SIZE, spec["rows"] // pages)
    seen: set[str] = set()
    records: list[dict] = []
    for page in range(pages):
        offset = page * step
        payload = _fetch_patiently(spec["dataset"], offset)
        for item in payload.get("rows", []):
            text = spec["extract"](item.get("row") or {})
            if not text or text in seen:
                continue
            seen.add(text)
            records.append(
                {
                    "id": f"{spec['short']}-{item.get('row_idx', offset)}",
                    "text": text,
                    "label": 0,
                    "source": spec["short"],
                    "split": split_for(text),
                }
            )
        print(f"  {spec['short']}: offset {offset} -> {len(records)} rows", file=sys.stderr)
        time.sleep(max(PAGE_SLEEP_SECONDS, 2.0))  # the rows API rate-limits bursts
    return records[:per_source]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-source", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SOURCES:
        path = OUT_DIR / f"{spec['short']}.jsonl"
        if path.exists() and not args.force:
            print(f"skip {path.name} (exists)", file=sys.stderr)
            continue
        records = download(spec, args.per_source)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        test = sum(1 for r in records if r["split"] == "test")
        print(f"wrote {path.name}: {len(records)} rows ({test} test)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
