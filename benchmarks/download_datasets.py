#!/usr/bin/env python3.11
"""Download the prompt-injection datasets used by the static benchmark.

Only the Python standard library is used, together with the public Hugging Face
*datasets-server* "rows" API::

    https://datasets-server.huggingface.co/rows
        ?dataset=<urlencoded dataset id>
        &config=default
        &split=<split>
        &offset=<offset>
        &length=100

Every dataset is written to ``benchmarks/data/<short>.jsonl`` with one JSON
object per line::

    {"id": "<short>-<split>-<index>", "text": "...", "label": 0|1,
     "source": "<short>", "split": "<split>"}

Files that already exist are skipped unless ``--force`` is passed.

Usage::

    python3.11 benchmarks/download_datasets.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "sentinel-llm-guardrails-static-benchmark/1.0 (+stdlib urllib)"
PAGE_SIZE = 100
PAGE_SLEEP_SECONDS = 0.5
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30.0

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Row extractors: map a raw dataset row to (text, label) or None to skip it.
# ---------------------------------------------------------------------------


def _normalise_label(value) -> int | None:
    """Map the many label spellings used by these datasets onto 0/1."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float):
        return int(value) if value in (0.0, 1.0) else None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "injection", "inject", "jailbreak", "malicious"}:
            return 1
        if token in {"0", "false", "no", "benign", "safe", "legit", "legitimate", "normal"}:
            return 0
    return None


def _extract_deepset(row: dict):
    """deepset/prompt-injections -> fields `text`, `label` (1 = injection)."""
    text = row.get("text")
    label = _normalise_label(row.get("label"))
    if not isinstance(text, str) or label is None:
        return None
    return text, label


def _extract_jackhhao(row: dict):
    """jackhhao/jailbreak-classification -> fields `prompt`, `type`."""
    text = row.get("prompt")
    label = _normalise_label(row.get("type"))
    if not isinstance(text, str) or label is None:
        return None
    return text, label


def _extract_gandalf(row: dict):
    """Lakera/gandalf_ignore_instructions -> field `text`, all label 1."""
    text = row.get("text")
    if not isinstance(text, str):
        return None
    return text, 1


def _extract_safeguard(row: dict):
    """xTRam1/safe-guard-prompt-injection -> fields `text`, `label`."""
    text = row.get("text")
    label = _normalise_label(row.get("label"))
    if not isinstance(text, str) or label is None:
        return None
    return text, label


DATASETS = (
    {
        "short": "deepset",
        "dataset": "deepset/prompt-injections",
        "splits": ("train", "test"),
        "extract": _extract_deepset,
    },
    {
        "short": "jackhhao",
        "dataset": "jackhhao/jailbreak-classification",
        "splits": ("train", "test"),
        "extract": _extract_jackhhao,
    },
    {
        "short": "gandalf",
        "dataset": "Lakera/gandalf_ignore_instructions",
        "splits": ("train", "validation", "test"),
        "extract": _extract_gandalf,
    },
    {
        "short": "safeguard",
        "dataset": "xTRam1/safe-guard-prompt-injection",
        "splits": ("train", "test"),
        "extract": _extract_safeguard,
    },
)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def fetch_page(dataset: str, split: str, offset: int) -> dict:
    """Fetch one page of rows, retrying on 429/5xx/timeouts with backoff."""
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": PAGE_SIZE,
        }
    )
    url = f"{ROWS_API}?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable:
                # 404 / 400 etc. will not get better by retrying.
                raise
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            last_error = exc

        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"    retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.1f}s after {last_error}",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"giving up on {dataset}/{split} at offset {offset} after {MAX_ATTEMPTS} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def download_split(spec: dict, split: str) -> list[dict]:
    """Page through one split and return normalised records."""
    short = spec["short"]
    dataset = spec["dataset"]
    extract = spec["extract"]

    records: list[dict] = []
    offset = 0
    num_rows_total: int | None = None

    while True:
        payload = fetch_page(dataset, split, offset)
        rows = payload.get("rows") or []
        if num_rows_total is None:
            total = payload.get("num_rows_total")
            num_rows_total = int(total) if isinstance(total, int) else None

        for position, item in enumerate(rows):
            row = item.get("row") if isinstance(item, dict) else None
            if not isinstance(row, dict):
                continue
            extracted = extract(row)
            if extracted is None:
                continue
            text, label = extracted
            if not isinstance(text, str) or not text.strip():
                continue
            records.append(
                {
                    "id": f"{short}-{split}-{offset + position}",
                    "text": text,
                    "label": int(label),
                    "source": short,
                    "split": split,
                }
            )

        offset += len(rows)

        done = len(rows) < PAGE_SIZE
        if num_rows_total is not None and offset >= num_rows_total:
            done = True
        if done:
            break

        time.sleep(PAGE_SLEEP_SECONDS)

    total_text = "unknown" if num_rows_total is None else str(num_rows_total)
    print(f"  {short}/{split}: kept {len(records)} rows (rows on server: {total_text})")
    return records


def download_dataset(spec: dict, force: bool = False) -> Path:
    """Download every split of one dataset into benchmarks/data/<short>.jsonl."""
    out_path = DATA_DIR / f"{spec['short']}.jsonl"
    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} already exists (use --force to re-download)")
        return out_path

    records: list[dict] = []
    for split in spec["splits"]:
        records.extend(download_split(spec, split))

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    positives = sum(1 for record in records if record["label"] == 1)
    negatives = len(records) - positives
    print(f"[done] {out_path.name}: {len(records)} rows (label=1: {positives}, label=0: {negatives})")
    return out_path


def summarise(path: Path) -> tuple[int, int, int]:
    """Return (rows, label=1, label=0) for a written jsonl file."""
    total = positives = negatives = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if record.get("label") == 1:
                positives += 1
            elif record.get("label") == 0:
                negatives += 1
    return total, positives, negatives


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the datasets used by the static detection benchmark "
        "(Hugging Face datasets-server rows API, stdlib only)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when benchmarks/data/<short>.jsonl already exists",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for spec in DATASETS:
        print(f"[get ] {spec['short']} <- {spec['dataset']} (splits: {', '.join(spec['splits'])})")
        try:
            download_dataset(spec, force=args.force)
        except Exception as exc:  # noqa: BLE001 - keep going with the other datasets
            failures.append(spec["short"])
            print(f"[fail] {spec['short']}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print()
    print(f"{'dataset':<12} {'rows':>7} {'label=1':>9} {'label=0':>9}")
    print(f"{'-' * 12} {'-' * 7} {'-' * 9} {'-' * 9}")
    for spec in DATASETS:
        path = DATA_DIR / f"{spec['short']}.jsonl"
        if not path.exists():
            print(f"{spec['short']:<12} {'-':>7} {'-':>9} {'-':>9}")
            continue
        total, positives, negatives = summarise(path)
        print(f"{spec['short']:<12} {total:>7} {positives:>9} {negatives:>9}")

    if failures:
        print(
            f"\n{len(failures)} dataset(s) failed: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
