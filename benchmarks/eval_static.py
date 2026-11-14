#!/usr/bin/env python3.11
"""Static prompt-injection detection benchmark for sentinel-llm-guardrails.

Measures precision/recall of the built-in input guardrail on fixed, publicly
available labelled corpora — both with the learned injection classifier enabled
(the shipped default) and with the regex + heuristic stack alone. A text counts
as *detected* only when ``"LLM01_prompt_injection" in result.checks_failed``;
any other block (secrets, PII, ...) is recorded as ``other_block`` and never as
a detection.

Outputs (under ``benchmarks/results/``):
    static.json                full machine-readable report
    static.md                  markdown tables (also printed to stdout)
    static_errors_sample.jsonl up to 40 false negatives + 40 false positives

Usage::

    python3.11 benchmarks/eval_static.py [--seed 0] [--threshold 0.45]
                                        [--split {all,dev,test}] [--no-classifier]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_llm_guardrails import LLMGuardrails, __version__  # noqa: E402

DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"

DEFAULT_THRESHOLD = 0.45
THRESHOLD_SWEEP = (0.25, 0.35, 0.45, 0.55, 0.65)
MAX_TEXT_CHARS = 32000
ERROR_SAMPLE_PER_CLASS = 40
ERROR_TEXT_CHARS = 300
Z_95 = 1.959963984540054

INJECTION_CHECK = "LLM01_prompt_injection"
REGEX_REASON = "regex_injection"
HEURISTIC_MECHANISM = "heuristic_score"
CLASSIFIER_MECHANISM = "classifier"
MECHANISMS = (REGEX_REASON, HEURISTIC_MECHANISM, CLASSIFIER_MECHANISM)

DATASET_ORDER = ("deepset", "jackhhao", "gandalf", "safeguard")
DATASET_LABELS = {
    "deepset": "deepset/prompt-injections",
    "jackhhao": "jackhhao/jailbreak-classification",
    "gandalf": "Lakera/gandalf_ignore_instructions",
    "safeguard": "xTRam1/safe-guard-prompt-injection",
    "combined": "**combined**",
}

# dev = rows whose split is train or validation, test = rows whose split is test.
DEV_SPLITS = ("train", "validation")


# ---------------------------------------------------------------------------
# Guard construction and small helpers
# ---------------------------------------------------------------------------


def make_guard(threshold: float, use_classifier: bool = False) -> LLMGuardrails:
    """Build a guard with effectively unlimited budget and auditing off."""
    return LLMGuardrails(
        daily_request_limit=10**9,
        daily_token_limit=10**12,
        audit_enabled=False,
        injection_score_threshold=threshold,
        use_classifier=use_classifier,
    )


def matches_split(split: str, wanted: str) -> bool:
    """``dev`` = train/validation, ``test`` = test, ``all`` = everything."""
    value = (split or "").strip().lower()
    if wanted == "all":
        return True
    if wanted == "dev":
        return value in DEV_SPLITS
    return value == "test"


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _contains(values, needle: str) -> bool:
    """Membership test that also tolerates enums / prefixed names."""
    lowered = needle.lower()
    for value in values:
        if value == needle:
            return True
        if isinstance(value, str):
            candidate = value
        else:
            name = getattr(value, "name", None)
            if not isinstance(name, str):
                name = getattr(value, "value", None)
            candidate = name if isinstance(name, str) else None
        if candidate is not None and lowered in candidate.lower():
            return True
    return False


def _result_score(result):
    """Best-effort retrieval of the guardrail's numeric risk score."""
    for attribute in ("injection_score", "risk_score", "score"):
        value = getattr(result, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _normalise_label(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, float):
        return int(value) if value in (0.0, 1.0) else None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes"}:
            return 1
        if token in {"0", "false", "no"}:
            return 0
    return None


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int) -> list[float] | None:
    """Wilson score interval (95%) as [low, high], or None when total == 0."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + (Z_95 * Z_95) / total
    centre = (proportion + (Z_95 * Z_95) / (2.0 * total)) / denominator
    spread = (Z_95 / denominator) * math.sqrt(
        proportion * (1.0 - proportion) / total + (Z_95 * Z_95) / (4.0 * total * total)
    )
    return [max(0.0, centre - spread), min(1.0, centre + spread)]


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Confusion-matrix derived metrics; None means "n/a" (zero denominator)."""
    positives = tp + fn
    negatives = fp + tn

    detection_rate = (tp / positives) if positives else None
    false_positive_rate = (fp / negatives) if negatives else None
    precision = (tp / (tp + fp)) if (tp + fp) else None

    if detection_rate is None or precision is None:
        f1 = None
    elif detection_rate + precision == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * detection_rate * precision / (detection_rate + precision)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_positives": positives,
        "n_negatives": negatives,
        "detection_rate": detection_rate,
        "detection_rate_ci95": wilson_ci(tp, positives),
        "fpr": false_positive_rate,
        "fpr_ci95": wilson_ci(fp, negatives),
        "precision": precision,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def discover_datasets(data_dir: Path) -> list[tuple[str, Path]]:
    """Known datasets first (in a stable order), then any extra *.jsonl files."""
    available = {path.stem: path for path in sorted(data_dir.glob("*.jsonl"))}
    ordered: list[tuple[str, Path]] = []
    for short in DATASET_ORDER:
        if short in available:
            ordered.append((short, available.pop(short)))
    for short in sorted(available):
        ordered.append((short, available[short]))
    return ordered


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] {path.name}:{line_number}: invalid JSON, skipped", file=sys.stderr)
                continue
            text = row.get("text")
            label = _normalise_label(row.get("label"))
            if not isinstance(text, str) or label is None or not text.strip():
                continue
            source = str(row.get("source") or path.stem)
            split = str(row.get("split") or "unknown")
            record_id = str(row.get("id") or f"{source}-{split}-{len(records)}")
            records.append(
                {
                    "id": record_id,
                    "text": text,
                    "label": label,
                    "source": source,
                    "split": split,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _sample_entry(kind: str, record: dict, mechanism: str, other_block: bool, result) -> dict:
    text = record["text"]
    return {
        "kind": kind,
        "dataset": record["source"],
        "split": record.get("split") or None,
        "id": record["id"],
        "label": record["label"],
        "mechanism": mechanism,
        "other_block": bool(other_block),
        "score": _result_score(result),
        "text_length": len(text),
        "text_truncated": len(text) > ERROR_TEXT_CHARS,
        "text": text[:ERROR_TEXT_CHARS],
    }


def _mechanism(risk_reasons) -> str:
    if _contains(risk_reasons, REGEX_REASON):
        return REGEX_REASON
    if _contains(risk_reasons, CLASSIFIER_MECHANISM):
        return CLASSIFIER_MECHANISM
    return HEURISTIC_MECHANISM


def evaluate_records(
    records: list[dict],
    threshold: float,
    use_classifier: bool = False,
    collect_pools: bool = False,
):
    """Run the guard over ``records``.

    Returns ``(metrics, false_negatives, false_positives)``; the two pools are
    only populated when ``collect_pools`` is true (they keep truncated texts).
    """
    guard = make_guard(threshold, use_classifier=use_classifier)

    tp = fp = tn = fn = 0
    skipped_too_long = 0
    errors = 0
    other_blocks = 0
    mechanisms = {
        "tp": {name: 0 for name in MECHANISMS},
        "fp": {name: 0 for name in MECHANISMS},
    }
    false_negatives: list[dict] = []
    false_positives: list[dict] = []

    for record in records:
        text = record["text"]
        if len(text) > MAX_TEXT_CHARS:
            skipped_too_long += 1
            continue

        try:
            result = guard.check_input(text, user_id="bench")
        except Exception as exc:  # noqa: BLE001 - a crashing check counts as a miss
            errors += 1
            if errors <= 5:
                print(f"[warn] check_input raised {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        checks_failed = _as_list(getattr(result, "checks_failed", None))
        risk_reasons = _as_list(getattr(result, "risk_reasons", None))

        flagged = _contains(checks_failed, INJECTION_CHECK)
        other_block = (not flagged) and bool(checks_failed)
        mechanism = _mechanism(risk_reasons)
        label = record["label"]

        if flagged and label == 1:
            tp += 1
            mechanisms["tp"][mechanism] += 1
        elif flagged and label == 0:
            fp += 1
            mechanisms["fp"][mechanism] += 1
            if collect_pools:
                false_positives.append(
                    _sample_entry("false_positive", record, mechanism, other_block, result)
                )
        elif label == 1:
            fn += 1
            if collect_pools:
                false_negatives.append(
                    _sample_entry("false_negative", record, mechanism, other_block, result)
                )
        else:
            tn += 1

        if other_block:
            other_blocks += 1

    metrics = compute_metrics(tp, fp, tn, fn)
    metrics.update(
        {
            "n_records": len(records),
            "n_evaluated": tp + fp + tn + fn,
            "skipped_too_long": skipped_too_long,
            "errors": errors,
            "other_blocks": other_blocks,
            "mechanisms": mechanisms,
        }
    )
    return metrics, false_negatives, false_positives


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_rate(value) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _fmt_rate_ci(value, interval) -> str:
    if value is None:
        return "n/a"
    text = f"{100.0 * value:.2f}%"
    if interval:
        low, high = interval
        text += f" [{100.0 * low:.2f}%, {100.0 * high:.2f}%]"
    return text


def _metrics_table(datasets: dict) -> list[str]:
    lines = [
        "| Dataset | Evaluated | Positives | Negatives | TP | FP | TN | FN | "
        "Detection rate (95% CI) | FPR (95% CI) | Precision | F1 | Skipped too long | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for short, metrics in datasets.items():
        lines.append(
            "| {name} | {n} | {pos} | {neg} | {tp} | {fp} | {tn} | {fn} | {dr} | {fpr} | "
            "{precision} | {f1} | {skipped} | {errors} |".format(
                name=DATASET_LABELS.get(short, short),
                n=metrics.get("n_evaluated", 0),
                pos=metrics.get("n_positives", 0),
                neg=metrics.get("n_negatives", 0),
                tp=metrics.get("tp", 0),
                fp=metrics.get("fp", 0),
                tn=metrics.get("tn", 0),
                fn=metrics.get("fn", 0),
                dr=_fmt_rate_ci(metrics.get("detection_rate"), metrics.get("detection_rate_ci95")),
                fpr=_fmt_rate_ci(metrics.get("fpr"), metrics.get("fpr_ci95")),
                precision=_fmt_rate(metrics.get("precision")),
                f1=_fmt_rate(metrics.get("f1")),
                skipped=metrics.get("skipped_too_long", 0),
                errors=metrics.get("errors", 0),
            )
        )
    return lines


def _mechanism_table(datasets: dict) -> list[str]:
    lines = [
        "| Dataset | TP via regex | TP via classifier | TP via heuristic score | "
        "FP via regex | FP via classifier | FP via heuristic score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for short, metrics in datasets.items():
        mechanisms = metrics.get("mechanisms") or {}
        tp_mech = mechanisms.get("tp") or {}
        fp_mech = mechanisms.get("fp") or {}
        lines.append(
            f"| {DATASET_LABELS.get(short, short)} "
            f"| {tp_mech.get(REGEX_REASON, 0)} | {tp_mech.get(CLASSIFIER_MECHANISM, 0)} "
            f"| {tp_mech.get(HEURISTIC_MECHANISM, 0)} "
            f"| {fp_mech.get(REGEX_REASON, 0)} | {fp_mech.get(CLASSIFIER_MECHANISM, 0)} "
            f"| {fp_mech.get(HEURISTIC_MECHANISM, 0)} |"
        )
    return lines


def render_markdown(payload: dict) -> str:
    classifier_on = payload["classifier_enabled"]
    datasets_on = payload["datasets_with_classifier"]
    datasets_off = payload["datasets_without_classifier"]

    lines: list[str] = []

    lines.append("# sentinel-llm-guardrails — static prompt-injection detection benchmark")
    lines.append("")
    lines.append(f"- Library version: `{payload['library_version']}`")
    lines.append(f"- Run date (UTC): `{payload['run_date_utc']}`")
    lines.append(f"- Split: `{payload['split']}`")
    lines.append(f"- Injection score threshold: `{payload['threshold']}`")
    lines.append(f"- Random seed: `{payload['seed']}`")
    lines.append(
        "- Learned classifier: `{}`".format(
            "enabled (results are also repeated with it disabled below)"
            if classifier_on
            else "disabled via `--no-classifier`"
        )
    )
    lines.append(f"- Texts longer than {payload['max_text_chars']} characters: skipped.")
    lines.append(
        '- Detection rule: a text is *flagged* only if `"LLM01_prompt_injection" in '
        'result.checks_failed`; any other block (secrets, PII, ...) is recorded as '
        "`other_block` and never as a detection."
    )
    lines.append("")

    if classifier_on:
        lines.append("## Detection metrics per dataset — injection classifier ON")
        lines.append("")
        lines.append(
            "Shipped default configuration: regex → learned classifier → heuristic score."
        )
        lines.append("")
        lines.extend(_metrics_table(datasets_on))
        lines.append("")

    lines.append(
        "## Detection metrics per dataset — injection classifier OFF "
        "(regex + heuristic score only)"
    )
    lines.append("")
    lines.extend(_metrics_table(datasets_off))
    lines.append("")
    lines.append(
        "`Detection rate` is recall on label-1 rows, `FPR` is the false-positive rate on label-0 rows. "
        "`n/a` marks a metric whose denominator is zero — `Lakera/gandalf_ignore_instructions` only "
        "contains injection rows, so its FPR, precision and F1 are undefined."
    )
    lines.append("")

    primary_name = "classifier ON" if classifier_on else "classifier OFF"
    primary = datasets_on if classifier_on else datasets_off
    lines.append(f"## Detections by mechanism ({primary_name})")
    lines.append("")
    lines.append(
        'A detection is attributed to `regex_injection` when `"regex_injection" in '
        "result.risk_reasons`, to `classifier` when the learned model crossed its "
        "threshold, otherwise to the heuristic injection score."
    )
    lines.append("")
    lines.extend(_mechanism_table(primary))
    lines.append("")

    lines.append("## Threshold sweep (regex + heuristic only, combined datasets)")
    lines.append("")
    lines.append(
        "`injection_score_threshold` only affects the heuristic scorer, so the sweep is "
        "run with the classifier disabled."
    )
    lines.append("")
    lines.append("| Threshold | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in payload["threshold_sweep"]:
        lines.append(
            "| {t} | {tp} | {fp} | {tn} | {fn} | {dr} | {fpr} |".format(
                t=f"{row['threshold']:.2f}",
                tp=row["tp"],
                fp=row["fp"],
                tn=row["tn"],
                fn=row["fn"],
                dr=_fmt_rate_ci(row["detection_rate"], row["detection_rate_ci95"]),
                fpr=_fmt_rate_ci(row["fpr"], row["fpr_ci95"]),
            )
        )
    lines.append("")

    sample = payload["error_sample"]
    lines.append("## Error analysis sample")
    lines.append("")
    lines.append(
        f"Up to {sample['max_per_class']} random false negatives and {sample['max_per_class']} random "
        f"false positives (texts truncated to {sample['text_chars']} characters) are written to "
        f"`{sample['file']}`."
    )
    lines.append(
        f"Sampled {sample['false_negatives_sampled']} / {sample['false_negatives_available']} false "
        f"negatives and {sample['false_positives_sampled']} / {sample['false_positives_available']} "
        f"false positives from the `{sample['configuration']}` configuration."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- This is a **static** benchmark: it measures detection on fixed labelled corpora and says "
        "nothing about robustness against attackers who adapt to the detector (paraphrasing, encoding, "
        "multi-turn setups, tool-mediated injections). For adaptive-attack claims, cite published work "
        "instead of these numbers."
    )
    lines.append(
        "- The learned classifier is trained offline (see `benchmarks/train_classifier.py`) and its "
        "test rows are excluded from training; these numbers are only as honest as the split field in "
        "the source corpora."
    )
    lines.append(
        "- The corpora are third-party; check every dataset page for licence and intended use before "
        "publishing numbers."
    )
    lines.append(
        "- Labels reflect the dataset authors' judgement; a prompt that is benign in one context can be "
        "an attack in another, so precision/recall here are corpus-specific."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Static prompt-injection detection benchmark for sentinel-llm-guardrails."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for deterministic error sampling (default: 0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"injection_score_threshold for the main run (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--split",
        choices=("all", "dev", "test"),
        default="test",
        help="which rows to evaluate: dev = split train/validation, test = split test (default: test)",
    )
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="report only the regex + heuristic stack (no learned classifier)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_datasets(args.data_dir)
    if not discovered:
        print(
            f"[error] no *.jsonl datasets in {args.data_dir}; "
            "run `python3.11 benchmarks/download_datasets.py` first",
            file=sys.stderr,
        )
        return 2

    run_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    classifier_on = not args.no_classifier
    primary_configuration = "with_classifier" if classifier_on else "without_classifier"

    per_dataset_on: dict = {}
    per_dataset_off: dict = {}
    combined_records: list[dict] = []
    false_negative_pool: list[dict] = []
    false_positive_pool: list[dict] = []

    for short, path in discovered:
        records = load_records(path)
        filtered = [r for r in records if matches_split(r["split"], args.split)]
        combined_records.extend(filtered)
        print(
            f"[info] {short}: {len(filtered)}/{len(records)} rows match split={args.split}",
            file=sys.stderr,
        )

        if classifier_on:
            metrics_on, _, _ = evaluate_records(
                filtered, args.threshold, use_classifier=True, collect_pools=False
            )
            per_dataset_on[short] = metrics_on
            metrics_off, missed, false_alarms = evaluate_records(
                filtered, args.threshold, use_classifier=False, collect_pools=True
            )
            per_dataset_off[short] = metrics_off
            false_negative_pool.extend(missed)
            false_positive_pool.extend(false_alarms)
        else:
            metrics_off, missed, false_alarms = evaluate_records(
                filtered, args.threshold, use_classifier=False, collect_pools=True
            )
            per_dataset_off[short] = metrics_off
            false_negative_pool.extend(missed)
            false_positive_pool.extend(false_alarms)

    print(f"[info] combined: {len(combined_records)} rows", file=sys.stderr)

    if classifier_on:
        combined_on, _, _ = evaluate_records(
            combined_records, args.threshold, use_classifier=True, collect_pools=False
        )
        per_dataset_on["combined"] = combined_on
    combined_off, _, _ = evaluate_records(
        combined_records, args.threshold, use_classifier=False, collect_pools=False
    )
    per_dataset_off["combined"] = combined_off

    threshold_sweep: list[dict] = []
    for threshold in THRESHOLD_SWEEP:
        if abs(threshold - args.threshold) < 1e-9:
            metrics = combined_off
        else:
            print(f"[info] threshold sweep: t={threshold:.2f}", file=sys.stderr)
            metrics, _, _ = evaluate_records(
                combined_records, threshold, use_classifier=False, collect_pools=False
            )
        threshold_sweep.append(
            {
                "threshold": threshold,
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "detection_rate": metrics["detection_rate"],
                "detection_rate_ci95": metrics["detection_rate_ci95"],
                "fpr": metrics["fpr"],
                "fpr_ci95": metrics["fpr_ci95"],
            }
        )

    rng = random.Random(args.seed)
    sampled_fn = rng.sample(false_negative_pool, min(ERROR_SAMPLE_PER_CLASS, len(false_negative_pool)))
    sampled_fp = rng.sample(false_positive_pool, min(ERROR_SAMPLE_PER_CLASS, len(false_positive_pool)))

    errors_path = args.results_dir / "static_errors_sample.jsonl"
    with errors_path.open("w", encoding="utf-8") as handle:
        for entry in list(sampled_fn) + list(sampled_fp):
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    primary_metrics = per_dataset_on if classifier_on else per_dataset_off

    payload = {
        "benchmark": "sentinel-llm-guardrails static prompt-injection detection benchmark",
        "library_version": __version__,
        "run_date_utc": run_date,
        "split": args.split,
        "seed": args.seed,
        "threshold": args.threshold,
        "classifier_enabled": classifier_on,
        "primary_configuration": primary_configuration,
        "threshold_sweep_values": list(THRESHOLD_SWEEP),
        "max_text_chars": MAX_TEXT_CHARS,
        "flagging_rule": 'flagged iff "LLM01_prompt_injection" in result.checks_failed',
        "other_block_rule": "any other entry in checks_failed is recorded as other_block, never as a detection",
        "data_dir": str(args.data_dir),
        # `datasets` mirrors the primary configuration for backwards compatibility.
        "datasets": primary_metrics,
        "datasets_with_classifier": per_dataset_on if classifier_on else {},
        "datasets_without_classifier": per_dataset_off,
        "threshold_sweep": threshold_sweep,
        "error_sample": {
            "file": _relative_to_project(errors_path),
            "configuration": primary_configuration,
            "max_per_class": ERROR_SAMPLE_PER_CLASS,
            "text_chars": ERROR_TEXT_CHARS,
            "false_negatives_available": len(false_negative_pool),
            "false_positives_available": len(false_positive_pool),
            "false_negatives_sampled": len(sampled_fn),
            "false_positives_sampled": len(sampled_fp),
        },
        "notes": [
            "Static benchmark on fixed labelled corpora only.",
            "It does not measure robustness against adaptive attackers; cite published work for adaptive-attack claims.",
            "Lakera/gandalf_ignore_instructions contains only injection rows, so FPR/precision/F1 are n/a for it.",
            "Dataset licences are third-party and must be checked on each dataset page.",
        ],
    }

    json_path = args.results_dir / "static.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown = render_markdown(payload)
    md_path = args.results_dir / "static.md"
    md_path.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print()
    print(f"[info] wrote {json_path}")
    print(f"[info] wrote {md_path}")
    print(f"[info] wrote {errors_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
