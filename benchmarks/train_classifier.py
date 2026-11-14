#!/usr/bin/env python3.11
"""Train and honestly evaluate the stdlib-only prompt-injection classifier.

Protocol
--------
* Data: ``benchmarks/data/*.jsonl`` with fields ``id, text, label, source, split``.
  Texts longer than 32 000 characters are skipped.
* **Dev**  = rows whose split is ``train`` or ``validation``.
* **Test** = rows whose split is ``test``. Test rows are never used for training,
  for model selection or for threshold selection.
* Test rows whose normalised text (lowercase, whitespace collapsed) also occurs in
  dev are dropped before evaluation; the number dropped is reported per dataset.
* The decision threshold is picked by 5-fold cross-validation on dev: the lowest
  threshold whose out-of-fold false-positive rate on benign rows is
  ``<= --target-fpr``. The final model is then trained on all dev rows.
* Test evaluation reports the classifier alone and the deployed detector
  (regex OR classifier), with Wilson 95% confidence intervals.
* Leave-one-dataset-out: for each dataset, train on the dev rows of the other
  datasets (threshold chosen the same way) and evaluate on that dataset's test rows.

Usage::

    python3.11 benchmarks/train_classifier.py [--seed 0] [--target-fpr 0.01]
                                             [--epochs 8] [--skip-lodo]

Outputs: ``src/sentinel_llm_guardrails/models/injection_lr.json``,
``benchmarks/results/classifier.md`` and ``benchmarks/results/classifier.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

from eval_static import _normalise_label, compute_metrics  # noqa: E402
from sentinel_llm_guardrails import LLMGuardrails, __version__  # noqa: E402
from sentinel_llm_guardrails.classifier import (  # noqa: E402
    InjectionClassifier,
    featurize,
    train,
)

DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
MODEL_PATH = PROJECT_ROOT / "src" / "sentinel_llm_guardrails" / "models" / "injection_lr.json"

MAX_TEXT_CHARS = 32_000
FOLDS = 5
DATASET_ORDER = ("deepset", "jackhhao", "gandalf", "safeguard")
DATASET_LABELS = {
    "deepset": "deepset/prompt-injections",
    "jackhhao": "jackhhao/jailbreak-classification",
    "gandalf": "Lakera/gandalf_ignore_instructions",
    "safeguard": "xTRam1/safe-guard-prompt-injection",
    "combined": "**combined**",
}
DEV_SPLITS = ("train", "validation")
INJECTION_CHECK = "LLM01_prompt_injection"
# The heuristic scorer tops out at 1.0, so a threshold above it disables the
# heuristic and leaves the regex layer as the only non-classifier detector.
REGEX_ONLY_SCORE_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Loading / splitting
# ---------------------------------------------------------------------------


def load_rows(data_dir: Path) -> tuple[list[dict], dict[str, int]]:
    """Read every ``*.jsonl`` under ``data_dir``; returns (rows, skipped-too-long)."""
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        dataset = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"[warn] {path.name}:{line_number}: invalid JSON, skipped",
                        file=sys.stderr,
                    )
                    continue
                text = raw.get("text")
                label = _normalise_label(raw.get("label"))
                if not isinstance(text, str) or label is None or not text.strip():
                    continue
                if len(text) > MAX_TEXT_CHARS:
                    skipped[dataset] = skipped.get(dataset, 0) + 1
                    continue
                rows.append(
                    {
                        "id": str(raw.get("id") or f"{dataset}-{line_number}"),
                        "dataset": dataset,
                        "source": str(raw.get("source") or dataset),
                        "split": str(raw.get("split") or "unknown").strip().lower(),
                        "text": text,
                        "label": label,
                    }
                )
    return rows, skipped


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """``dev`` = split in {train, validation}; ``test`` = split == test."""
    dev = [row for row in rows if row["split"] in DEV_SPLITS]
    test = [row for row in rows if row["split"] == "test"]
    return dev, test


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe_test(dev_rows: list[dict], test_rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Drop test rows whose normalised text also appears in dev."""
    seen = {_normalise_text(row["text"]) for row in dev_rows}
    kept: list[dict] = []
    dropped: dict[str, int] = {}
    for row in test_rows:
        if _normalise_text(row["text"]) in seen:
            dropped[row["dataset"]] = dropped.get(row["dataset"], 0) + 1
            continue
        kept.append(row)
    return kept, dropped


def group_by_dataset(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["dataset"], []).append(row)
    ordered = [(name, groups[name]) for name in DATASET_ORDER if name in groups]
    ordered += [(name, groups[name]) for name in sorted(groups) if name not in DATASET_ORDER]
    return ordered


# ---------------------------------------------------------------------------
# Featurisation cache (each text is featurised at most once per run)
# ---------------------------------------------------------------------------


class FeatureCache:
    """Memoises :func:`featurize` so every training fit reuses the same vectors."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[int, float]] = {}
        self.computed = 0

    def get(self, text: str) -> dict[int, float]:
        cached = self._cache.get(text)
        if cached is None:
            cached = featurize(text)
            self._cache[text] = cached
            self.computed += 1
        return cached


# ---------------------------------------------------------------------------
# Threshold selection (cross-validation on dev)
# ---------------------------------------------------------------------------


def cross_val_probabilities(
    rows: list[dict],
    cache: FeatureCache,
    epochs: int,
    seed: int,
    folds: int = FOLDS,
    log=None,
) -> list[float]:
    """Out-of-fold probabilities for ``rows`` using seeded k-fold CV."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    fold_of = [0] * len(rows)
    for position, index in enumerate(order):
        fold_of[index] = position % folds

    probabilities = [0.0] * len(rows)
    for fold in range(folds):
        train_rows = [
            (cache.get(rows[index]["text"]), rows[index]["label"])
            for index in order
            if fold_of[index] != fold
        ]
        val_indexes = [index for index in order if fold_of[index] == fold]
        model = train(train_rows, epochs=epochs, seed=seed + fold)
        for index in val_indexes:
            probabilities[index] = model.predict_proba_features(cache.get(rows[index]["text"]))
        if log is not None:
            log(f"  fold {fold + 1}/{folds}: {len(train_rows)} train / {len(val_indexes)} val rows")
    return probabilities


def choose_threshold(probabilities: list[float], labels: list[int], target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on benign rows is <= ``target_fpr``.

    FPR is monotone non-increasing in the threshold, so the lowest qualifying
    candidate maximises recall subject to the false-positive budget.
    """
    benign = [p for p, y in zip(probabilities, labels) if y == 0]
    if not benign:
        return 0.5  # nothing to calibrate against; stay neutral
    for threshold in sorted(set(probabilities)):
        flagged = sum(1 for p in benign if p >= threshold)
        if flagged / len(benign) <= target_fpr:
            return threshold
    # Even the highest score flags too many benign rows — demand an off-the-scale score.
    return min(1.0, max(probabilities) + 1e-9)


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------


def _empty_confusion() -> dict:
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}


def _tally(confusion: dict, flagged: bool, label: int) -> None:
    if flagged and label == 1:
        confusion["tp"] += 1
    elif flagged:
        confusion["fp"] += 1
    elif label == 1:
        confusion["fn"] += 1
    else:
        confusion["tn"] += 1


def evaluate_alone(model, threshold: float, rows: list[dict], cache: FeatureCache):
    """Classifier-only decisions, split per dataset plus a combined tally."""
    per_dataset: dict[str, dict] = {}
    combined = _empty_confusion()
    for row in rows:
        flagged = model.predict_proba_features(cache.get(row["text"])) >= threshold
        _tally(per_dataset.setdefault(row["dataset"], _empty_confusion()), flagged, row["label"])
        _tally(combined, flagged, row["label"])
    return per_dataset, combined


def evaluate_regex_or_classifier(model, threshold: float, rows: list[dict]):
    """The deployed detector: regex layer OR the learned classifier.

    ``injection_score_threshold=2.0`` disables the heuristic scorer (it can never
    reach 2.0) so this measures exactly "regex OR classifier".

    The freshly trained ``model`` is injected into the guard so the result never
    depends on whichever model file happens to be packaged (``--model-out``).
    """
    guard = LLMGuardrails(
        daily_request_limit=10 ** 9,
        daily_token_limit=10 ** 12,
        audit_enabled=False,
        injection_score_threshold=REGEX_ONLY_SCORE_THRESHOLD,
        use_classifier=True,
        classifier_threshold=threshold,
        classifier=model,
    )
    per_dataset: dict[str, dict] = {}
    combined = _empty_confusion()
    for row in rows:
        result = guard.check_input(row["text"], user_id="bench")
        flagged = INJECTION_CHECK in (result.checks_failed or [])
        _tally(per_dataset.setdefault(row["dataset"], _empty_confusion()), flagged, row["label"])
        _tally(combined, flagged, row["label"])
    return per_dataset, combined


def metrics_from_confusion(confusion: dict) -> dict:
    return compute_metrics(confusion["tp"], confusion["fp"], confusion["tn"], confusion["fn"])


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def _render_eval_table(title: str, datasets: dict) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append("| Dataset | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) | Precision |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |")
    for name, metrics in datasets.items():
        lines.append(
            "| {label} | {tp} | {fp} | {tn} | {fn} | {dr} | {fpr} | {prec} |".format(
                label=DATASET_LABELS.get(name, name),
                tp=metrics["tp"],
                fp=metrics["fp"],
                tn=metrics["tn"],
                fn=metrics["fn"],
                dr=_fmt_rate_ci(metrics["detection_rate"], metrics["detection_rate_ci95"]),
                fpr=_fmt_rate_ci(metrics["fpr"], metrics["fpr_ci95"]),
                prec=_fmt_rate(metrics["precision"]),
            )
        )
    lines.append("")
    return lines


def render_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# sentinel-llm-guardrails — learned prompt-injection classifier")
    lines.append("")
    lines.append(
        "Training and evaluation report for the stdlib-only logistic-regression "
        "classifier shipped in `src/sentinel_llm_guardrails/models/injection_lr.json`."
    )
    lines.append("")
    lines.append(f"- Library version: `{payload['library_version']}`")
    lines.append(f"- Run date (UTC): `{payload['run_date_utc']}`")
    lines.append(f"- Random seed: `{payload['seed']}`")
    lines.append(f"- SGD epochs per fit: `{payload['epochs']}`")
    lines.append(f"- CV folds: `{payload['folds']}`")
    lines.append(f"- Target out-of-fold FPR for threshold selection: `{payload['target_fpr']}`")
    lines.append(f"- Chosen threshold (from dev CV): `{payload['threshold']:.4f}`")
    lines.append(f"- Trained on dev datasets: `{', '.join(payload['trained_on'])}`")
    lines.append(f"- Model file: `{payload['model_path']}` ({payload['n_weights']} stored weights)")
    lines.append(f"- Dev rows: `{payload['dev_rows']}`, test rows after de-duplication: `{payload['test_rows']}`")
    lines.append("")

    lines.append("## Data")
    lines.append("")
    lines.append("| Dataset | Dev rows | Test rows | Test rows dropped as dev duplicates |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in payload["data"]:
        lines.append(
            f"| {DATASET_LABELS.get(row['dataset'], row['dataset'])} | {row['dev_rows']} | "
            f"{row['test_rows']} | {row['dropped_duplicates']} |"
        )
    lines.append("")
    lines.append(
        "Dev = rows whose split is `train` or `validation`; test = rows whose split is `test`. "
        "A test row is dropped when its normalised text (lowercase, whitespace collapsed) also "
        "appears in dev."
    )
    lines.append("")

    lines.append("## Test evaluation")
    lines.append("")
    lines.append(
        "`classifier alone` uses only the learned model; `regex OR classifier` is the deployed "
        "detector (`LLM01_prompt_injection` raised by the regex layer or by the classifier). "
        "The heuristic scorer is disabled in the second column so the two contributions stay "
        "separable."
    )
    lines.append("")
    lines.extend(_render_eval_table("Classifier alone", payload["classifier_alone"]))
    lines.extend(_render_eval_table("Regex OR classifier", payload["regex_or_classifier"]))

    lines.append("## Leave-one-dataset-out generalization")
    lines.append("")
    if payload["skip_lodo"]:
        lines.append("_Skipped (`--skip-lodo`)._")
        lines.append("")
    else:
        lines.append(
            "For each held-out dataset the model is trained on the *other* datasets' dev rows only "
            "(threshold chosen by CV on that training set) and evaluated on the held-out dataset's "
            "test rows, using the classifier alone."
        )
        lines.append("")
        lines.append(
            "| Held-out dataset | Train rows | Threshold | Test rows | Detection rate (95% CI) | FPR (95% CI) |"
        )
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        for name, metrics in payload["leave_one_dataset_out"].items():
            label = DATASET_LABELS.get(name, name)
            if metrics.get("status"):
                lines.append(f"| {label} | 0 | n/a | 0 | n/a | n/a |")
                continue
            lines.append(
                "| {label} | {train} | {threshold:.4f} | {test} | {dr} | {fpr} |".format(
                    label=label,
                    train=metrics["train_rows"],
                    threshold=metrics["threshold"],
                    test=metrics["test_rows"],
                    dr=_fmt_rate_ci(metrics["detection_rate"], metrics["detection_rate_ci95"]),
                    fpr=_fmt_rate_ci(metrics["fpr"], metrics["fpr_ci95"]),
                )
            )
        lines.append("")

    lines.append("## Notes and limitations")
    lines.append("")
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the stdlib-only prompt-injection classifier."
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="target out-of-fold FPR used to pick the threshold (default: 0.01)",
    )
    parser.add_argument("--epochs", type=int, default=8, help="SGD epochs per fit (default: 8)")
    parser.add_argument(
        "--skip-lodo",
        action="store_true",
        help="skip leave-one-dataset-out generalization (much faster)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--benign-dir",
        type=Path,
        default=DATA_DIR.parent / "data_benign",
        help="extra benign technical prompts (download_benign.py); loaded when present",
    )
    parser.add_argument("--no-benign", action="store_true", help="ignore --benign-dir")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=MODEL_PATH,
        help="where to write the model JSON (default: the packaged model)",
    )
    args = parser.parse_args(argv)
    model_path = args.model_out.resolve()

    run_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    if not data_dir.is_dir():
        print(
            f"[error] no dataset directory at {data_dir}; "
            "run `python3.11 benchmarks/download_datasets.py` first",
            file=sys.stderr,
        )
        return 2

    rows, skipped_too_long = load_rows(data_dir)
    benign_dir = args.benign_dir.resolve()
    if not args.no_benign and benign_dir.is_dir() and benign_dir != data_dir:
        # Benign hard negatives (coding / security / general questions). Their
        # train halves are used for training; their test halves stay held out.
        benign_rows, benign_skipped = load_rows(benign_dir)
        rows.extend(benign_rows)
        skipped_too_long.update(benign_skipped)
        log(f"[info] added {len(benign_rows)} benign rows from {benign_dir.name}/")
    if not rows:
        print(f"[error] no usable rows loaded from {data_dir}", file=sys.stderr)
        return 2

    dev_rows, test_rows = split_rows(rows)
    log(f"[info] loaded {len(rows)} rows: {len(dev_rows)} dev / {len(test_rows)} test")
    if not dev_rows or not test_rows:
        print(
            "[error] need both dev (split train/validation) and test (split test) rows",
            file=sys.stderr,
        )
        return 2

    test_rows, duplicates = dedupe_test(dev_rows, test_rows)
    log(f"[info] dropped {sum(duplicates.values())} test rows duplicated from dev")
    if not test_rows:
        print("[error] no test rows left after de-duplication", file=sys.stderr)
        return 2

    cache = FeatureCache()

    # 1) Threshold selection — cross-validation on dev only.
    log(f"[info] {FOLDS}-fold CV on {len(dev_rows)} dev rows (epochs={args.epochs})")
    out_of_fold = cross_val_probabilities(
        dev_rows, cache, args.epochs, args.seed, log=log
    )
    threshold = choose_threshold(
        out_of_fold, [row["label"] for row in dev_rows], args.target_fpr
    )
    log(f"[info] selected threshold {threshold:.4f} (target FPR <= {args.target_fpr})")

    # 2) Final model on every dev row.
    log(f"[info] training final model on {len(dev_rows)} dev rows")
    model = train(
        [(cache.get(row["text"]), row["label"]) for row in dev_rows],
        epochs=args.epochs,
        seed=args.seed,
    )
    model.threshold = threshold
    trained_on = sorted({row["dataset"] for row in dev_rows})
    model.trained_on = trained_on
    log(f"[info] {len(model.weights)} weights kept after pruning")

    # Persist before evaluating the deployed detector so LLMGuardrails picks the
    # freshly trained model up through the normal (cached) loading path.
    model.save(model_path)
    log(f"[info] wrote {model_path}")

    # 3) Test evaluation.
    log("[info] evaluating on test rows (classifier alone)")
    alone_per_dataset, alone_combined = evaluate_alone(model, threshold, test_rows, cache)
    log("[info] evaluating on test rows (regex OR classifier)")
    regex_per_dataset, regex_combined = evaluate_regex_or_classifier(model, threshold, test_rows)

    classifier_alone = {
        name: metrics_from_confusion(confusion)
        for name, confusion in group_by_dataset(test_rows) and {
            name: alone_per_dataset.get(name, _empty_confusion()) for name, _ in group_by_dataset(test_rows)
        }.items()
    }
    regex_or_classifier = {
        name: metrics_from_confusion(confusion)
        for name, confusion in {
            name: regex_per_dataset.get(name, _empty_confusion()) for name, _ in group_by_dataset(test_rows)
        }.items()
    }
    classifier_alone["combined"] = metrics_from_confusion(alone_combined)
    regex_or_classifier["combined"] = metrics_from_confusion(regex_combined)

    # 4) Leave-one-dataset-out generalization.
    lodo: dict = {}
    if args.skip_lodo:
        log("[info] skipping leave-one-dataset-out (--skip-lodo)")
    else:
        for held_out, _ in group_by_dataset(dev_rows):
            train_rows = [row for row in dev_rows if row["dataset"] != held_out]
            eval_rows = [row for row in test_rows if row["dataset"] == held_out]
            if not train_rows or not eval_rows:
                lodo[held_out] = {"status": "skipped"}
                log(f"[info] LODO {held_out}: skipped (not enough rows)")
                continue
            log(f"[info] LODO {held_out}: training on {len(train_rows)} rows")
            oof = cross_val_probabilities(
                train_rows, cache, args.epochs, args.seed, log=None
            )
            lodo_threshold = choose_threshold(
                oof, [row["label"] for row in train_rows], args.target_fpr
            )
            lodo_model = train(
                [(cache.get(row["text"]), row["label"]) for row in train_rows],
                epochs=args.epochs,
                seed=args.seed,
            )
            _, lodo_confusion = evaluate_alone(lodo_model, lodo_threshold, eval_rows, cache)
            lodo_metrics = metrics_from_confusion(lodo_confusion)
            lodo_metrics.update(
                {
                    "train_rows": len(train_rows),
                    "test_rows": len(eval_rows),
                    "threshold": lodo_threshold,
                }
            )
            lodo[held_out] = lodo_metrics
            log(
                f"[info] LODO {held_out}: detection {_fmt_rate(lodo_metrics['detection_rate'])}, "
                f"FPR {_fmt_rate(lodo_metrics['fpr'])}"
            )

    data_summary = [
        {
            "dataset": name,
            "dev_rows": sum(1 for row in dev_rows if row["dataset"] == name),
            "test_rows": sum(1 for row in test_rows if row["dataset"] == name),
            "dropped_duplicates": duplicates.get(name, 0),
        }
        for name, _ in group_by_dataset(rows)
    ]

    # The shipped model records what it was trained on plus its test metrics.
    model.metrics = {
        "test": {
            name: {
                key: metrics[key]
                for key in ("tp", "fp", "tn", "fn", "detection_rate", "fpr", "precision")
            }
            for name, metrics in classifier_alone.items()
        },
        "regex_or_classifier_test": {
            name: {
                key: metrics[key]
                for key in ("tp", "fp", "tn", "fn", "detection_rate", "fpr", "precision")
            }
            for name, metrics in regex_or_classifier.items()
        },
    }
    model.save(model_path)

    payload = {
        "benchmark": "sentinel-llm-guardrails learned prompt-injection classifier",
        "library_version": __version__,
        "run_date_utc": run_date,
        "seed": args.seed,
        "epochs": args.epochs,
        "folds": FOLDS,
        "target_fpr": args.target_fpr,
        "threshold": threshold,
        "trained_on": trained_on,
        "model_path": _relative(model_path),
        "n_weights": len(model.weights),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "skipped_too_long": skipped_too_long,
        "skip_lodo": bool(args.skip_lodo),
        "data": data_summary,
        "test_rows_dropped_as_dev_duplicates": duplicates,
        "classifier_alone": classifier_alone,
        "regex_or_classifier": regex_or_classifier,
        "leave_one_dataset_out": lodo,
        "notes": [
            "Test rows are never used for training, model selection or threshold selection.",
            "The threshold is the lowest out-of-fold threshold whose FPR on benign dev rows is <= --target-fpr.",
            "LODO trains on the other datasets' dev rows only; it is a generalization check, not a production model.",
            "These numbers come from fixed public corpora and say nothing about robustness against adaptive attackers.",
            "Dataset licences are third-party and must be checked on each dataset page.",
        ],
    }

    json_path = results_dir / "classifier.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(payload)
    md_path = results_dir / "classifier.md"
    md_path.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print()
    print(f"[info] wrote {json_path}")
    print(f"[info] wrote {md_path}")
    print(f"[info] wrote {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
