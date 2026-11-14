#!/usr/bin/env python3.11
"""False-positive rate on benign technical prompts (``benchmarks/data_benign``).

For each source (codealpaca, cyber, dolly) this reports how often benign
prompts are flagged as prompt injection by:

* ``regex-only``  - ``LLMGuardrails(use_classifier=False)`` (pre-0.4.0 stack)
* ``full``        - ``LLMGuardrails()`` (regex + classifier + heuristics)
* ``classifier``  - the classifier alone at its stored threshold

with Wilson 95% intervals, on the ``test`` half and on all rows. The highest-
scoring false positives are written to ``results/benign_fp_sample.jsonl`` so
they can be read by hand.

Usage::

    python3.11 benchmarks/eval_benign.py [--label before] [--model PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from eval_static import wilson_ci  # noqa: E402
from sentinel_llm_guardrails import InjectionClassifier, LLMGuardrails  # noqa: E402

INJECTION = "LLM01_prompt_injection"


def _guard(use_classifier: bool, threshold: float | None, model=None) -> LLMGuardrails:
    return LLMGuardrails(
        daily_request_limit=10**9,
        daily_token_limit=10**12,
        audit_enabled=False,
        use_classifier=use_classifier,
        classifier_threshold=threshold,
        classifier=model,
    )


def _rate(flagged: int, total: int) -> dict:
    return {
        "flagged": flagged,
        "n": total,
        "fpr": flagged / total if total else None,
        "ci95": wilson_ci(flagged, total),
    }


def _fmt(cell: dict) -> str:
    if not cell["n"]:
        return "n/a"
    lo, hi = cell["ci95"]
    return f"{100 * cell['fpr']:.2f}% [{100 * lo:.2f}–{100 * hi:.2f}] ({cell['flagged']}/{cell['n']})"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=HERE / "data_benign")
    parser.add_argument("--model", type=Path, default=None, help="model JSON (default: packaged)")
    parser.add_argument("--label", default="current")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "benign.json")
    args = parser.parse_args(argv)

    model = InjectionClassifier.from_file(args.model) if args.model else InjectionClassifier.default()
    if model is None:
        print("classifier model not found", file=sys.stderr)
        return 1
    threshold = model.threshold
    regex_only = _guard(False, None)
    full = _guard(True, threshold, model)

    results: dict = {"label": args.label, "threshold": threshold, "sources": {}}
    false_positives: list[dict] = []
    for path in sorted(args.data_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        counts = {split: {"regex": 0, "full": 0, "clf": 0, "n": 0} for split in ("test", "all")}
        for row in rows:
            text = row["text"]
            p = model.predict_proba(text)
            hit_regex = INJECTION in regex_only.check_input(text, user_id="b").checks_failed
            hit_full = INJECTION in full.check_input(text, user_id="b").checks_failed
            for split in ("all", row["split"]):
                if split not in counts:
                    continue
                c = counts[split]
                c["n"] += 1
                c["regex"] += hit_regex
                c["full"] += hit_full
                c["clf"] += p >= threshold
            if hit_full:
                false_positives.append(
                    {"source": path.stem, "split": row["split"], "p": round(p, 3),
                     "regex": hit_regex, "text": text[:400]}
                )
        results["sources"][path.stem] = {
            split: {key: _rate(c[key], c["n"]) for key in ("regex", "full", "clf")}
            for split, c in counts.items()
        }

    # Pooled rows across sources.
    pooled = {}
    for split in ("test", "all"):
        pooled[split] = {}
        for key in ("regex", "full", "clf"):
            flagged = sum(s[split][key]["flagged"] for s in results["sources"].values())
            n = sum(s[split][key]["n"] for s in results["sources"].values())
            pooled[split][key] = _rate(flagged, n)
    results["sources"]["ALL"] = pooled

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    false_positives.sort(key=lambda r: -r["p"])
    sample = args.out.with_name("benign_fp_sample.jsonl")
    with sample.open("w", encoding="utf-8") as handle:
        for row in false_positives[:60]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"## Benign false-positive rate ({args.label}, threshold {threshold:.4f})\n")
    for split in ("test", "all"):
        print(f"### split: {split}\n")
        print("| source | regex-only | full stack | classifier alone |")
        print("|---|---|---|---|")
        for name, data in results["sources"].items():
            d = data[split]
            print(f"| {name} | {_fmt(d['regex'])} | {_fmt(d['full'])} | {_fmt(d['clf'])} |")
        print()
    print(f"{len(false_positives)} full-stack false positives; top 60 in {sample.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
