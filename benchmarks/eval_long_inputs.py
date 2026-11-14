#!/usr/bin/env python3.11
"""Measure detection of injections hidden inside long benign text.

Short prompts are what the classifier was trained on. This benchmark asks what
happens when one of those attacks is embedded in a long document instead.

Construction (deterministic, seeded):

* **long benign**: benign prompts concatenated (blank-line separated) up to
  5k, 8k or 12k characters;
* **long attack, paragraph**: the same filler with one attack prompt
  (< 600 chars) inserted as its own paragraph at a random position;
* **long attack, inline**: the attack glued with a space onto a benign piece,
  so splitting on blank lines gives no free help.

Benign filler and attacks come from rows with split ``train``/``validation``
(DEV, used only to calibrate each scheme's threshold to <= 1% FPR on long
benign) or ``test`` (TEST, used for the reported numbers). The threshold is
never set below the model's own threshold.

Schemes compared: the shipped sliding window (4000 chars, stride 2000) and
smaller sliding windows. Results go to ``results/long_inputs.md`` and ``.json``.

Usage::

    python3.11 benchmarks/eval_long_inputs.py [--n-test 400] [--model PATH]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from eval_static import wilson_ci  # noqa: E402
from sentinel_llm_guardrails.classifier import InjectionClassifier, featurize  # noqa: E402

SCHEMES = {
    "4000/2000 (shipped)": (4000, 2000, 16),
    "1000/500": (1000, 500, 64),
    "600/300": (600, 300, 112),
    "400/200": (400, 200, 160),
}
LENGTHS = (5000, 8000, 12000)


def load(dirs: list[Path]) -> list[dict]:
    rows = []
    for directory in dirs:
        for path in sorted(directory.glob("*.jsonl")):
            rows += [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return rows


def pools(rows: list[dict], splits: set[str]) -> tuple[list[str], list[str]]:
    usable = [r for r in rows if r["split"] in splits and len(r["text"]) < 2000]
    benign = [r["text"] for r in usable if int(r["label"]) == 0]
    attacks = [r["text"] for r in usable if int(r["label"]) == 1 and len(r["text"]) < 600]
    return benign, attacks


def build(benign: list[str], attacks: list[str], n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        target = rng.choice(LENGTHS)
        parts, size = [], 0
        while size < target:
            piece = rng.choice(benign)
            parts.append(piece)
            size += len(piece) + 2
        attack = rng.choice(attacks)
        k = rng.randrange(len(parts))
        inline = list(parts)
        inline[k] = inline[k] + " " + attack
        cases.append(("benign", "\n\n".join(parts)))
        cases.append(("paragraph", "\n\n".join(parts[:k] + [attack] + parts[k:])))
        cases.append(("inline", "\n\n".join(inline)))
    return cases


def windows(text: str, size: int, stride: int, cap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    last = len(text) - size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    if len(starts) > cap:  # spread evenly, keeping the first and last window
        starts = [starts[round(i * (len(starts) - 1) / (cap - 1))] for i in range(cap)]
    return [text[s:s + size] for s in starts]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--n-dev", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)

    model = InjectionClassifier.from_file(args.model) if args.model else InjectionClassifier.default()
    rows = load([HERE / "data", HERE / "data_benign"])
    dev = build(*pools(rows, {"train", "validation"}), args.n_dev, seed=1)
    test = build(*pools(rows, {"test"}), args.n_test, seed=2)

    def score(text: str, scheme: tuple[int, int, int]) -> float:
        return max(model.predict_proba_features(featurize(w)) for w in windows(text, *scheme))

    results = {"model_threshold": model.threshold, "n_test_per_kind": args.n_test, "schemes": {}}
    for name, scheme in SCHEMES.items():
        dev_benign = sorted(score(t, scheme) for kind, t in dev if kind == "benign")
        threshold = max(model.threshold, dev_benign[int(0.99 * len(dev_benign))] + 1e-9)
        counts = {"benign": [0, 0], "paragraph": [0, 0], "inline": [0, 0]}
        start = time.perf_counter()
        for kind, text in test:
            counts[kind][1] += 1
            counts[kind][0] += score(text, scheme) >= threshold
        ms = 1000 * (time.perf_counter() - start) / len(test)
        results["schemes"][name] = {
            "threshold": threshold,
            "ms_per_text": ms,
            **{k: {"hits": h, "n": n, "rate": h / n, "ci95": wilson_ci(h, n)} for k, (h, n) in counts.items()},
        }

    def cell(d: dict) -> str:
        lo, hi = d["ci95"]
        return f"{100 * d['rate']:.1f}% [{100 * lo:.1f}–{100 * hi:.1f}]"

    lines = [
        "# Injections hidden in long benign text",
        "",
        f"Model threshold {model.threshold:.4f}. Each scheme's threshold is calibrated on DEV-built "
        f"long benign text to <= 1% FPR (never below the model threshold); numbers below are on "
        f"TEST-built texts ({args.n_test} of each kind, 5k/8k/12k chars). See the script docstring "
        "for the construction.",
        "",
        "| Window (size/stride) | Threshold | Detection, own paragraph | Detection, inline | Long-benign FPR | ms/text |",
        "| --- | ---: | --- | --- | --- | ---: |",
    ]
    for name, r in results["schemes"].items():
        lines.append(
            f"| {name} | {r['threshold']:.3f} | {cell(r['paragraph'])} | {cell(r['inline'])} | "
            f"{cell(r['benign'])} | {r['ms_per_text']:.0f} |"
        )
    lines += [
        "",
        "For comparison, the same attack prompts on their own are detected at about 97% "
        "(results/classifier.md). A bag-of-features classifier averages one short attack with "
        "thousands of benign characters, so the signal is diluted; smaller windows recover some "
        "of it, but at a calibrated threshold most hidden attacks are still missed.",
    ]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "long_inputs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.results_dir / "long_inputs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
