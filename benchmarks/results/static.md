# sentinel-llm-guardrails — static prompt-injection detection benchmark

- Library version: `0.4.0`
- Run date (UTC): `2026-11-10T06:27:59Z`
- Split: `test`
- Injection score threshold: `0.45`
- Random seed: `0`
- Learned classifier: `enabled (results are also repeated with it disabled below)`
- Texts longer than 32000 characters: skipped.
- Detection rule: a text is *flagged* only if `"LLM01_prompt_injection" in result.checks_failed`; any other block (secrets, PII, ...) is recorded as `other_block` and never as a detection.

## Detection metrics per dataset — injection classifier ON

Shipped default configuration: regex → learned classifier → heuristic score.

| Dataset | Evaluated | Positives | Negatives | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) | Precision | F1 | Skipped too long | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| deepset/prompt-injections | 116 | 60 | 56 | 41 | 2 | 54 | 19 | 68.33% [55.77%, 78.69%] | 3.57% [0.98%, 12.12%] | 95.35% | 79.61% | 0 | 0 |
| jackhhao/jailbreak-classification | 262 | 139 | 123 | 136 | 4 | 119 | 3 | 97.84% [93.85%, 99.26%] | 3.25% [1.27%, 8.06%] | 97.14% | 97.49% | 0 | 0 |
| Lakera/gandalf_ignore_instructions | 112 | 112 | 0 | 110 | 0 | 0 | 2 | 98.21% [93.72%, 99.51%] | n/a | 100.00% | 99.10% | 0 | 0 |
| xTRam1/safe-guard-prompt-injection | 2060 | 650 | 1410 | 647 | 12 | 1398 | 3 | 99.54% [98.65%, 99.84%] | 0.85% [0.49%, 1.48%] | 98.18% | 98.85% | 0 | 0 |
| **combined** | 2550 | 961 | 1589 | 934 | 18 | 1571 | 27 | 97.19% [95.94%, 98.06%] | 1.13% [0.72%, 1.78%] | 98.11% | 97.65% | 0 | 0 |

## Detection metrics per dataset — injection classifier OFF (regex + heuristic score only)

| Dataset | Evaluated | Positives | Negatives | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) | Precision | F1 | Skipped too long | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| deepset/prompt-injections | 116 | 60 | 56 | 0 | 0 | 56 | 60 | 0.00% [0.00%, 6.02%] | 0.00% [0.00%, 6.42%] | n/a | n/a | 0 | 0 |
| jackhhao/jailbreak-classification | 262 | 139 | 123 | 48 | 0 | 123 | 91 | 34.53% [27.14%, 42.76%] | 0.00% [0.00%, 3.03%] | 100.00% | 51.34% | 0 | 0 |
| Lakera/gandalf_ignore_instructions | 112 | 112 | 0 | 24 | 0 | 0 | 88 | 21.43% [14.84%, 29.91%] | n/a | 100.00% | 35.29% | 0 | 0 |
| xTRam1/safe-guard-prompt-injection | 2060 | 650 | 1410 | 108 | 0 | 1410 | 542 | 16.62% [13.95%, 19.67%] | 0.00% [0.00%, 0.27%] | 100.00% | 28.50% | 0 | 0 |
| **combined** | 2550 | 961 | 1589 | 180 | 0 | 1589 | 781 | 18.73% [16.39%, 21.32%] | 0.00% [0.00%, 0.24%] | 100.00% | 31.55% | 0 | 0 |

`Detection rate` is recall on label-1 rows, `FPR` is the false-positive rate on label-0 rows. `n/a` marks a metric whose denominator is zero — `Lakera/gandalf_ignore_instructions` only contains injection rows, so its FPR, precision and F1 are undefined.

## Detections by mechanism (classifier ON)

A detection is attributed to `regex_injection` when `"regex_injection" in result.risk_reasons`, to `classifier` when the learned model crossed its threshold, otherwise to the heuristic injection score.

| Dataset | TP via regex | TP via classifier | TP via heuristic score | FP via regex | FP via classifier | FP via heuristic score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepset/prompt-injections | 0 | 41 | 0 | 0 | 2 | 0 |
| jackhhao/jailbreak-classification | 48 | 88 | 0 | 0 | 4 | 0 |
| Lakera/gandalf_ignore_instructions | 24 | 86 | 0 | 0 | 0 | 0 |
| xTRam1/safe-guard-prompt-injection | 108 | 539 | 0 | 0 | 12 | 0 |
| **combined** | 180 | 754 | 0 | 0 | 18 | 0 |

## Threshold sweep (regex + heuristic only, combined datasets)

`injection_score_threshold` only affects the heuristic scorer, so the sweep is run with the classifier disabled.

| Threshold | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.25 | 180 | 0 | 1589 | 781 | 18.73% [16.39%, 21.32%] | 0.00% [0.00%, 0.24%] |
| 0.35 | 180 | 0 | 1589 | 781 | 18.73% [16.39%, 21.32%] | 0.00% [0.00%, 0.24%] |
| 0.45 | 180 | 0 | 1589 | 781 | 18.73% [16.39%, 21.32%] | 0.00% [0.00%, 0.24%] |
| 0.55 | 180 | 0 | 1589 | 781 | 18.73% [16.39%, 21.32%] | 0.00% [0.00%, 0.24%] |
| 0.65 | 180 | 0 | 1589 | 781 | 18.73% [16.39%, 21.32%] | 0.00% [0.00%, 0.24%] |

## Error analysis sample

Up to 40 random false negatives and 40 random false positives (texts truncated to 300 characters) are written to `benchmarks/results/static_errors_sample.jsonl`.
Sampled 40 / 781 false negatives and 0 / 0 false positives from the `with_classifier` configuration.

## Limitations

- This is a **static** benchmark: it measures detection on fixed labelled corpora and says nothing about robustness against attackers who adapt to the detector (paraphrasing, encoding, multi-turn setups, tool-mediated injections). For adaptive-attack claims, cite published work instead of these numbers.
- The learned classifier is trained offline (see `benchmarks/train_classifier.py`) and its test rows are excluded from training; these numbers are only as honest as the split field in the source corpora.
- The corpora are third-party; check every dataset page for licence and intended use before publishing numbers.
- Labels reflect the dataset authors' judgement; a prompt that is benign in one context can be an attack in another, so precision/recall here are corpus-specific.
