# sentinel-llm-guardrails — learned prompt-injection classifier

Training and evaluation report for the stdlib-only logistic-regression classifier shipped in `src/sentinel_llm_guardrails/models/injection_lr.json`.

- Library version: `0.4.0`
- Run date (UTC): `2026-11-10T02:25:20Z`
- Random seed: `0`
- SGD epochs per fit: `8`
- CV folds: `5`
- Target out-of-fold FPR for threshold selection: `0.01`
- Chosen threshold (from dev CV): `0.4267`
- Trained on dev datasets: `deepset, gandalf, jackhhao, safeguard`
- Model file: `src/sentinel_llm_guardrails/models/injection_lr.json` (60000 stored weights)
- Dev rows: `10714`, test rows after de-duplication: `2397`

## Data

| Dataset | Dev rows | Test rows | Test rows dropped as dev duplicates |
| --- | ---: | ---: | ---: |
| deepset/prompt-injections | 546 | 115 | 1 |
| jackhhao/jailbreak-classification | 1044 | 255 | 7 |
| Lakera/gandalf_ignore_instructions | 888 | 112 | 0 |
| xTRam1/safe-guard-prompt-injection | 8236 | 1915 | 145 |

Dev = rows whose split is `train` or `validation`; test = rows whose split is `test`. A test row is dropped when its normalised text (lowercase, whitespace collapsed) also appears in dev.

## Test evaluation

`classifier alone` uses only the learned model; `regex OR classifier` is the deployed detector (`LLM01_prompt_injection` raised by the regex layer or by the classifier). The heuristic scorer is disabled in the second column so the two contributions stay separable.

### Classifier alone

| Dataset | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) | Precision |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| deepset/prompt-injections | 44 | 3 | 53 | 15 | 74.58% [62.20%, 83.94%] | 5.36% [1.84%, 14.61%] | 93.62% |
| jackhhao/jailbreak-classification | 131 | 4 | 118 | 2 | 98.50% [94.68%, 99.59%] | 3.28% [1.28%, 8.13%] | 97.04% |
| Lakera/gandalf_ignore_instructions | 111 | 0 | 0 | 1 | 99.11% [95.12%, 99.84%] | n/a | 100.00% |
| xTRam1/safe-guard-prompt-injection | 527 | 19 | 1367 | 2 | 99.62% [98.63%, 99.90%] | 1.37% [0.88%, 2.13%] | 96.52% |
| **combined** | 813 | 26 | 1538 | 20 | 97.60% [96.32%, 98.44%] | 1.66% [1.14%, 2.42%] | 96.90% |

### Regex OR classifier

| Dataset | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) | Precision |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| deepset/prompt-injections | 44 | 3 | 53 | 15 | 74.58% [62.20%, 83.94%] | 5.36% [1.84%, 14.61%] | 93.62% |
| jackhhao/jailbreak-classification | 131 | 4 | 118 | 2 | 98.50% [94.68%, 99.59%] | 3.28% [1.28%, 8.13%] | 97.04% |
| Lakera/gandalf_ignore_instructions | 111 | 0 | 0 | 1 | 99.11% [95.12%, 99.84%] | n/a | 100.00% |
| xTRam1/safe-guard-prompt-injection | 527 | 19 | 1367 | 2 | 99.62% [98.63%, 99.90%] | 1.37% [0.88%, 2.13%] | 96.52% |
| **combined** | 813 | 26 | 1538 | 20 | 97.60% [96.32%, 98.44%] | 1.66% [1.14%, 2.42%] | 96.90% |

## Leave-one-dataset-out generalization

For each held-out dataset the model is trained on the *other* datasets' dev rows only (threshold chosen by CV on that training set) and evaluated on the held-out dataset's test rows, using the classifier alone.

| Held-out dataset | Train rows | Threshold | Test rows | Detection rate (95% CI) | FPR (95% CI) |
| --- | ---: | ---: | ---: | --- | --- |
| deepset/prompt-injections | 10168 | 0.3212 | 115 | 28.81% [18.84%, 41.38%] | 0.00% [0.00%, 6.42%] |
| jackhhao/jailbreak-classification | 9670 | 0.3511 | 255 | 98.50% [94.68%, 99.59%] | 21.31% [14.98%, 29.40%] |
| Lakera/gandalf_ignore_instructions | 9826 | 0.4384 | 112 | 92.86% [86.54%, 96.34%] | n/a |
| xTRam1/safe-guard-prompt-injection | 2478 | 0.5549 | 1915 | 55.77% [51.51%, 59.94%] | 2.09% [1.46%, 2.99%] |

## Notes and limitations

- Test rows are never used for training, model selection or threshold selection.
- The threshold is the lowest out-of-fold threshold whose FPR on benign dev rows is <= --target-fpr.
- LODO trains on the other datasets' dev rows only; it is a generalization check, not a production model.
- These numbers come from fixed public corpora and say nothing about robustness against adaptive attackers.
- Dataset licences are third-party and must be checked on each dataset page.

