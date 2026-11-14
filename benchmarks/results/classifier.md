# sentinel-llm-guardrails — learned prompt-injection classifier

Training and evaluation report for the stdlib-only logistic-regression classifier shipped in `src/sentinel_llm_guardrails/models/injection_lr.json`.

- Library version: `0.4.0`
- Run date (UTC): `2026-11-10T06:14:45Z`
- Random seed: `0`
- SGD epochs per fit: `8`
- CV folds: `5`
- Target out-of-fold FPR for threshold selection: `0.01`
- Chosen threshold (from dev CV): `0.3670`
- Trained on dev datasets: `codealpaca, cyber, deepset, dolly, gandalf, jackhhao, safeguard`
- Model file: `src/sentinel_llm_guardrails/models/injection_lr.json` (60000 stored weights)
- Dev rows: `13699`, test rows after de-duplication: `5323`

## Data

| Dataset | Dev rows | Test rows | Test rows dropped as dev duplicates |
| --- | ---: | ---: | ---: |
| deepset/prompt-injections | 546 | 115 | 1 |
| jackhhao/jailbreak-classification | 1044 | 255 | 7 |
| Lakera/gandalf_ignore_instructions | 888 | 112 | 0 |
| xTRam1/safe-guard-prompt-injection | 8236 | 1914 | 146 |
| codealpaca | 981 | 1019 | 0 |
| cyber | 951 | 976 | 0 |
| dolly | 1053 | 932 | 12 |

Dev = rows whose split is `train` or `validation`; test = rows whose split is `test`. A test row is dropped when its normalised text (lowercase, whitespace collapsed) also appears in dev.

## Test evaluation

`classifier alone` uses only the learned model; `regex OR classifier` is the deployed detector (`LLM01_prompt_injection` raised by the regex layer or by the classifier). The heuristic scorer is disabled in the second column so the two contributions stay separable.

### Classifier alone

| Dataset | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) | Precision |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| deepset/prompt-injections | 41 | 2 | 54 | 18 | 69.49% [56.85%, 79.75%] | 3.57% [0.98%, 12.12%] | 95.35% |
| jackhhao/jailbreak-classification | 130 | 4 | 118 | 3 | 97.74% [93.58%, 99.23%] | 3.28% [1.28%, 8.13%] | 97.01% |
| Lakera/gandalf_ignore_instructions | 110 | 0 | 0 | 2 | 98.21% [93.72%, 99.51%] | n/a | 100.00% |
| xTRam1/safe-guard-prompt-injection | 527 | 12 | 1373 | 2 | 99.62% [98.63%, 99.90%] | 0.87% [0.50%, 1.51%] | 97.77% |
| codealpaca | 0 | 5 | 1014 | 0 | n/a | 0.49% [0.21%, 1.14%] | 0.00% |
| cyber | 0 | 3 | 973 | 0 | n/a | 0.31% [0.10%, 0.90%] | 0.00% |
| dolly | 0 | 5 | 927 | 0 | n/a | 0.54% [0.23%, 1.25%] | 0.00% |
| **combined** | 808 | 31 | 4459 | 25 | 97.00% [95.61%, 97.96%] | 0.69% [0.49%, 0.98%] | 96.31% |

### Regex OR classifier

| Dataset | TP | FP | TN | FN | Detection rate (95% CI) | FPR (95% CI) | Precision |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| deepset/prompt-injections | 41 | 2 | 54 | 18 | 69.49% [56.85%, 79.75%] | 3.57% [0.98%, 12.12%] | 95.35% |
| jackhhao/jailbreak-classification | 130 | 4 | 118 | 3 | 97.74% [93.58%, 99.23%] | 3.28% [1.28%, 8.13%] | 97.01% |
| Lakera/gandalf_ignore_instructions | 110 | 0 | 0 | 2 | 98.21% [93.72%, 99.51%] | n/a | 100.00% |
| xTRam1/safe-guard-prompt-injection | 527 | 12 | 1373 | 2 | 99.62% [98.63%, 99.90%] | 0.87% [0.50%, 1.51%] | 97.77% |
| codealpaca | 0 | 5 | 1014 | 0 | n/a | 0.49% [0.21%, 1.14%] | 0.00% |
| cyber | 0 | 5 | 971 | 0 | n/a | 0.51% [0.22%, 1.19%] | 0.00% |
| dolly | 0 | 5 | 927 | 0 | n/a | 0.54% [0.23%, 1.25%] | 0.00% |
| **combined** | 808 | 33 | 4457 | 25 | 97.00% [95.61%, 97.96%] | 0.73% [0.52%, 1.03%] | 96.08% |

## Leave-one-dataset-out generalization

For each held-out dataset the model is trained on the *other* datasets' dev rows only (threshold chosen by CV on that training set) and evaluated on the held-out dataset's test rows, using the classifier alone.

| Held-out dataset | Train rows | Threshold | Test rows | Detection rate (95% CI) | FPR (95% CI) |
| --- | ---: | ---: | ---: | --- | --- |
| deepset/prompt-injections | 13153 | 0.2703 | 115 | 30.51% [20.25%, 43.15%] | 0.00% [0.00%, 6.42%] |
| jackhhao/jailbreak-classification | 12655 | 0.3820 | 255 | 95.49% [90.51%, 97.92%] | 9.02% [5.11%, 15.43%] |
| Lakera/gandalf_ignore_instructions | 12811 | 0.3601 | 112 | 91.07% [84.34%, 95.08%] | n/a |
| xTRam1/safe-guard-prompt-injection | 5463 | 0.4819 | 1914 | 67.11% [62.99%, 70.98%] | 6.35% [5.19%, 7.76%] |
| codealpaca | 12718 | 0.4083 | 1019 | n/a | 2.55% [1.75%, 3.71%] |
| cyber | 12748 | 0.3729 | 976 | n/a | 6.15% [4.81%, 7.83%] |
| dolly | 12646 | 0.3848 | 932 | n/a | 0.32% [0.11%, 0.94%] |

## Notes and limitations

- Test rows are never used for training, model selection or threshold selection.
- The threshold is the lowest out-of-fold threshold whose FPR on benign dev rows is <= --target-fpr.
- LODO trains on the other datasets' dev rows only; it is a generalization check, not a production model.
- These numbers come from fixed public corpora and say nothing about robustness against adaptive attackers.
- Dataset licences are third-party and must be checked on each dataset page.

