# Model card: prompt-injection classifier

- **Artifact:** `src/sentinel_llm_guardrails/models/injection_lr.json` (~1.2 MB JSON)
- **Code:** `src/sentinel_llm_guardrails/classifier.py` (standard library only)
- **Task:** classify a user prompt as "prompt injection" (OWASP LLM01) or benign
- **Licence:** MIT; trained on third-party datasets whose licences must be checked separately

## Model

- **Features:** hashed word 1-2-gram plus character 3-5-gram features, `2^20` (1,048,576) CRC32 buckets, L2-normalised.
- **Estimator:** logistic regression trained with plain SGD, 8 epochs, class-balanced sample weights.
- **Size:** 60,000 pruned weights (anything below `1e-4` is dropped), serialised as ~1.2 MB of JSON.
- **Runtime:** standard library only, no network access, roughly 0.2 ms per check on an Apple M-series laptop.
- **Long inputs:** scored in overlapping 4000-character windows (2000-character stride, at most 16 windows) and the **maximum** window probability is returned. This removes the hard 4000-character cut-off, but it does **not** make hidden injections detectable: see "Injections inside long text" below.

## Training data

Train/validation splits only, 13,699 rows in total.

Injection datasets (attacks and benign prompts):

- `deepset/prompt-injections`
- `jackhhao/jailbreak-classification`
- `Lakera/gandalf_ignore_instructions`
- `xTRam1/safe-guard-prompt-injection`

Benign "hard negatives" (train half only; `benchmarks/download_benign.py`, 2,000-row deterministic samples split 50/50 by text hash):

- `sahil2801/CodeAlpaca-20k` (coding tasks, CC-BY-4.0)
- `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` (security questions, Apache-2.0)
- `databricks/databricks-dolly-15k` (general questions, CC-BY-SA-3.0)

These were added because the first 0.4.0 model, trained on the injection datasets alone, flagged 8.7% of held-out coding prompts and 20.1% of held-out security questions (e.g. "Write a Python function that validates a credit card with Luhn", "Create a table to store customer information").

**Decision threshold:** `0.3670`, chosen by 5-fold cross-validation so that at most 1% of out-of-fold benign rows are flagged.

## Held-out evaluation

Test splits, after removing rows that duplicated training rows. Regex OR classifier (the deployed decision); 95% Wilson intervals.

| Test set | Metric | This model | First 0.4.0 model (no hard negatives) | Regex only |
| --- | --- | --- | --- | --- |
| 4 injection datasets (833 attacks, 1,563 benign) | Detection | **97.00%** (95.61-97.96) | 97.60% | 18.73% |
| | False positives | **1.15%** (0.73-1.81) | 1.66% | 0.00% |
| Benign technical prompts (2,927; coding, security, general) | False positives | **0.51%** (0.31-0.84) | 10.1% | 0.07% |

Leave-one-dataset-out (train on the other datasets, test on the held-out one):

| Held-out dataset | Detection | FPR |
| --- | --- | --- |
| `deepset/prompt-injections` | 30.51% | 0.00% |
| `jackhhao/jailbreak-classification` | 95.49% | 9.02% |
| `Lakera/gandalf_ignore_instructions` | 91.07% | n/a |
| `xTRam1/safe-guard-prompt-injection` | 67.11% | 6.35% |
| codealpaca (benign only) | n/a | 2.55% |
| cyber (benign only) | n/a | 6.15% |
| dolly (benign only) | n/a | 0.32% |

## Injections inside long text

`benchmarks/eval_long_inputs.py` hides one held-out attack (< 600 characters) inside 5k-12k characters of held-out benign prompts, as its own paragraph or glued inline. Each window scheme's threshold is calibrated on separately built dev texts to <= 1% false positives.

| Window | Detection (paragraph / inline) | Long-benign FPR |
| --- | --- | --- |
| 4000/2000 (shipped) | 0.2% / 0.0% | 0.0% |
| 400/200 (best tried) | 35.0% / 35.2% | 2.0% |

The same attacks on their own are caught about 97% of the time. A single linear model over bag-of-features is diluted by the surrounding benign text; detecting injections embedded in documents needs a different approach. Full table: `benchmarks/results/long_inputs.md`.


## Limitations

- Trained on **direct user prompts only**. It does not detect injections embedded in long documents (see above), and it has not been tested on real tool outputs or web pages.
- Cross-dataset generalisation is weak: leave-one-dataset-out detection ranges from 30.51% to 95.49%, and benign prompts from an unseen domain are flagged at up to 6.15%.
- Mostly English; non-English traffic is out of scope for the published numbers.
- These are static benchmark numbers and say nothing about an adaptive attacker who optimises against this model.
- The dataset licences are third-party. Check them before redistributing the data or the derived weights.
- The default threshold flags roughly 0.5-1.2% of benign prompts on the test sets above; your traffic may differ. Calibrate on your own logs with `calibrate_threshold(model, benign_texts, target_fpr=0.01)` and pass the result to `LLMGuardrails(classifier_threshold=t)`, or disable the model entirely with `LLMGuardrails(use_classifier=False)`.

## Reproducing

    python3.11 benchmarks/download_datasets.py && python3.11 benchmarks/download_benign.py
    python3.11 benchmarks/train_classifier.py
    python3.11 benchmarks/eval_static.py && python3.11 benchmarks/eval_benign.py && python3.11 benchmarks/eval_long_inputs.py
