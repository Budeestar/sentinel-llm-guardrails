# Benchmarks

Dependency-free scripts (Python ≥ 3.10 standard library) that measure the
prompt-injection detector of `sentinel-llm-guardrails` on public datasets.
Results for the shipped model are in [`results/`](results/) and summarised in
[`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md).

## Data

| Script | Downloads | Into |
|---|---|---|
| `download_datasets.py` | `deepset/prompt-injections`, `jackhhao/jailbreak-classification`, `Lakera/gandalf_ignore_instructions`, `xTRam1/safe-guard-prompt-injection` (attacks + benign) | `data/` |
| `download_benign.py` | 2,000-row deterministic samples of `sahil2801/CodeAlpaca-20k`, `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`, `databricks/databricks-dolly-15k` (benign coding / security / general prompts, split 50/50 by text hash) | `data_benign/` |

Both use the Hugging Face datasets-server API and are not committed. The
datasets are third-party: check each licence on its Hugging Face page.

## Protocol

- **Dev** = rows whose split is `train` or `validation`; **test** = rows whose split is `test`. Test rows are never used for training, model selection or threshold choice.
- Test rows whose normalised text also appears in dev are dropped.
- The threshold is the lowest one whose **out-of-fold** false-positive rate (5-fold cross-validation on dev) is ≤ 1%.
- Rates are reported with Wilson 95% confidence intervals.

## Scripts

| Script | Measures | Output |
|---|---|---|
| `train_classifier.py` | Trains the classifier on dev (injection datasets + `data_benign/` train halves), evaluates on test (classifier alone, and regex OR classifier), and runs leave-one-dataset-out generalisation | `results/classifier.md`, `.json`; the packaged model (or `--model-out`) |
| `eval_static.py` | The full guardrail per dataset: confusion matrix, detection rate, FPR, precision, detections by mechanism, heuristic threshold sweep; `--split {all,dev,test}`, `--no-classifier` | `results/static.md`, `.json` |
| `eval_benign.py` | False-positive rate on the benign coding / security / general prompts (regex-only, full stack, classifier alone) | `results/benign.json` |
| `eval_long_inputs.py` | Detection of attacks hidden inside 5–12k characters of benign text, for several window sizes, with thresholds calibrated on dev | `results/long_inputs.md`, `.json` |

## Reproduce

    python3.11 benchmarks/download_datasets.py
    python3.11 benchmarks/download_benign.py
    python3.11 benchmarks/train_classifier.py          # ~10-15 min on a laptop
    python3.11 benchmarks/eval_static.py
    python3.11 benchmarks/eval_benign.py
    python3.11 benchmarks/eval_long_inputs.py

`results/v0.4.0-pre-hardneg/` keeps the results of the first 0.4.0 model,
trained without the benign hard negatives, for comparison.

## Limits

These are static, public corpora of mostly English direct prompts. They say
nothing about an adaptive attacker who optimises against this detector, and
the long-input results show that attacks embedded in documents are mostly
missed. See the model card for details.
