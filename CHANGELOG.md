# Changelog

## [0.4.0] — 2026-11-10

### Added
- Learned prompt-injection classifier (`sentinel_llm_guardrails.classifier`): hashed word/char n-grams + logistic regression, stdlib only, ~1.2 MB model in `models/injection_lr.json`. Runs after the regex layer. See `docs/MODEL_CARD.md`.
- `LLMGuardrails(use_classifier=True, classifier_threshold=None)` options.
- `calibrate_threshold(model, benign_texts, target_fpr)` to choose a threshold on your own normal traffic.
- Long inputs are scored in overlapping 4000-character windows (max) instead of only the first 4000 characters. This does **not** make injections hidden in long text detectable (0% in `benchmarks/eval_long_inputs.py`); see Known limitations.
- `LLMGuardrails(classifier=...)` to supply your own `InjectionClassifier`.
- Benign "hard negative" training data (coding, security and general questions; `benchmarks/download_benign.py`) and `benchmarks/eval_benign.py`, `benchmarks/eval_long_inputs.py`.
- `benchmarks/`: dataset download, static evaluation and classifier training with held-out test and leave-one-dataset-out checks.
- Regression tests for homoglyph normalization, window scoring and benign technical prompts.

### Changed
- **The classifier is on by default.** On held-out public test data, detection rose from 18.7% to 97.0%; 1.15% of benign prompts in those datasets and 0.51% of benign coding/security/general prompts are now flagged (previously about 0%). Pass `use_classifier=False` to keep 0.3.x behaviour, or calibrate the threshold.

### Known limitations
- Weak cross-dataset generalization (leave-one-dataset-out detection 29–99%, one dataset at 21% false positives). Trained on direct user prompts only, mostly English; injections hidden inside long text are almost never caught (0.2% at 4000-character windows; 35% at best with 400-character windows at a 2% false-positive rate); not evaluated on real tool outputs or against adaptive attackers.

## [0.3.1] — 2026-08-05

### Added
- `LLMGuardrailsService.check_input/check_output/check_tool` now record per-tenant latency + audit.
- `audit_enabled`, `audit_dir`, `latency_budget_ms`, `latency_on_breach` service options.
- `GuardrailResult` carries `tenant_id`, `latency`, and `audit_seq`.
- Restored full service integration test suite.

## [0.3.0] — 2026-08-05

### Added
- Latency budgets with warn / fail_open / fail_closed breach policies.
- Per-tenant, append-only, hash-chained JSONL audit trail.
- Prompt digests and HMAC tamper verification instead of storing raw prompts.
- `sentinel_llm_guardrails.telemetry` module (stdlib only).

## [0.2.0] — 2026-08-01

### Added
- Evasion normalization (zero-width, NFKC, homoglyphs, leet, spelled-out words)
- Heuristic injection risk score with configurable threshold
- Expanded secret patterns (GitHub, Slack, Stripe, OpenAI, JWT)
- Hardening tests and CI workflow

### Changed
- Broader OWASP LLM01 injection regex catalogue

## [0.1.0] — 2026-07-31

### Added
- Initial MIT release: input/output checks, PII redaction, rate limits, OPA policy
