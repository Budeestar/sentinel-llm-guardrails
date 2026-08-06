# Changelog

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
