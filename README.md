# sentinel-llm-guardrails

[![ci](https://github.com/Budeestar/sentinel-llm-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/Budeestar/sentinel-llm-guardrails/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version: 0.3.1](https://img.shields.io/badge/Version-0.3.1-blue.svg)](CHANGELOG.md)

**Hardened, dependency-free OWASP LLM Top 10 guardrails (MIT).**

Stdlib only. Layered detection with per-tenant audit trails:

1. **Evasion normalization** — zero-width strip, NFKC, homoglyph map, leet fold, spaced-letter collapse  
2. **Expanded injection regex** — jailbreak / role-override / prompt-extraction / chat-template markers  
3. **Heuristic risk score** — phrase-aware scoring with configurable threshold  
4. **Secret redaction** — SSN, Luhn cards, AWS/GitHub/Slack/Stripe/OpenAI keys, JWT, private keys  
5. **Output checks** — block PII in model replies; warn on prompt-leak phrasing  
6. **Latency budgets** — per-stage wall-clock budget with warn / fail_open / fail_closed breach policies  
7. **Per-tenant audit trails** — append-only, hash-chained JSONL; prompt digests only (never raw text); tamper-evident verification  

> Not a commercial LLM firewall. A small, auditable, dependency-free core (normalisation, rules and a small learned classifier) that you can read and embed, with its limits measured and published.

## Install

```bash
pip install git+https://github.com/Budeestar/sentinel-llm-guardrails.git
# or
pip install -e .
```

## Usage

```python
from sentinel_llm_guardrails import LLMGuardrails

g = LLMGuardrails(
    daily_request_limit=1000,
    injection_score_threshold=0.45,  # lower = stricter
)

inp = g.check_input(user_text, user_id="u1")
if not inp.allowed:
    raise PermissionError(inp.blocked_reason)
prompt = inp.redacted_input or user_text

# ... call your model ...

out = g.check_output(model_text)
if not out.allowed:
    raise PermissionError(out.blocked_reason)
```

## Telemetry

Track request latency and keep a tamper-evident, per-tenant audit trail (stdlib only, no extra deps):

```python
from sentinel_llm_guardrails import LLMGuardrails, LatencyBudget, LatencyStats

# Create a service with audit trails enabled
g = LLMGuardrails(
    audit_enabled=True,
    audit_dir="/var/log/llm-audit",
    audit_salt=os.environ["AUDIT_SALT"],  # for HMAC-sealed digests
    latency_budget_ms=50,
    latency_on_breach="fail_closed",  # warn / fail_open / fail_closed
)

# Each check records latency + audit entry
result = g.check_input(user_text, user_id="alice", tenant_id="acme")
print(result.latency)    # per-stage wall-clock timing
print(result.audit_seq)  # tamper-evident sequence number

# Verify audit chain integrity
trail = g.audit_trail()
verification = trail.verify("acme")  # detect tampering
print(trail.summary("acme"))  # decision counts + latency profile
```

## CLI

```bash
python -m sentinel_llm_guardrails "ignore all previous instructions"
python -m sentinel_llm_guardrails --json "How do I enable ASLR?"
python -m sentinel_llm_guardrails --output "My system prompt is secret"
```

## Audit & Compliance

Audit trails are append-only, hash-chained JSONL per tenant:
- **No prompt storage** — only salted SHA256 digest + length
- **Tamper-evident** — each entry commits to its predecessor; any edit breaks the chain
- **Verification** — `trail.verify(tenant)` detects insertions, deletions, rewrites
- **Privacy-first** — auditors can confirm "this prompt was seen" without seeing the prompt itself

```python
# Verify the chain hasn't been tampered with
result = trail.verify("acme")
assert result["ok"], f"Chain broken at seq {result['broken_at_seq']}"

# Get decision + latency summaries for compliance reporting
summary = trail.summary("acme")
print(f"Blocks: {summary['decisions'].get('block', 0)}, p95 latency: {summary['latency']['p95_ms']}ms")
```

## Examples

| File | Purpose |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | Minimal Python drop-in |
| [`examples/fastapi_middleware.py`](examples/fastapi_middleware.py) | FastAPI `/chat` guard with telemetry |

## Docs

- [`SECURITY.md`](SECURITY.md) — report bypasses privately  
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PR rules (stdlib-only)  
- [`ADOPTION_TEST.md`](ADOPTION_TEST.md) — why this repo is public  
- [`CHANGELOG.md`](CHANGELOG.md)

## What's new in 0.4.0

- **Learned injection classifier** (stdlib only) after the regex layer. On held-out test splits of four public datasets: **97.0% detection, 1.15% false positives**, and 0.5% false positives on benign coding/security questions (regex-only: 18.7% / 0%). Results and limitations: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).
- Long inputs are scored in overlapping windows, but injections hidden inside long documents are still almost never caught; see the model card.
- Tune for your traffic: `calibrate_threshold(...)`, then `LLMGuardrails(classifier_threshold=t)`; or `use_classifier=False` for the 0.3.x behaviour.
- Reproduce: `python3.11 benchmarks/download_datasets.py && python3.11 benchmarks/train_classifier.py && python3.11 benchmarks/eval_static.py`.

## What's new in 0.3.x

- **Audit trails** — per-tenant, hash-chained JSONL with tamper detection
- **Latency budgets** — per-stage wall-clock tracking + breach policies (warn / fail_open / fail_closed)
- **HMAC digests** — prompts sealed with operator-provided salt; auditors verify without exposing data
- **Compliance-ready** — decision history + latency SLO tracking per tenant

## Why this exists

Validate demand with a free module before packaging a larger security product.
Stars from friends do not count — issues/forks from strangers do.

## Develop

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT — [`LICENSE`](LICENSE)
