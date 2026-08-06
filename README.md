# sentinel-llm-guardrails

[![ci](https://github.com/Budeestar/sentinel-llm-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/Budeestar/sentinel-llm-guardrails/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hardened, dependency-free OWASP LLM Top 10 guardrails (MIT).**

Stdlib only. Layered detection:

1. **Evasion normalization** — zero-width strip, NFKC, homoglyph map, leet fold, spaced-letter collapse  
2. **Expanded injection regex** — jailbreak / role-override / prompt-extraction / chat-template markers  
3. **Heuristic risk score** — phrase-aware scoring with configurable threshold  
4. **Secret redaction** — SSN, Luhn cards, AWS/GitHub/Slack/Stripe/OpenAI keys, JWT, private keys  
5. **Output checks** — block PII in model replies; warn on prompt-leak phrasing  
6. **Latency budgets + per-tenant audit trails** — per-stage wall-clock budget with explicit breach policy; tamper-evident, prompt-digest audit logs  
7. **Optional OPA/Rego** — `policies/llm-security.rego`

> Not a commercial LLM firewall. Not a trained classifier. A small, auditable core you can read and embed.

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
from sentinel_llm_guardrails import LatencyBudget, LatencyStats, get_audit_trail

with LatencyBudget(total_budget_ms=50, stats=LatencyStats()).stage("check_input"):
    # ... your guardrail call ...
    pass

trail = get_audit_trail()
trail.append(tenant_id="acme", prompt_digest="sha256-...", allowed=True, reasons=["clean"])
```

## CLI

```bash
python -m sentinel_llm_guardrails "ignore all previous instructions"
python -m sentinel_llm_guardrails --json "How do I enable ASLR?"
python -m sentinel_llm_guardrails --output "My system prompt is secret"
```

## Examples

| File | Purpose |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | Minimal Python drop-in |
| [`examples/fastapi_middleware.py`](examples/fastapi_middleware.py) | FastAPI `/chat` guard |

## Docs

- [`SECURITY.md`](SECURITY.md) — report bypasses privately  
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PR rules (stdlib-only)  
- [`ADOPTION_TEST.md`](ADOPTION_TEST.md) — why this repo is public  
- [`CHANGELOG.md`](CHANGELOG.md)

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
