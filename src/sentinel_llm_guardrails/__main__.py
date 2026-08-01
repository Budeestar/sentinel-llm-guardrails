"""CLI: python -m sentinel_llm_guardrails "user text here" """

from __future__ import annotations

import argparse
import json
import sys

from .guardrails import LLMGuardrails


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentinel-llm-guardrails",
        description="Check LLM input (or --output) against OWASP-style guardrails",
    )
    parser.add_argument("text", nargs="?", help="Text to check (or stdin)")
    parser.add_argument("--output", action="store_true", help="Validate as model output")
    parser.add_argument("--user", default="cli", help="User id for rate limits")
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    args = parser.parse_args(argv)

    text = args.text
    if text is None:
        text = sys.stdin.read()
    if not text:
        parser.error("no text provided")

    g = LLMGuardrails()
    result = g.check_output(text) if args.output else g.check_input(text, user_id=args.user)
    payload = {
        "allowed": result.allowed,
        "blocked_reason": result.blocked_reason,
        "warnings": result.warnings,
        "redacted_input": result.redacted_input,
        "checks_passed": result.checks_passed,
        "checks_failed": result.checks_failed,
        "risk_score": result.risk_score,
        "risk_reasons": result.risk_reasons,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        status = "ALLOW" if result.allowed else "BLOCK"
        print(f"{status}: {result.blocked_reason or 'ok'}")
        if result.warnings:
            for w in result.warnings:
                print(f"  warn: {w}")
        if result.redacted_input:
            print(f"  redacted: {result.redacted_input}")
    return 0 if result.allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
