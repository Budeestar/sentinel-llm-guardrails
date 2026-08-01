"""sentinel-llm-guardrails — hardened OWASP LLM Top 10 checks (MIT, stdlib only)."""

from .guardrails import (
    DEFAULT_INJECTION_SCORE_THRESHOLD,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_CHARS,
    GuardrailResult,
    HeuristicInjectionScorer,
    LLMGuardrails,
    LLMGuardrailsService,
    _credit_card_match,
    _luhn_ok,
    normalize_for_detection,
)

__version__ = "0.2.0"
__all__ = [
    "GuardrailResult",
    "HeuristicInjectionScorer",
    "LLMGuardrails",
    "LLMGuardrailsService",
    "MAX_INPUT_CHARS",
    "MAX_OUTPUT_CHARS",
    "DEFAULT_INJECTION_SCORE_THRESHOLD",
    "normalize_for_detection",
    "_luhn_ok",
    "_credit_card_match",
    "__version__",
]
