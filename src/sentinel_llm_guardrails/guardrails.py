"""OWASP LLM Top 10 guardrails — stdlib only, hardened heuristics.

MIT. Not a trained ML model — layered regex + phrase scoring + evasion
normalization so common jailbreaks and secret leaks are harder to slip past.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .telemetry import (
    LATENCY_STATS,
    LatencyBudget,
    TenantAuditTrail,
)

__version__ = "0.4.0"


@dataclass
class GuardrailResult:
    allowed: bool
    blocked_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    redacted_input: str | None = None
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    risk_reasons: list[str] = field(default_factory=list)
    tenant_id: str = "default"
    latency: dict[str, Any] = field(default_factory=dict)
    audit_seq: int | None = None


# ── Evasion normalization ─────────────────────────────────────────────

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",  # cyrillic
    "і": "i", "ј": "j", "ѕ": "s",
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "Ｆ": "F", "Ｇ": "G",
    "Ｈ": "H", "Ｉ": "I", "Ｊ": "J", "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N",
    "Ｏ": "O", "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T", "Ｕ": "U",
    "Ｖ": "V", "Ｗ": "W", "Ｘ": "X", "Ｙ": "Y", "Ｚ": "Z",
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e", "ｆ": "f", "ｇ": "g",
    "ｈ": "h", "ｉ": "i", "ｊ": "j", "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n",
    "ｏ": "o", "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t", "ｕ": "u",
    "ｖ": "v", "ｗ": "w", "ｘ": "x", "ｙ": "y", "ｚ": "z",
    "0": "0", "1": "1", "３": "3", "４": "4", "５": "5",
})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def normalize_for_detection(text: str) -> str:
    """Strip zero-width chars, NFKC, map common homoglyphs; collapse spelled-out words."""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_ZERO_WIDTH)
    t = t.translate(_HOMOGLYPHS)
    # Collapse only spelled-out words: "i g n o r e" / "i.g.n.o.r.e" — not normal spaces.
    def _collapse_spelled(match: re.Match[str]) -> str:
        return re.sub(r"[\s._\-]+", "", match.group(0))

    t = re.sub(
        r"\b(?:[A-Za-z](?:[\s._\-]{1,2})){2,}[A-Za-z]\b",
        _collapse_spelled,
        t,
    )
    return t


def _leet_fold(text: str) -> str:
    return text.lower().translate(_LEET)


# ── Injection patterns (raw + applied to normalized text) ─────────────

_INJECTION_RAW = [
    r"ignore\s+(all\s+)?previous\s+(instructions|prompts|rules|context)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions|rules|prompt)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"from\s+now\s+on\s+you\s+(are|will|must)",
    r"system\s*:\s*you\s+(are|must|should|will)",
    r"\[system\]",
    r"\[/?inst\]",
    r"<<\s*sys\s*>>",
    r"<\|?(system|im_start|endoftext)\|?>",
    r"repeat\s+(your\s+)?(system|hidden|initial)\s+prompt",
    r"show\s+(me\s+)?(your\s+)?(system|hidden|initial)\s+prompt",
    r"print\s+(your\s+)?(system|developer)\s+(prompt|message|instructions)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"bypass\s+(all\s+)?(safety|filters|guardrails|restrictions)",
    r"jailbreak",
    r"\bDAN\s+mode\b",
    r"do\s+anything\s+now",
    r"developer\s+mode\s+(enabled|on)",
    r"opposite\s+of\s+your\s+(usual|normal)\s+(rules|instructions)",
    r"respond\s+without\s+(any\s+)?(filters|restrictions|ethics)",
    r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(unrestricted|uncensored)",
    r"roleplay\s+as\s+(an?\s+)?(evil|unrestricted|jailbroken)",
    r"encode\s+(the\s+)?(system\s+)?prompt\s+in\s+base64",
    r"base64[-_\s]?decode\s+and\s+(execute|follow|run)",
    r"sudo\s+mode",
    r"admin\s+override",
    r"disable\s+(your\s+)?(safety|content)\s+(filters?|policy)",
]

INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_RAW]

# ── Heuristic scorer ──────────────────────────────────────────────────

class HeuristicInjectionScorer:
    """Phrase-aware risk score (0–1). Complements regex; not a trained model."""

    ROLE_PHRASES = (
        "ignore previous", "disregard previous", "forget your", "bypass safety",
        "you are now", "pretend you", "roleplay as", "act as", "from now on",
        "new instructions", "override your", "no restrictions", "without filter",
    )
    SYSTEM_MARKERS = (
        "[system]", "[assistant]", "[user]", "[inst]", "<<sys>>",
        "system:", "assistant:", "<|im_start|>", "<|system|>",
    )
    EXTRACT_PHRASES = (
        "system prompt", "hidden prompt", "developer message", "initial instructions",
        "reveal your", "show your prompt", "print your instructions",
    )

    def score(self, text: str) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0
        norm = normalize_for_detection(text)
        low = _leet_fold(norm)

        role_hits = sum(1 for p in self.ROLE_PHRASES if p in low)
        if role_hits:
            score += min(0.35, role_hits * 0.12)
            reasons.append(f"role_override:{role_hits}")

        sys_hits = sum(1 for m in self.SYSTEM_MARKERS if m in low)
        if sys_hits:
            score += min(0.3, sys_hits * 0.15)
            reasons.append(f"system_markers:{sys_hits}")

        ext_hits = sum(1 for p in self.EXTRACT_PHRASES if p in low)
        if ext_hits:
            score += min(0.25, ext_hits * 0.12)
            reasons.append(f"prompt_extraction:{ext_hits}")

        if any(m in text for m in ("---", ">>>", "###SYSTEM", "===", "***")):
            score += 0.08
            reasons.append("context_break")

        # Dense instruction verbs
        verbs = sum(1 for v in ("must", "shall", "obey", "comply", "execute") if v in low.split())
        if verbs >= 3:
            score += 0.1
            reasons.append(f"imperative_density:{verbs}")

        return min(1.0, score), reasons


# ── PII / secrets ─────────────────────────────────────────────────────

PII_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "Credit Card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}\d(?!\d)"),
    "AWS Access Key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "AWS Secret Key": re.compile(
        r"(?i)(?:aws[_-]?secret[_-]?(?:access[_-]?)?key|secret[_-]?access[_-]?key)"
        r"\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})[\"']?"
    ),
    "GitHub Token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}\b"),
    "Slack Token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "Stripe Key": re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b"),
    "OpenAI Key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "JWT": re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    "Private Key": re.compile(
        r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"
    ),
    "Password in URL": re.compile(r"://[^/:@\s]+:[^@\s/]+@"),
    "Email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b"),
}


def _luhn_ok(digits: str) -> bool:
    n = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(n) <= 19:
        return False
    checksum = 0
    parity = (len(n) - 2) % 2
    for i, d in enumerate(n[:-1]):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return (checksum + n[-1]) % 10 == 0


def _credit_card_match(text: str) -> bool:
    for m in PII_PATTERNS["Credit Card"].finditer(text):
        digits = re.sub(r"[^0-9]", "", m.group(0))
        if _luhn_ok(digits):
            return True
    return False


MAX_INPUT_CHARS = 32_000
MAX_OUTPUT_CHARS = 200_000
MAX_TRACKED_USERS = 4096
DEFAULT_INJECTION_SCORE_THRESHOLD = 0.45


class LLMGuardrails:
    """Thread-safe input/output guardrails for LLM applications."""

    def __init__(
        self,
        daily_request_limit: int = 1000,
        daily_token_limit: int = 500_000,
        injection_score_threshold: float = DEFAULT_INJECTION_SCORE_THRESHOLD,
        latency_budget_ms: float | None = None,
        latency_on_breach: str = "warn",
        audit_enabled: bool = True,
        audit_dir: str | Path | None = None,
        audit_salt: str | None = None,
        use_classifier: bool = True,
        classifier_threshold: float | None = None,
        classifier: Any = None,
    ) -> None:
        self._daily_request_limit = daily_request_limit
        self._daily_token_limit = daily_token_limit
        self._injection_score_threshold = injection_score_threshold
        self._scorer = HeuristicInjectionScorer()
        self._request_counts: OrderedDict[str, int] = OrderedDict()
        self._token_counts: OrderedDict[str, int] = OrderedDict()
        self._day_key = ""
        self._blocked_count = 0
        self._lock = threading.Lock()
        self._latency_budget_ms = latency_budget_ms
        self._latency_on_breach = latency_on_breach
        self._audit_enabled = audit_enabled
        self._audit_trail: TenantAuditTrail | None = None
        # Learned classifier — resolved lazily on first use so that
        # `use_classifier=False` (and regex-only deployments) never touch it.
        self._use_classifier = use_classifier
        self._classifier_threshold_override = classifier_threshold
        # `classifier` lets callers supply their own InjectionClassifier (e.g. one
        # retrained on their traffic); None means the packaged default model.
        self._classifier_model: Any = classifier
        self._classifier_resolved = classifier is not None
        if audit_enabled:
            self._audit_trail = TenantAuditTrail(directory=audit_dir, salt=audit_salt)

    @classmethod
    def service(cls, **kwargs: float | int) -> LLMGuardrails:
        return cls(**kwargs)  # type: ignore[arg-type]

    def _get_classifier(self) -> Any:
        """Return the loaded classifier, or ``None`` when unavailable/disabled.

        Loading is lazy and cached: a missing or corrupt model file degrades
        silently to the regex + heuristic stack (a single warning is emitted by
        :mod:`sentinel_llm_guardrails.classifier`).
        """
        if not self._use_classifier:
            return None
        if not self._classifier_resolved:
            from .classifier import InjectionClassifier  # local: keeps import cost lazy

            self._classifier_model = InjectionClassifier.default()
            self._classifier_resolved = True
        return self._classifier_model

    def _touch_user(self, table: OrderedDict[str, int], user_id: str) -> int:
        if user_id in table:
            table.move_to_end(user_id)
            return table[user_id]
        if len(table) >= MAX_TRACKED_USERS:
            table.popitem(last=False)
        table[user_id] = 0
        return 0

    def _injection_hit(self, text: str) -> tuple[bool, float, list[str], str | None]:
        norm = normalize_for_detection(text)
        folded = _leet_fold(norm)
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text) or pattern.search(norm) or pattern.search(folded):
                return True, 1.0, ["regex_injection"], "Prompt injection detected (OWASP LLM01)"

        # Learned classifier: runs only when the model is present. A sub-threshold
        # score deliberately falls through to the heuristic scorer below, so the
        # pre-existing behaviour is never lost.
        model = self._get_classifier()
        if model is not None:
            threshold = self._classifier_threshold_override
            if threshold is None:
                threshold = float(getattr(model, "threshold", 0.5) or 0.5)
            try:
                probability = float(model.predict_proba(text))
            except Exception:  # noqa: BLE001 - a guardrail must never crash
                probability = None
            if probability is not None and probability >= threshold:
                return (
                    True,
                    probability,
                    ["classifier"],
                    f"Prompt injection classifier p={probability:.2f} >= {threshold:.2f}",
                )

        score, reasons = self._scorer.score(text)
        if score >= self._injection_score_threshold:
            return True, score, reasons, (
                f"Prompt injection risk score {score:.2f} >= {self._injection_score_threshold}"
            )
        return False, score, reasons, None

    def _check_input_core(self, user_input: str, user_id: str = "default") -> GuardrailResult:
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        warnings: list[str] = []

        if not isinstance(user_input, str):
            return GuardrailResult(
                allowed=False, blocked_reason="user_input must be a string",
                checks_failed=["input_type"],
            )
        if len(user_input) > MAX_INPUT_CHARS:
            with self._lock:
                self._blocked_count += 1
            return GuardrailResult(
                allowed=False,
                blocked_reason=f"Input exceeds {MAX_INPUT_CHARS} characters",
                checks_failed=["input_size"],
            )

        hit, score, reasons, reason = self._injection_hit(user_input)
        if hit:
            checks_failed.append("LLM01_prompt_injection")
            with self._lock:
                self._blocked_count += 1
            return GuardrailResult(
                allowed=False, blocked_reason=reason,
                checks_failed=checks_failed, risk_score=score, risk_reasons=reasons,
            )
        checks_passed.append("LLM01_prompt_injection")

        redacted = user_input
        found_pii: list[str] = []
        for pii_type, pattern in PII_PATTERNS.items():
            if pii_type == "Credit Card" and not _credit_card_match(user_input):
                continue
            if pattern.search(user_input):
                found_pii.append(pii_type)
                redacted = pattern.sub(f"[REDACTED_{pii_type.upper().replace(' ', '_')}]", redacted)

        if found_pii:
            warnings.append(f"PII detected and redacted: {', '.join(found_pii)}")
            checks_passed.append("LLM02_pii_redacted")
        else:
            checks_passed.append("LLM02_no_pii")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        estimated_tokens = len(user_input) // 4
        with self._lock:
            if today != self._day_key:
                self._day_key = today
                self._request_counts.clear()
                self._token_counts.clear()

            count = self._touch_user(self._request_counts, user_id)
            if count >= self._daily_request_limit:
                checks_failed.append("LLM04_rate_limit")
                self._blocked_count += 1
                return GuardrailResult(
                    allowed=False,
                    blocked_reason=f"Daily request limit ({self._daily_request_limit}) exceeded",
                    checks_failed=checks_failed, checks_passed=checks_passed,
                    risk_score=score, risk_reasons=reasons,
                )
            self._request_counts[user_id] = count + 1

            tokens = self._touch_user(self._token_counts, user_id)
            if tokens + estimated_tokens > self._daily_token_limit:
                checks_failed.append("LLM04_token_limit")
                self._blocked_count += 1
                return GuardrailResult(
                    allowed=False,
                    blocked_reason=f"Daily token limit ({self._daily_token_limit}) exceeded",
                    checks_failed=checks_failed, checks_passed=checks_passed,
                    risk_score=score, risk_reasons=reasons,
                )
            self._token_counts[user_id] = tokens + estimated_tokens
        checks_passed.append("LLM04_rate_limit")

        return GuardrailResult(
            allowed=True, warnings=warnings,
            redacted_input=redacted if found_pii else None,
            checks_passed=checks_passed, checks_failed=checks_failed,
            risk_score=score, risk_reasons=reasons,
        )

    def _check_output_core(self, llm_output: str) -> GuardrailResult:
        checks_passed: list[str] = []
        warnings: list[str] = []

        if not isinstance(llm_output, str):
            return GuardrailResult(
                allowed=False, blocked_reason="llm_output must be a string",
                checks_failed=["output_type"],
            )
        if len(llm_output) > MAX_OUTPUT_CHARS:
            with self._lock:
                self._blocked_count += 1
            return GuardrailResult(
                allowed=False,
                blocked_reason=f"Output exceeds {MAX_OUTPUT_CHARS} characters",
                checks_failed=["LLM05_output_size"],
            )

        for pattern in (
            re.compile(r"my\s+system\s+prompt\s+is", re.IGNORECASE),
            re.compile(r"my\s+(initial\s+)?instructions\s+are", re.IGNORECASE),
            re.compile(r"i\s+was\s+told\s+to\s+(never|always|ignore)", re.IGNORECASE),
            re.compile(r"the\s+developer\s+message\s+(says|is)", re.IGNORECASE),
        ):
            if pattern.search(llm_output):
                warnings.append("Potential system prompt leak detected in output")

        found_pii_output: list[str] = []
        for pii_type, pattern in PII_PATTERNS.items():
            if pii_type == "Credit Card" and not _credit_card_match(llm_output):
                continue
            if pattern.search(llm_output):
                found_pii_output.append(pii_type)

        if found_pii_output:
            with self._lock:
                self._blocked_count += 1
            return GuardrailResult(
                allowed=False,
                blocked_reason=f"PII detected in LLM output: {', '.join(found_pii_output)}",
                warnings=warnings, checks_failed=["LLM05_output_pii"],
                checks_passed=checks_passed,
            )

        checks_passed.append("LLM05_output_validation")
        return GuardrailResult(allowed=True, warnings=warnings, checks_passed=checks_passed)

    def _new_budget(self) -> LatencyBudget:
        from . import telemetry as _telemetry
        total = self._latency_budget_ms if self._latency_budget_ms is not None else _telemetry.DEFAULT_TOTAL_BUDGET_MS
        return LatencyBudget(total_budget_ms=total, on_breach=self._latency_on_breach, stats=LATENCY_STATS)

    def audit_trail(self) -> TenantAuditTrail | None:
        return self._audit_trail

    def _record(
        self,
        result: GuardrailResult,
        user_id: str,
        tenant_id: str,
        stage: str,
        content: str,
    ) -> None:
        if self._audit_trail is None:
            return
        entry = self._audit_trail.record(
            tenant_id,
            decision="allow" if result.allowed else "block",
            stage=stage,
            user_id=user_id,
            content=content if isinstance(content, str) else str(content),
            reason=result.blocked_reason,
            injection_score=result.risk_score,
            latency=result.latency,
            extra={
                "checks_passed": result.checks_passed,
                "checks_failed": result.checks_failed,
            },
        )
        result.audit_seq = entry["seq"]

    def check_input(
        self,
        user_input: str,
        user_id: str = "default",
        tenant_id: str = "default",
    ) -> GuardrailResult:
        budget = self._new_budget()
        with budget.stage("input"):
            result = self._check_input_core(user_input, user_id)
        result.tenant_id = tenant_id
        result.latency = budget.to_dict()
        if budget.should_block() and result.allowed:
            result.allowed = False
            result.blocked_reason = f"Latency budget exceeded: {budget.elapsed_ms:.1f}ms"
            result.checks_failed.append("latency_budget")
            result.warnings.extend(budget.warnings())
        self._record(result, user_id, tenant_id, "input", user_input)
        return result

    def check_output(
        self,
        llm_output: str,
        tenant_id: str = "default",
    ) -> GuardrailResult:
        budget = self._new_budget()
        with budget.stage("output"):
            result = self._check_output_core(llm_output)
        result.tenant_id = tenant_id
        result.latency = budget.to_dict()
        if budget.should_block() and result.allowed:
            result.allowed = False
            result.blocked_reason = f"Latency budget exceeded: {budget.elapsed_ms:.1f}ms"
            result.checks_failed.append("latency_budget")
            result.warnings.extend(budget.warnings())
        self._record(result, "default", tenant_id, "output", llm_output)
        return result

    def check_tool(
        self,
        tool: str,
        user_role: str = "user",
        tenant_id: str = "default",
    ) -> GuardrailResult:
        budget = self._new_budget()
        with budget.stage("tool"):
            # Public default is permissive: customers can override in subclass/ACL.
            allowed = True
            checks_passed = ["LLM03_tool_authorised"]
            checks_failed: list[str] = []
            blocked_reason: str | None = None
        result = GuardrailResult(
            allowed=allowed,
            blocked_reason=blocked_reason,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )
        result.tenant_id = tenant_id
        result.latency = budget.to_dict()
        if budget.should_block():
            result.allowed = False
            result.blocked_reason = f"Latency budget exceeded: {budget.elapsed_ms:.1f}ms"
            result.checks_failed.append("latency_budget")
            result.warnings.extend(budget.warnings())
        self._record(result, user_role, tenant_id, "tool", tool)
        return result

    def get_stats(self) -> dict:
        with self._lock:
            stats = {
                "total_blocked": self._blocked_count,
                "daily_requests": dict(self._request_counts),
                "daily_tokens": dict(self._token_counts),
                "tracked_users": len(self._request_counts),
                "version": __version__,
                "injection_score_threshold": self._injection_score_threshold,
                "audit_enabled": self._audit_enabled,
                "audit_tenants": self._audit_trail.tenants() if self._audit_trail else [],
                "latency_on_breach": self._latency_on_breach,
                "latency": LATENCY_STATS.snapshot(),
            }
        return stats


LLMGuardrailsService = LLMGuardrails
