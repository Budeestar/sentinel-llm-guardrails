"""
LLM guardrail telemetry — latency budgets + per-tenant audit trails.

The independent viability assessment (1 Aug 2026) noted that competing LLM
security offerings had moved past regex matching to "classifier-based
detection, latency budgets, and per-tenant audit trails". Detection scoring
lives in ``llm_guardrails``; this module supplies the other two:

* :class:`LatencyBudget` — per-stage wall-clock budget with an explicit
  breach policy (``warn`` / ``fail_open`` / ``fail_closed``). An inline
  guardrail sits on the request path, so an unbounded check is an
  availability defect, not just a slow one.
* :class:`LatencyStats` — process-wide p50/p95/p99 per stage over a bounded
  window, so a budget can be set from observed data rather than guesswork.
* :class:`TenantAuditTrail` — append-only, hash-chained JSONL per tenant.
  Prompts are never stored; only a salted digest, length, and the decision.
  This is what an auditor needs (tamper-evident decision history) without
  turning the audit log into a second copy of the customer's prompt data.

Stdlib only, thread-safe, and safe to import when no tenant is configured.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_TOTAL_BUDGET_MS = float(os.environ.get("SENTINEL_LLM_LATENCY_BUDGET_MS", "50"))
DEFAULT_BREACH_POLICY = os.environ.get("SENTINEL_LLM_LATENCY_ON_BREACH", "warn")
AUDIT_ENABLED = os.environ.get("SENTINEL_LLM_AUDIT_ENABLED", "1") not in ("0", "false", "False")

BREACH_POLICIES = ("warn", "fail_open", "fail_closed")

# Window of samples retained per stage for percentile estimation.
_STATS_WINDOW = 2048

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _evidence_root() -> Path:
    return Path(os.environ.get("SENTINEL_EVIDENCE_DIR", "sentinel_evidence"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_tenant_id(tenant_id: str | None) -> str:
    """Normalise a tenant id to a value safe to use as a filename.

    Rejecting rather than sanitising would break callers mid-request, so an
    unusable id is replaced by a stable digest-derived label.
    """
    candidate = (tenant_id or "default").strip()
    if _TENANT_RE.match(candidate):
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8", "replace")).hexdigest()[:16]
    return f"tenant-{digest}"


# ── Latency budget ───────────────────────────────────────────────────────


@dataclass
class StageTiming:
    stage: str
    elapsed_ms: float
    budget_ms: float | None = None
    breached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "budget_ms": self.budget_ms,
            "breached": self.breached,
        }


class LatencyStats:
    """Bounded-window latency percentiles per stage (process-wide)."""

    def __init__(self, window: int = _STATS_WINDOW) -> None:
        self._window = window
        self._samples: dict[str, deque[float]] = {}
        self._breaches: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, stage: str, elapsed_ms: float, breached: bool = False) -> None:
        with self._lock:
            bucket = self._samples.get(stage)
            if bucket is None:
                bucket = deque(maxlen=self._window)
                self._samples[stage] = bucket
            bucket.append(float(elapsed_ms))
            if breached:
                self._breaches[stage] = self._breaches.get(stage, 0) + 1

    @staticmethod
    def _percentile(sorted_values: list[float], pct: float) -> float:
        if not sorted_values:
            return 0.0
        if len(sorted_values) == 1:
            return round(sorted_values[0], 3)
        # Nearest-rank on a 0-indexed sorted list.
        rank = pct / 100.0 * (len(sorted_values) - 1)
        low = int(rank)
        high = min(low + 1, len(sorted_values) - 1)
        frac = rank - low
        return round(sorted_values[low] * (1 - frac) + sorted_values[high] * frac, 3)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {stage: sorted(vals) for stage, vals in self._samples.items()}
            breaches = dict(self._breaches)
        out: dict[str, Any] = {}
        for stage, vals in stages.items():
            out[stage] = {
                "count": len(vals),
                "min_ms": round(vals[0], 3) if vals else 0.0,
                "p50_ms": self._percentile(vals, 50),
                "p95_ms": self._percentile(vals, 95),
                "p99_ms": self._percentile(vals, 99),
                "max_ms": round(vals[-1], 3) if vals else 0.0,
                "breaches": breaches.get(stage, 0),
            }
        return out

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._breaches.clear()


#: Shared stats collector; exposed so the API/CLI can report percentiles.
LATENCY_STATS = LatencyStats()


@dataclass
class LatencyBudget:
    """Track wall-clock cost of guardrail stages against a budget.

    ``on_breach`` decides what the caller should do when the total budget is
    exceeded:

    * ``warn`` — record it, allow the decision to stand (default).
    * ``fail_open`` — allow the request; the guardrail is treated as advisory.
    * ``fail_closed`` — block the request; correctness over availability.
    """

    total_budget_ms: float = DEFAULT_TOTAL_BUDGET_MS
    stage_budgets_ms: dict[str, float] = field(default_factory=dict)
    on_breach: str = DEFAULT_BREACH_POLICY
    stats: LatencyStats | None = None
    timings: list[StageTiming] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.on_breach not in BREACH_POLICIES:
            self.on_breach = "warn"
        if self.stats is None:
            self.stats = LATENCY_STATS

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            budget = self.stage_budgets_ms.get(name)
            breached = budget is not None and elapsed_ms > budget
            timing = StageTiming(name, elapsed_ms, budget, breached)
            self.timings.append(timing)
            if self.stats is not None:
                self.stats.record(name, elapsed_ms, breached)

    @property
    def elapsed_ms(self) -> float:
        return sum(t.elapsed_ms for t in self.timings)

    @property
    def exceeded(self) -> bool:
        return self.elapsed_ms > self.total_budget_ms

    @property
    def breached_stages(self) -> list[str]:
        return [t.stage for t in self.timings if t.breached]

    def should_block(self) -> bool:
        """True when policy says a budget breach must fail the request."""
        return self.exceeded and self.on_breach == "fail_closed"

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.exceeded:
            out.append(
                f"Latency budget exceeded: {self.elapsed_ms:.1f}ms > "
                f"{self.total_budget_ms:.1f}ms (policy={self.on_breach})"
            )
        for stage in self.breached_stages:
            out.append(f"Stage '{stage}' exceeded its budget")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_ms": round(self.elapsed_ms, 3),
            "total_budget_ms": self.total_budget_ms,
            "exceeded": self.exceeded,
            "on_breach": self.on_breach,
            "breached_stages": self.breached_stages,
            "stages": [t.to_dict() for t in self.timings],
        }


# ── Per-tenant audit trail ───────────────────────────────────────────────


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class TenantAuditTrail:
    """Append-only, hash-chained decision log, partitioned per tenant.

    Each entry commits to its predecessor, so removing or editing a past
    decision invalidates every hash after it and :meth:`verify` reports the
    exact sequence number where the chain broke.

    Prompt text is never written. ``content_sha256`` is an HMAC keyed by
    ``SENTINEL_LLM_AUDIT_SALT`` when set, which lets an operator confirm
    "this exact prompt was seen" without the log itself disclosing prompts.
    """

    GENESIS = "0" * 64

    def __init__(self, directory: str | Path | None = None, salt: str | None = None) -> None:
        base = _evidence_root() / "llm_audit"
        self.dir = Path(directory) if directory else base
        self.dir.mkdir(parents=True, exist_ok=True)
        self._salt = (salt if salt is not None else os.environ.get("SENTINEL_LLM_AUDIT_SALT", "")).encode()
        self._lock = threading.Lock()
        self._tips: dict[str, tuple[int, str]] = {}

    def _path(self, tenant: str) -> Path:
        return self.dir / f"{tenant}.jsonl"

    def digest(self, text: str) -> str:
        raw = text.encode("utf-8", "replace")
        if self._salt:
            return hmac.new(self._salt, raw, hashlib.sha256).hexdigest()
        return hashlib.sha256(raw).hexdigest()

    def _tip(self, tenant: str) -> tuple[int, str]:
        """Return (last_seq, last_hash), reading from disk once per tenant."""
        cached = self._tips.get(tenant)
        if cached is not None:
            return cached
        seq, prev = 0, self.GENESIS
        path = self._path(tenant)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seq = int(entry.get("seq", seq))
                    prev = str(entry.get("hash", prev))
        self._tips[tenant] = (seq, prev)
        return seq, prev

    def record(
        self,
        tenant_id: str,
        *,
        decision: str,
        stage: str,
        user_id: str = "default",
        content: str | None = None,
        reason: str | None = None,
        injection_score: float | None = None,
        engines: list[str] | None = None,
        latency: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one decision and return the written entry."""
        tenant = safe_tenant_id(tenant_id)
        payload: dict[str, Any] = {
            "ts": _now_iso(),
            "tenant_id": tenant,
            "user_id": user_id,
            "stage": stage,
            "decision": decision,
            "reason": reason,
            "injection_score": round(float(injection_score), 4) if injection_score is not None else None,
            "engines": list(engines or []),
            "latency": latency or {},
        }
        if content is not None:
            payload["content_sha256"] = self.digest(content)
            payload["content_chars"] = len(content)
            payload["content_hmac"] = bool(self._salt)
        if extra:
            payload["extra"] = extra

        with self._lock:
            seq, prev_hash = self._tip(tenant)
            seq += 1
            payload["seq"] = seq
            payload["prev_hash"] = prev_hash
            payload["hash"] = hashlib.sha256(
                (prev_hash + _canonical(payload)).encode("utf-8")
            ).hexdigest()
            with self._path(tenant).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
            self._tips[tenant] = (seq, payload["hash"])
        return payload

    def tenants(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.jsonl"))

    def read(self, tenant_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        path = self._path(safe_tenant_id(tenant_id))
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if limit is not None and limit > 0:
            return entries[-limit:]
        return entries

    def verify(self, tenant_id: str) -> dict[str, Any]:
        """Recompute the chain and report the first inconsistency, if any."""
        tenant = safe_tenant_id(tenant_id)
        entries = self.read(tenant)
        prev_hash = self.GENESIS
        for index, entry in enumerate(entries, start=1):
            recorded_hash = entry.get("hash")
            body = {k: v for k, v in entry.items() if k != "hash"}
            expected = hashlib.sha256((prev_hash + _canonical(body)).encode("utf-8")).hexdigest()
            if entry.get("prev_hash") != prev_hash:
                return {
                    "ok": False, "tenant_id": tenant, "entries": len(entries),
                    "broken_at_seq": entry.get("seq", index), "error": "prev_hash mismatch",
                }
            if recorded_hash != expected:
                return {
                    "ok": False, "tenant_id": tenant, "entries": len(entries),
                    "broken_at_seq": entry.get("seq", index), "error": "hash mismatch",
                }
            if int(entry.get("seq", -1)) != index:
                return {
                    "ok": False, "tenant_id": tenant, "entries": len(entries),
                    "broken_at_seq": entry.get("seq", index), "error": "sequence gap",
                }
            prev_hash = str(recorded_hash)
        return {"ok": True, "tenant_id": tenant, "entries": len(entries), "tip_hash": prev_hash}

    def verify_all(self) -> dict[str, Any]:
        results = {tenant: self.verify(tenant) for tenant in self.tenants()}
        return {
            "ok": all(r.get("ok") for r in results.values()),
            "tenants": len(results),
            "results": results,
        }

    def summary(self, tenant_id: str) -> dict[str, Any]:
        """Decision counts and latency profile for one tenant."""
        entries = self.read(tenant_id)
        decisions: dict[str, int] = {}
        stages: dict[str, int] = {}
        latencies: list[float] = []
        breaches = 0
        for entry in entries:
            decisions[str(entry.get("decision"))] = decisions.get(str(entry.get("decision")), 0) + 1
            stages[str(entry.get("stage"))] = stages.get(str(entry.get("stage")), 0) + 1
            latency = entry.get("latency") or {}
            if isinstance(latency, dict):
                if isinstance(latency.get("elapsed_ms"), (int, float)):
                    latencies.append(float(latency["elapsed_ms"]))
                if latency.get("exceeded"):
                    breaches += 1
        ordered = sorted(latencies)
        return {
            "tenant_id": safe_tenant_id(tenant_id),
            "entries": len(entries),
            "decisions": decisions,
            "stages": stages,
            "latency": {
                "count": len(ordered),
                "p50_ms": LatencyStats._percentile(ordered, 50),
                "p95_ms": LatencyStats._percentile(ordered, 95),
                "p99_ms": LatencyStats._percentile(ordered, 99),
                "budget_breaches": breaches,
            },
            "first_ts": entries[0].get("ts") if entries else None,
            "last_ts": entries[-1].get("ts") if entries else None,
        }


_TRAIL: TenantAuditTrail | None = None
_TRAIL_LOCK = threading.Lock()


def get_audit_trail(directory: str | Path | None = None) -> TenantAuditTrail:
    """Process-wide audit trail, so the hash chain has a single writer."""
    global _TRAIL
    if directory is not None:
        return TenantAuditTrail(directory)
    with _TRAIL_LOCK:
        if _TRAIL is None:
            _TRAIL = TenantAuditTrail()
        return _TRAIL
