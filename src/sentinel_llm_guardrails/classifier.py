"""Learned prompt-injection classifier — stdlib only.

The library stays dependency-free: text is hashed into a fixed number of buckets
with :func:`zlib.crc32` and scored by a tiny logistic-regression model whose
(pruned) weights live in ``models/injection_lr.json`` next to this module.
Training happens offline in ``benchmarks/train_classifier.py``.

Nothing here imports third-party code and the module never touches the network.
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import warnings
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BUCKETS",
    "DEFAULT_THRESHOLD",
    "MAX_FEATURIZE_CHARS",
    "MAX_WINDOWS",
    "WINDOW_STRIDE",
    "InjectionClassifier",
    "calibrate_threshold",
    "default_model_path",
    "featurize",
    "train",
]

MODEL_VERSION = 1
BUCKETS = 2 ** 20          # 1 048 576 hashed feature buckets
BUCKET_MASK = BUCKETS - 1
MAX_FEATURIZE_CHARS = 4000  # inputs are truncated before featurisation (speed)
WINDOW_STRIDE = MAX_FEATURIZE_CHARS // 2  # stride between long-input scoring windows
MAX_WINDOWS = 16           # hard cap on windows per input (32000-char input limit)
DEFAULT_THRESHOLD = 0.5
PRUNE_ABS = 1e-4           # weights smaller than this are dropped from the model
MAX_KEPT_WEIGHTS = 60_000  # hard cap on the number of stored weights
MODEL_FILENAME = "injection_lr.json"

_TOKEN_RE = re.compile(r"\w+")

_MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_FILENAME

# Errors that mean "the model simply is not usable" — the caller must never crash.
_LOAD_ERRORS = (OSError, ValueError, TypeError, KeyError)

_DEFAULT_LOCK = threading.Lock()
_DEFAULT_CACHE: dict[str, Any] = {"loaded": False, "model": None}
_WARNED = False


def default_model_path() -> Path:
    """Path of the packaged model file (it may or may not exist)."""
    return _MODEL_PATH


def _sigmoid(z: float) -> float:
    if z <= -60.0:
        return 0.0
    if z >= 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _warn_once(path: Path, exc: BaseException) -> None:
    """Emit the "classifier unavailable" warning at most once per process."""
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    warnings.warn(
        f"sentinel-llm-guardrails: prompt-injection classifier unavailable at {path} "
        f"({type(exc).__name__}: {exc}); falling back to regex + heuristics.",
        RuntimeWarning,
        stacklevel=3,
    )


# ── Featurisation ─────────────────────────────────────────────────────


def featurize(text: str) -> dict[int, float]:
    """Hash ``text`` into a sparse, L2-normalised feature vector.

    Features (binary presence, each hashed with ``zlib.crc32`` into one of
    :data:`BUCKETS` buckets):

    * word unigrams and bigrams over ``re.findall(r"\\w+", text)``;
    * character 3- to 5-grams of the normalised text, with whitespace collapsed.

    The text is first run through the library's evasion normalisation
    (:func:`~sentinel_llm_guardrails.guardrails.normalize_for_detection` plus
    ``_leet_fold``) and truncated to the first :data:`MAX_FEATURIZE_CHARS`
    characters.  Those two helpers are imported lazily because ``guardrails``
    imports this module for the classifier path — a module-level import would be
    circular.
    """
    if not isinstance(text, str):
        text = str(text)

    from .guardrails import _leet_fold, normalize_for_detection  # local: avoid cycle

    folded = _leet_fold(normalize_for_detection(text[:MAX_FEATURIZE_CHARS]))
    folded = " ".join(folded.split())  # collapse whitespace

    buckets: dict[int, float] = {}

    def mark(prefix: str, value: str) -> None:
        key = prefix + value
        buckets[zlib.crc32(key.encode("utf-8")) & BUCKET_MASK] = 1.0

    tokens = _TOKEN_RE.findall(folded)
    for token in tokens:
        mark("w:", token)
    for first, second in zip(tokens, tokens[1:]):
        mark("b:", first + "\x1f" + second)

    length = len(folded)
    if length >= 3:
        for n in (3, 4, 5):
            for index in range(length - n + 1):
                mark("c:", folded[index:index + n])

    squared = 0.0
    for value in buckets.values():
        squared += value * value
    if squared > 0.0:
        norm = math.sqrt(squared)
        for key in buckets:
            buckets[key] /= norm
    return buckets


# ── Model ─────────────────────────────────────────────────────────────


def _window_starts(length: int) -> list[int]:
    """Offsets of the windows used to score a text of ``length`` characters.

    Windows are :data:`MAX_FEATURIZE_CHARS` wide with a stride of
    :data:`WINDOW_STRIDE` and always include a final window ending at the last
    character, so any span up to one stride long lies wholly inside some window.
    At most :data:`MAX_WINDOWS` starts are returned; when the text would need
    more, the starts are spread evenly between the first and the last window.
    """
    last = max(0, length - MAX_FEATURIZE_CHARS)
    starts = list(range(0, last + 1, WINDOW_STRIDE))
    if not starts or starts[-1] != last:
        starts.append(last)
    if len(starts) > MAX_WINDOWS:
        step = last / (MAX_WINDOWS - 1)
        starts = sorted({int(round(index * step)) for index in range(MAX_WINDOWS)})
        starts[0] = 0
        starts[-1] = last
    return starts


class InjectionClassifier:
    """Binary logistic regression over hashed text features (stdlib only)."""

    def __init__(
        self,
        weights: Mapping[int, float] | None = None,
        bias: float = 0.0,
        threshold: float = DEFAULT_THRESHOLD,
        buckets: int = BUCKETS,
        trained_on: Sequence[str] | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        self.weights: dict[int, float] = {
            int(bucket): float(weight) for bucket, weight in (weights or {}).items()
        }
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.buckets = int(buckets)
        self.trained_on = list(trained_on or [])
        self.metrics = dict(metrics or {})

    # -- scoring -------------------------------------------------------

    def predict_proba(self, text: str) -> float:
        """Injection probability (0–1) for a raw string.

        A short input is featurised in one shot.  An input longer than
        :data:`MAX_FEATURIZE_CHARS` is split into overlapping windows of
        :data:`MAX_FEATURIZE_CHARS` characters (stride :data:`WINDOW_STRIDE`, at
        most :data:`MAX_WINDOWS` of them, the last one anchored at the final
        character) and the **maximum** window probability is returned, so an
        injection hidden after thousands of benign characters is still scored.
        """
        if not isinstance(text, str):
            text = str(text)
        if len(text) <= MAX_FEATURIZE_CHARS:
            return self.predict_proba_features(featurize(text))

        best = 0.0
        for start in _window_starts(len(text)):
            window = text[start:start + MAX_FEATURIZE_CHARS]
            probability = self.predict_proba_features(featurize(window))
            if probability > best:
                best = probability
        return best

    def predict_proba_features(self, features: Mapping[int, float]) -> float:
        """Injection probability for an already featurised text (see :func:`featurize`)."""
        weights = self.weights
        z = self.bias
        for bucket, value in features.items():
            weight = weights.get(bucket)
            if weight is not None:
                z += weight * value
        return _sigmoid(z)

    def is_injection(self, text: str) -> bool:
        """Whether ``text`` scores at or above this model's threshold."""
        return self.predict_proba(text) >= self.threshold

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready payload; only the surviving (non-zero) weights are stored."""
        return {
            "version": MODEL_VERSION,
            "buckets": self.buckets,
            "bias": self.bias,
            "weights": {
                str(bucket): round(weight, 6) for bucket, weight in self.weights.items()
            },
            "threshold": self.threshold,
            "trained_on": list(self.trained_on),
            "metrics": dict(self.metrics),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InjectionClassifier":
        return cls(
            weights={
                int(bucket): float(weight)
                for bucket, weight in (payload.get("weights") or {}).items()
            },
            bias=float(payload.get("bias", 0.0)),
            threshold=float(payload.get("threshold", DEFAULT_THRESHOLD)),
            buckets=int(payload.get("buckets", BUCKETS)),
            trained_on=payload.get("trained_on") or [],
            metrics=payload.get("metrics") or {},
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "InjectionClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: model payload must be a JSON object")
        return cls.from_dict(payload)

    @classmethod
    def default(cls, path: str | Path | None = None) -> "InjectionClassifier | None":
        """Load the packaged model (lazy, cached, thread-safe).

        Returns ``None`` — and warns once — when the file is missing or unusable,
        so callers can silently fall back to the regex + heuristic stack.
        """
        if path is not None:
            try:
                return cls.from_file(path)
            except _LOAD_ERRORS as exc:  # pragma: no cover - path is caller supplied
                _warn_once(Path(path), exc)
                return None

        with _DEFAULT_LOCK:
            if not _DEFAULT_CACHE["loaded"]:
                try:
                    _DEFAULT_CACHE["model"] = cls.from_file(_MODEL_PATH)
                except _LOAD_ERRORS as exc:
                    _warn_once(_MODEL_PATH, exc)
                    _DEFAULT_CACHE["model"] = None
                _DEFAULT_CACHE["loaded"] = True
            return _DEFAULT_CACHE["model"]  # type: ignore[return-value]


# ── Threshold calibration ─────────────────────────────────────────────


def calibrate_threshold(
    model: InjectionClassifier,
    benign_texts: Iterable[str],
    target_fpr: float = 0.01,
) -> float:
    """Lowest score threshold whose false-positive rate on benign text is acceptable.

    ``benign_texts`` should be a sample of *your own* normal traffic (customer
    prompts, tickets, documents), because the packaged model was trained on public
    datasets and its default threshold may not fit your domain.  Scores come from
    :meth:`InjectionClassifier.predict_proba`, so long texts are windowed exactly
    as they are at request time.

    Returns the smallest ``t`` such that
    ``mean(predict_proba(text) >= t for text in benign_texts) <= target_fpr``.
    Feed the result straight into the guardrails::

        threshold = calibrate_threshold(InjectionClassifier.default(), my_logs)
        guard = LLMGuardrails(classifier_threshold=threshold)

    Raises :class:`ValueError` when ``benign_texts`` is empty or when
    ``target_fpr`` is outside ``[0, 1)``.
    """
    rate = float(target_fpr)
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"target_fpr must be in [0, 1), got {target_fpr!r}")

    scores = [float(model.predict_proba(text)) for text in benign_texts]
    count = len(scores)
    if count == 0:
        raise ValueError("calibrate_threshold needs at least one benign text")

    allowed = int(rate * count)  # how many benign rows may reach the threshold
    if allowed >= count:  # pragma: no cover - unreachable while target_fpr < 1
        return 0.0
    scores.sort(reverse=True)
    cut = scores[allowed]  # (allowed + 1)-th highest benign score
    threshold = math.nextafter(cut, math.inf)  # strictly above `cut`
    if threshold <= cut:  # pragma: no cover - nextafter cannot fail here
        threshold = cut + 1e-12
    return threshold


# ── Training ──────────────────────────────────────────────────────────


def _split_row(row: Any) -> tuple[Mapping[int, float], int]:
    """Accept ``(text, label)`` or ``(features, label)`` (dict rows also work)."""
    if isinstance(row, Mapping):
        features = row.get("features")
        if features is None:
            features = featurize(str(row.get("text", "")))
        return features, int(row.get("label", 0))
    if isinstance(row, (tuple, list)) and len(row) == 2:
        features, label = row
        if not isinstance(features, Mapping):
            features = featurize(str(features))
        return features, int(label)
    raise ValueError("rows must be (text, label) or (features, label) pairs")


def train(
    rows: Iterable[Any],
    epochs: int = 8,
    lr: float = 0.5,
    l2: float = 1e-6,
    seed: int = 0,
) -> InjectionClassifier:
    """Fit a logistic-regression classifier with plain SGD on log loss.

    ``rows`` is an iterable of ``(text, label)`` pairs; ``(features, label)``
    pairs are accepted too so callers can featurise once and train many models
    (cross-validation, leave-one-dataset-out, ...).

    Classes are balanced by weighting every sample with
    ``n_total / (2 * n_class)`` and the update order is reshuffled each epoch by a
    seeded :class:`random.Random`.  Weights below ``1e-4`` are pruned and at most
    the :data:`MAX_KEPT_WEIGHTS` largest-|w| weights are kept.
    """
    prepared: list[tuple[list[tuple[int, float]], int]] = []
    for row in rows:
        features, label = _split_row(row)
        prepared.append((list(features.items()), int(label)))

    if not prepared:
        return InjectionClassifier(weights={}, bias=0.0)

    n_total = len(prepared)
    n_pos = sum(1 for _, label in prepared if label)
    n_neg = n_total - n_pos
    weight_pos = (n_total / (2.0 * n_pos)) if n_pos else 0.0
    weight_neg = (n_total / (2.0 * n_neg)) if n_neg else 0.0

    # A flat list keyed by bucket avoids a multi-million-entry dict while training;
    # only the surviving weights are exported as a sparse dict at the end.
    weights = [0.0] * BUCKETS
    bias = 0.0

    rng = random.Random(seed)
    order = list(range(n_total))
    for _ in range(max(0, int(epochs))):
        rng.shuffle(order)
        for index in order:
            features, label = prepared[index]
            z = bias
            for bucket, value in features:
                z += weights[bucket] * value
            if z > 60.0:
                z = 60.0
            elif z < -60.0:
                z = -60.0
            probability = 1.0 / (1.0 + math.exp(-z))
            sample_weight = weight_pos if label else weight_neg
            error = (probability - label) * sample_weight
            for bucket, value in features:
                current = weights[bucket]
                weights[bucket] = current - lr * (error * value + l2 * current)
            bias -= lr * error

    kept = [index for index, weight in enumerate(weights) if weight and abs(weight) >= PRUNE_ABS]
    if len(kept) > MAX_KEPT_WEIGHTS:
        kept.sort(key=lambda index: abs(weights[index]), reverse=True)
        del kept[MAX_KEPT_WEIGHTS:]

    return InjectionClassifier(weights={index: weights[index] for index in kept}, bias=bias)
