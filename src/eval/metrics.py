"""Retrieval, generation, and calibration metrics for visrag-calibrated.

VisRAG Section 3.3 reports MRR@10 and Recall@10, and generation accuracy with
the relaxed exact match used by ChartQA and PlotQA: a numeric answer is
correct when it is within 5% of the reference. Text answers are normalized,
then accepted on exact match or a token-boundary substring match.

Calibration uses the retrieval posterior p(d|q) = softmax(cos(q, d) / tau).
Expected Calibration Error follows Guo et al. (15 equal-width bins on the
top-1 confidence). Negative log-likelihood is the mean negative log
probability of the relevant set.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

# ChartQA / PlotQA relative tolerance cited by VisRAG Section 3.3.
NUMERIC_RELATIVE_TOLERANCE = 0.05

_ARTICLE_TOKENS = frozenset({"a", "an", "the"})
_NUMBER_BODY = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)

__all__ = [
    "NUMERIC_RELATIVE_TOLERANCE",
    "compute_ece_and_nll",
    "compute_retrieval_metrics",
    "is_within_5_percent",
    "relaxed_exact_match",
]


def compute_retrieval_metrics(
    sim_matrix: np.ndarray,
    pos_doc_ids: list,
    doc_ids: list,
    k: int = 10,
) -> dict:
    """Exact MRR@k and Recall@k from a query-document similarity matrix.

    Scores are ranked descending. Ties keep the smaller column index first
    (stable mergesort). MRR@k is the mean reciprocal rank of the first
    relevant document, and is 0 when that document is not in the top k.
    Recall@k is the mean fraction of relevant documents that appear in the
    top k. Both are fractions in [0, 1]; multiply by 100 to match the
    VisRAG tables. A relevant id missing from ``doc_ids`` cannot be retrieved
    and still counts in the recall denominator.

    ``pos_doc_ids[i]`` is one document id or a sequence of ids (SlideVQA has
    multiple positive pages). ``doc_ids[j]`` is the corpus id of column j.
    """
    scores = np.asarray(sim_matrix, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"sim_matrix must have shape [n_queries, n_docs], got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("sim_matrix must be finite")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)

    n_queries, n_docs = scores.shape
    if n_queries == 0 or n_docs == 0:
        raise ValueError("sim_matrix must contain at least one query and one document")
    canonical_docs = [_canonicalize_id(doc_id) for doc_id in list(doc_ids)]
    if len(canonical_docs) != n_docs:
        raise ValueError(
            f"doc_ids length {len(canonical_docs)} does not match n_docs {n_docs}"
        )
    if len(set(canonical_docs)) != len(canonical_docs):
        raise ValueError("doc_ids contains duplicates")
    if len(pos_doc_ids) != n_queries:
        raise ValueError(
            f"pos_doc_ids length {len(pos_doc_ids)} does not match n_queries {n_queries}"
        )

    order = np.argsort(-scores, axis=1, kind="mergesort")
    topk = min(k, n_docs)
    reciprocal_ranks = np.empty(n_queries, dtype=np.float64)
    recalls = np.empty(n_queries, dtype=np.float64)

    for query_index in range(n_queries):
        positives = _positive_id_set(pos_doc_ids[query_index])
        ranked_columns = order[query_index, :topk]
        top_ids = {canonical_docs[int(column)] for column in ranked_columns}
        reciprocal_ranks[query_index] = 0.0
        for rank, column in enumerate(ranked_columns, start=1):
            if canonical_docs[int(column)] in positives:
                reciprocal_ranks[query_index] = 1.0 / rank
                break
        if positives:
            recalls[query_index] = len(positives & top_ids) / len(positives)
        else:
            recalls[query_index] = 0.0

    return {
        f"MRR@{k}": float(reciprocal_ranks.mean()),
        f"Recall@{k}": float(recalls.mean()),
        "k": k,
        "n_queries": int(n_queries),
    }


def is_within_5_percent(pred: str, target: str) -> bool:
    """Return whether both strings are numeric and within a 5% relative error.

    Percentages are parsed on the percent scale and then divided by 100, so
    ``"50%"`` matches ``"0.5"`` and does not match ``"50"``. A reference of
    zero matches only a prediction of zero. Non-numeric strings return False
    rather than raising; text comparison lives in ``relaxed_exact_match``.
    """
    pred_value = _try_parse_number("" if pred is None else str(pred))
    target_value = _try_parse_number("" if target is None else str(target))
    if pred_value is None or target_value is None:
        return False
    return _within_relative_tolerance(pred_value, target_value)


def relaxed_exact_match(pred: str, targets: list[str]) -> float:
    """Relaxed exact match against one or more reference answers.

    Returns 1.0 when ``pred`` matches any reference and 0.0 otherwise.
    Dataset accuracy is the mean of this value over examples.

    A pair of pure numbers or percentages uses the 5% relative tolerance.
    Every other pair is lowercased and stripped, punctuation is removed,
    articles are dropped, and the prediction is accepted on exact match or
    when either normalized string contains the other on token boundaries.
    """
    if isinstance(targets, (str, bytes)):
        raise TypeError("targets must be a sequence of reference answers, not a single string")
    if targets is None:
        return 0.0
    references = list(targets)
    if not references:
        return 0.0
    prediction = "" if pred is None else str(pred)
    for target in references:
        reference = "" if target is None else str(target)
        if _answers_match(prediction, reference):
            return 1.0
    return 0.0


def compute_ece_and_nll(
    cos_scores: np.ndarray,
    pos_mask: np.ndarray,
    tau: float,
    n_bins: int = 15,
) -> dict:
    """ECE, NLL, and entropy of ``p(d|q) = softmax(cos_scores / tau)``.

    ``cos_scores`` and ``pos_mask`` have shape ``[n_queries, k]``. A positive
    entry of ``pos_mask`` marks a relevant retrieved document. The posterior
    is a distribution over those k documents.

    NLL is the mean of ``-log sum_{j in relevant} p(d_j|q)``. Queries with no
    positive inside the top k are omitted from the NLL, because the truncated
    posterior gives that relevant set probability zero.

    ECE is the top-1 expected calibration error. Confidence is ``max p(d|q)``
    and the prediction is correct when the argmax document is relevant. Ties
    take the smallest column. Bins are ``n_bins`` equal-width intervals on
    ``[0, 1]``, with confidence 1 placed in the last bin. Empty bins have
    NaN confidence and accuracy so a reliability diagram leaves a gap there.
    Entropy is the natural log, averaged over queries. Its maximum is
    ``log(k)``.
    """
    scores = np.asarray(cos_scores, dtype=np.float64)
    raw_mask = np.asarray(pos_mask)
    if scores.ndim != 2:
        raise ValueError(f"cos_scores must have shape [n_queries, k], got {scores.shape}")
    if raw_mask.shape != scores.shape:
        raise ValueError(
            f"pos_mask shape {raw_mask.shape} does not match cos_scores shape {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("cos_scores must be finite")
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau must be a positive finite scalar, got {tau}")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    n_bins = int(n_bins)

    positives = raw_mask.astype(bool) if raw_mask.dtype == bool else raw_mask.astype(np.float64) > 0
    log_probs = _row_log_softmax(scores / tau)
    probs = np.exp(log_probs)
    entropy = -(probs * np.where(probs > 0.0, log_probs, 0.0)).sum(axis=1)
    log_pos_mass = _masked_logsumexp(log_probs, positives)
    valid = np.isfinite(log_pos_mass)
    if valid.any():
        nll = float((-log_pos_mass[valid]).mean())
    else:
        nll = float("nan")

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = positives[np.arange(scores.shape[0]), predicted].astype(np.float64)
    bin_ids = np.clip(np.floor(confidence * n_bins).astype(np.int64), 0, n_bins - 1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_confidence = np.full(n_bins, np.nan, dtype=np.float64)
    bin_accuracy = np.full(n_bins, np.nan, dtype=np.float64)
    ece = 0.0
    n_queries = scores.shape[0]
    for bin_index in range(n_bins):
        selected = bin_ids == bin_index
        count = int(selected.sum())
        bin_counts[bin_index] = count
        if count == 0:
            continue
        bin_confidence[bin_index] = float(confidence[selected].mean())
        bin_accuracy[bin_index] = float(correct[selected].mean())
        ece += (count / n_queries) * abs(bin_accuracy[bin_index] - bin_confidence[bin_index])

    return {
        "ece": float(ece),
        "nll": nll,
        "mean_entropy": float(entropy.mean()),
        "mean_confidence": float(confidence.mean()),
        "top1_accuracy": float(correct.mean()),
        "n_queries": int(n_queries),
        "nll_queries": int(valid.sum()),
        "tau": tau,
        "bins": {
            "edges": bin_edges,
            "counts": bin_counts,
            "confidence": bin_confidence,
            "accuracy": bin_accuracy,
            "gap": np.abs(bin_accuracy - bin_confidence),
        },
    }


def _canonicalize_id(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _positive_id_set(item: Any) -> set:
    if isinstance(item, np.ndarray):
        if item.ndim == 0:
            values = [_canonicalize_id(item)]
        else:
            values = [_canonicalize_id(value) for value in item.tolist()]
    elif isinstance(item, (str, bytes)):
        values = [_canonicalize_id(item)]
    elif isinstance(item, (list, tuple, set)):
        values = [_canonicalize_id(value) for value in item]
    else:
        values = [_canonicalize_id(item)]
    return set(values)


def _try_parse_number(text: str) -> float | None:
    """Parse a single float or percentage. Return None when text remains."""
    if text is None:
        return None
    parsed = str(text).strip().lower()
    if not parsed:
        return None
    if len(parsed) > 1 and parsed.endswith("."):
        parsed = parsed[:-1].rstrip()
    parsed = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", parsed)
    parsed = re.sub(r"[$€£¥]", "", parsed).strip()
    is_percent = False
    for suffix in ("percentage", "percent", "%"):
        if parsed.endswith(suffix):
            is_percent = True
            parsed = parsed[: -len(suffix)].strip()
            break
    if _NUMBER_BODY.fullmatch(parsed or "") is None:
        return None
    value = float(parsed)
    if not math.isfinite(value):
        return None
    if is_percent:
        value /= 100.0
    return value


def _within_relative_tolerance(pred_value: float, target_value: float) -> bool:
    if target_value == 0.0:
        return pred_value == 0.0
    relative = abs(pred_value - target_value) / abs(target_value)
    return relative <= NUMERIC_RELATIVE_TOLERANCE or math.isclose(
        relative,
        NUMERIC_RELATIVE_TOLERANCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _normalize_text(text: str, *, drop_articles: bool) -> str:
    normalized = str(text).strip().lower()
    normalized = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", normalized)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    tokens = normalized.split()
    if drop_articles:
        tokens = [token for token in tokens if token not in _ARTICLE_TOKENS]
    return " ".join(tokens)


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {text} "


def _text_match(pred: str, target: str) -> bool:
    pred_exact = _normalize_text(pred, drop_articles=False)
    target_exact = _normalize_text(target, drop_articles=False)
    if pred_exact == target_exact:
        if pred_exact:
            return True
        return pred.strip().lower() == target.strip().lower()
    pred_norm = _normalize_text(pred, drop_articles=True)
    target_norm = _normalize_text(target, drop_articles=True)
    if not pred_norm or not target_norm:
        return False
    if pred_norm == target_norm:
        return True
    return _contains_phrase(pred_norm, target_norm) or _contains_phrase(target_norm, pred_norm)


def _answers_match(pred: str, target: str) -> bool:
    pred_value = _try_parse_number(pred)
    target_value = _try_parse_number(target)
    if pred_value is not None and target_value is not None:
        return is_within_5_percent(pred, target)
    return _text_match(pred, target)


def _row_log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _masked_logsumexp(log_probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Log-sum-exp of ``log_probs`` on ``mask``. NaN on rows with an empty mask."""
    out = np.full(log_probs.shape[0], np.nan, dtype=np.float64)
    valid = mask.any(axis=1)
    if not np.any(valid):
        return out
    selected = log_probs[valid]
    selected_mask = mask[valid]
    masked = np.where(selected_mask, selected, -np.inf)
    row_max = masked.max(axis=1)
    shifted = np.where(selected_mask, masked - row_max[:, None], -np.inf)
    out[valid] = row_max + np.log(np.exp(shifted).sum(axis=1))
    return out
