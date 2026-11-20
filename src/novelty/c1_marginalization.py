"""C1: Bayesian answer marginalization over the calibrated retrieval posterior.

VisRAG weighted selection (Eq. 4-5) scores each page's own answer as
``P(a|q,d) * lambda(q,d)``. ``P(a|q,d)`` is the reciprocal perplexity of the
answer, so ``gen_logprobs`` holds the mean token log-probability, and
``lambda = softmax(s)`` has no temperature. Cosine scores span a narrow range,
so that ``lambda`` is close to uniform.

C1 replaces the per-page argmax with the answer marginal

    P(a|q) = sum_j P(a|q,d_j) p(d_j|q),    p(d|q) = softmax(s / tau),

and sums the mass of candidates that relaxed exact match treats as the same
answer.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..eval.metrics import relaxed_exact_match

# Log-probabilities above zero beyond float noise are not probabilities.
_LOGPROB_TOLERANCE = 1e-6

__all__ = [
    "answers_equivalent",
    "baseline_weighted_selection",
    "bayes_marginalized_selection",
    "calibrated_weighted_selection",
    "retrieval_log_posterior",
    "retrieval_posterior",
]


def retrieval_log_posterior(cos_scores: np.ndarray, tau: float) -> np.ndarray:
    """``log softmax(cos_scores / tau)`` for a single query's retrieved pages."""
    scores = np.asarray(cos_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(f"cos_scores must be a non-empty 1-D array, got shape {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("cos_scores must be finite")
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau must be a positive finite scalar, got {tau}")
    logits = scores / tau
    shifted = logits - logits.max()
    return shifted - np.log(np.exp(shifted).sum())


def retrieval_posterior(cos_scores: np.ndarray, tau: float) -> np.ndarray:
    """``p(d|q) = softmax(cos_scores / tau)`` for a single query's retrieved pages."""
    return np.exp(retrieval_log_posterior(cos_scores, tau))


def answers_equivalent(a: str, b: str) -> bool:
    """Whether relaxed exact match accepts either answer against the other.

    The 5% numeric tolerance is relative to the reference, so both directions
    are checked to make the relation symmetric.
    """
    return relaxed_exact_match(a, [b]) == 1.0 or relaxed_exact_match(b, [a]) == 1.0


def baseline_weighted_selection(
    candidates: list[str],
    gen_logprobs: np.ndarray,
    cos_scores: np.ndarray,
) -> str:
    """VisRAG Eq. (4)-(5): argmax of ``exp(gen_logprobs[i]) * softmax(cos_scores)[i]``.

    ``candidates[i]`` is the answer generated from page i alone.
    """
    return _weighted_selection(candidates, gen_logprobs, cos_scores, tau=1.0)


def calibrated_weighted_selection(
    candidates: list[str],
    gen_logprobs: np.ndarray,
    cos_scores: np.ndarray,
    tau: float = 0.02,
) -> str:
    """Eq. (4) with the calibrated weight ``p(d_i|q) = softmax(cos_scores / tau)``."""
    return _weighted_selection(candidates, gen_logprobs, cos_scores, tau=tau)


def bayes_marginalized_selection(
    candidates: list[str],
    logprob_matrix: np.ndarray,
    cos_scores: np.ndarray,
    tau: float = 0.02,
    group_equivalent: bool = True,
) -> tuple[str, dict]:
    """C1: ``a* = argmax_a sum_j P(a|q,d_j) p(d_j|q)``.

    ``logprob_matrix[i, j]`` is ``log P(candidates[i] | q, d_j)``, scored by
    the generator with page j as context, shape ``[n_candidates, k]``. Use
    ``-inf`` for a pair that was not scored.

    With ``group_equivalent``, candidates are visited from highest to lowest
    marginal. Each one joins the first group whose representative (its
    highest-scoring member) is relaxed-exact-match equivalent, otherwise it
    starts a new group. Comparing against the representative alone stops
    chains such as 100 ~ 104 ~ 108 from merging into one group. A group
    scores the sum of its members' marginals. Members with an identical
    string are the same event and are counted once. The returned answer is
    the representative of the best group.
    """
    names = _validate_candidates(candidates)
    log_p = retrieval_log_posterior(cos_scores, tau)
    k = log_p.shape[0]
    matrix = np.asarray(logprob_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(names), k):
        raise ValueError(
            f"logprob_matrix must have shape [n_candidates, k] = {(len(names), k)}, got {matrix.shape}"
        )
    _validate_logprobs(matrix, "logprob_matrix")

    log_marginal = _logsumexp(matrix + log_p[None, :], axis=1)
    order = np.argsort(-log_marginal, kind="mergesort")

    groups: list[list[int]] = []
    for index in order.tolist():
        for group in groups:
            if group_equivalent and answers_equivalent(names[index], names[group[0]]):
                group.append(index)
                break
        else:
            groups.append([index])

    group_log_scores = np.empty(len(groups), dtype=np.float64)
    for g, group in enumerate(groups):
        distinct: dict[str, float] = {}
        for index in group:
            value = float(log_marginal[index])
            distinct[names[index]] = max(distinct.get(names[index], -np.inf), value)
        group_log_scores[g] = _logsumexp(np.array(list(distinct.values())), axis=0)

    best = int(np.argmax(group_log_scores))
    total = _logsumexp(group_log_scores, axis=0)
    if np.isfinite(total):
        group_posterior = np.exp(group_log_scores - total)
    else:
        group_posterior = np.zeros_like(group_log_scores)

    info = {
        "posterior": np.exp(log_p),
        "candidate_scores": np.exp(log_marginal),
        "groups": groups,
        "group_answers": [names[group[0]] for group in groups],
        "group_scores": np.exp(group_log_scores),
        "group_posterior": group_posterior,
        "best_group": best,
        "answer_index": groups[best][0],
    }
    return names[groups[best][0]], info


def _weighted_selection(
    candidates: Sequence[str],
    gen_logprobs: np.ndarray,
    cos_scores: np.ndarray,
    tau: float,
) -> str:
    names = _validate_candidates(candidates)
    log_weight = retrieval_log_posterior(cos_scores, tau)
    logprobs = np.asarray(gen_logprobs, dtype=np.float64)
    if logprobs.shape != log_weight.shape or len(names) != log_weight.shape[0]:
        raise ValueError(
            "candidates, gen_logprobs, and cos_scores must all have length k; got "
            f"{len(names)}, {logprobs.shape}, {log_weight.shape}"
        )
    _validate_logprobs(logprobs, "gen_logprobs")
    return names[int(np.argmax(logprobs + log_weight))]


def _validate_candidates(candidates: Sequence[str]) -> list[str]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of answers, not a single string")
    names = ["" if c is None else str(c) for c in candidates]
    if not names:
        raise ValueError("candidates must not be empty")
    return names


def _validate_logprobs(values: np.ndarray, name: str) -> None:
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError(f"{name} must not contain NaN or +inf")
    if (values > _LOGPROB_TOLERANCE).any():
        raise ValueError(f"{name} are log-probabilities and must be <= 0")


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    peak = np.max(values, axis=axis, keepdims=True)
    safe_peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        total = np.log(np.exp(values - safe_peak).sum(axis=axis, keepdims=True)) + safe_peak
    return np.squeeze(total, axis=axis)
