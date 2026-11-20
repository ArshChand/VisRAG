"""C3: adaptive retrieval depth from the calibrated posterior.

VisRAG passes a fixed top-k to the generator. C3 keeps the smallest prefix
of the ranked pages whose calibrated mass reaches ``rho``:

    k* = min(k_max, min{k : sum_{i<=k} p(d_i|q) >= rho}),   p = softmax(s / tau).

A sharp posterior gives k* = 1. A flat one falls back to k_max.
"""

from __future__ import annotations

import math

import numpy as np

from .c1_marginalization import retrieval_posterior

# Lets rho = 0.8 accept a prefix whose mass is 0.8 up to float rounding.
_MASS_TOLERANCE = 1e-12

__all__ = ["adaptive_k_indices", "select_adaptive_k"]


def adaptive_k_indices(
    cos_scores: np.ndarray,
    tau: float = 0.02,
    rho: float = 0.90,
    k_max: int = 6,
    renormalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices into ``cos_scores`` of the k* kept pages, best first, with their posterior.

    The posterior is computed over every page in ``cos_scores`` and ranked
    in descending order, with ties kept in input order. With
    ``renormalize``, the kept probabilities sum to 1 and can be used as
    generation weights directly. Otherwise they are the raw prefix of the
    full posterior.
    """
    rho = float(rho)
    if not math.isfinite(rho) or not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if isinstance(k_max, bool) or not isinstance(k_max, (int, np.integer)) or int(k_max) < 1:
        raise ValueError(f"k_max must be a positive integer, got {k_max!r}")

    posterior = retrieval_posterior(cos_scores, tau)
    order = np.argsort(-posterior, kind="mergesort")
    ranked = posterior[order]
    limit = min(int(k_max), ranked.shape[0])
    reached = np.nonzero(np.cumsum(ranked[:limit]) >= rho - _MASS_TOLERANCE)[0]
    k_star = int(reached[0]) + 1 if reached.size else limit

    kept = ranked[:k_star].copy()
    if renormalize:
        kept /= kept.sum()
    return order[:k_star], kept


def select_adaptive_k(
    cos_scores: np.ndarray,
    tau: float = 0.02,
    rho: float = 0.90,
    k_max: int = 6,
) -> tuple[int, np.ndarray]:
    """C3: return ``k*`` and the renormalized posterior of the top ``k*`` pages, best first.

    Use ``adaptive_k_indices`` to find which pages were kept when
    ``cos_scores`` is not already sorted in descending order.
    """
    indices, kept = adaptive_k_indices(cos_scores, tau=tau, rho=rho, k_max=k_max)
    return int(indices.shape[0]), kept
