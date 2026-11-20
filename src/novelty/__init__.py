"""Novelty modules layered on the VisRAG pipeline (C1, C2, C3, S2)."""

from .c1_marginalization import (
    answers_equivalent,
    baseline_weighted_selection,
    bayes_marginalized_selection,
    calibrated_weighted_selection,
    retrieval_log_posterior,
    retrieval_posterior,
)
from .c2_tiling import (
    baseline_horizontal_concat,
    compute_tiling_layout,
    relevance_proportional_tiling,
    relevance_tiling_weights,
)
from .c3_adaptive_k import adaptive_k_indices, select_adaptive_k
from .s2_learned_aggregator import (
    CalibratedAggregatorNet,
    aggregate_select,
    build_aggregator_features,
)

__all__ = [
    "CalibratedAggregatorNet",
    "adaptive_k_indices",
    "aggregate_select",
    "answers_equivalent",
    "baseline_horizontal_concat",
    "baseline_weighted_selection",
    "bayes_marginalized_selection",
    "build_aggregator_features",
    "calibrated_weighted_selection",
    "compute_tiling_layout",
    "relevance_proportional_tiling",
    "relevance_tiling_weights",
    "retrieval_log_posterior",
    "retrieval_posterior",
    "select_adaptive_k",
]
