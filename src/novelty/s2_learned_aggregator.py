"""S2: learned answer-correctness aggregator over calibrated retrieval features.

Each page i produces a candidate answer a_i, described by

    [p(d_i|q), P(a_i|q,d_i), cross_page_agreement, rank_norm]

``cross_page_agreement`` is the posterior mass of the other pages whose
answer is relaxed-exact-match equivalent to a_i. ``rank_norm`` is
i / (k - 1), with 0 for the top page. A small MLP, or a logistic regression
when ``hidden_dim = 0``, predicts whether a_i is correct and is trained with
BCEWithLogitsLoss. At inference the candidate with the highest predicted
probability is returned.

The checkpoint holds the network weights together with the fitted
(tau, alpha, rho) for C1-C3, so one file restores the whole calibrated
pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .c1_marginalization import answers_equivalent, retrieval_posterior

FEATURE_NAMES = ("posterior", "gen_prob", "cross_page_agreement", "rank_norm")
NUM_FEATURES = len(FEATURE_NAMES)

__all__ = [
    "FEATURE_NAMES",
    "NUM_FEATURES",
    "CalibratedAggregatorNet",
    "aggregate_select",
    "build_aggregator_features",
]


def build_aggregator_features(
    candidates: Sequence[str],
    gen_logprobs: np.ndarray,
    cos_scores: np.ndarray,
    tau: float = 0.02,
) -> np.ndarray:
    """Features ``[k, 4]`` for the per-page answers, with pages in rank order."""
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of answers, not a single string")
    names = ["" if c is None else str(c) for c in candidates]
    posterior = retrieval_posterior(cos_scores, tau)
    k = posterior.shape[0]
    logprobs = np.asarray(gen_logprobs, dtype=np.float64)
    if len(names) != k or logprobs.shape != (k,):
        raise ValueError(
            f"candidates, gen_logprobs, and cos_scores must all have length k; got {len(names)}, {logprobs.shape}, {k}"
        )
    if np.isnan(logprobs).any() or (logprobs > 1e-6).any():
        raise ValueError("gen_logprobs must be log-probabilities (<= 0, no NaN)")

    agreement = np.zeros(k, dtype=np.float64)
    for i in range(k):
        for j in range(k):
            if i != j and answers_equivalent(names[i], names[j]):
                agreement[i] += posterior[j]
    rank_norm = np.arange(k, dtype=np.float64) / (k - 1) if k > 1 else np.zeros(1)
    return np.stack([posterior, np.exp(logprobs), agreement, rank_norm], axis=1)


class CalibratedAggregatorNet(nn.Module):
    """Answer-correctness scorer: an MLP, or logistic regression when ``hidden_dim == 0``."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 0:
            raise ValueError(f"hidden_dim must be a non-negative integer, got {hidden_dim!r}")
        self.hidden_dim = hidden_dim
        if hidden_dim == 0:
            self.net = nn.Linear(NUM_FEATURES, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(NUM_FEATURES, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Correctness logits with shape ``features.shape[:-1]``."""
        if features.shape[-1] != NUM_FEATURES:
            raise ValueError(f"features must end in dimension {NUM_FEATURES}, got {tuple(features.shape)}")
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, features) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            logits = self(self._as_tensor(features))
        finally:
            self.train(was_training)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    def fit(
        self,
        features,
        labels,
        epochs: int = 300,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
        seed: int | None = 0,
    ) -> dict:
        """Full-batch Adam on BCEWithLogitsLoss. Returns the loss before and after training."""
        x = self._as_tensor(features)
        y = self._as_tensor(labels).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
            raise ValueError(f"features [n, {NUM_FEATURES}] and labels [n] do not match: {tuple(x.shape)}, {tuple(y.shape)}")
        if not torch.isfinite(x).all() or not ((y == 0) | (y == 1)).all():
            raise ValueError("features must be finite and labels must be 0 or 1")
        if epochs < 1 or lr <= 0:
            raise ValueError("epochs and lr must be positive")
        if seed is not None:
            torch.manual_seed(seed)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

        weight = None if pos_weight is None else torch.tensor(float(pos_weight), device=x.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train()
        with torch.no_grad():
            initial = float(loss_fn(self(x), y))
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            loss = loss_fn(self(x), y)
            loss.backward()
            optimizer.step()
        self.eval()
        with torch.no_grad():
            final = float(loss_fn(self(x), y))
        return {"loss_before": initial, "loss_after": final, "epochs": int(epochs), "n_examples": int(x.shape[0])}

    def save_checkpoint(self, path: str | Path, tau: float, alpha: float, rho: float, **extra) -> None:
        """Save the weights with the fitted (tau, alpha, rho) and any scalar extras."""
        hparams = {"tau": float(tau), "alpha": float(alpha), "rho": float(rho)}
        for name, value in hparams.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
                "hidden_dim": self.hidden_dim,
                "feature_names": list(FEATURE_NAMES),
                "hparams": {**hparams, **extra},
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str = "cpu") -> tuple["CalibratedAggregatorNet", dict]:
        """Restore a network and its hparams dict from ``save_checkpoint`` output."""
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        if list(payload.get("feature_names", [])) != list(FEATURE_NAMES):
            raise ValueError(f"Checkpoint {path} was trained on different features: {payload.get('feature_names')}")
        model = cls(hidden_dim=int(payload["hidden_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, dict(payload["hparams"])

    def _as_tensor(self, values) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(values, torch.Tensor):
            return values.detach().to(device=device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)


def aggregate_select(
    model: CalibratedAggregatorNet,
    candidates: Sequence[str],
    gen_logprobs: np.ndarray,
    cos_scores: np.ndarray,
    tau: float = 0.02,
) -> tuple[str, np.ndarray]:
    """Return the candidate with the highest predicted correctness and every candidate's probability."""
    features = build_aggregator_features(candidates, gen_logprobs, cos_scores, tau=tau)
    probs = model.predict_proba(features)
    return str(list(candidates)[int(np.argmax(probs))]), probs
