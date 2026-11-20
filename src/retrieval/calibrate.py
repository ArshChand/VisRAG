"""Temperature scaling for the VisRAG retrieval posterior.

VisRAG trains the retriever with InfoNCE at tau = 0.02. Cosine similarities
occupy a narrow range, so the same scale is the starting point for

    p(d | q) = softmax(cos(q, d) / tau).

``log_tau`` is the learnable parameter, which keeps tau positive. Fit it on
the dev split by minimizing the negative log-likelihood of the relevant set,
the one-parameter temperature-scaling objective.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Softmax temperature with a learnable ``log_tau``, initialized at 0.02."""

    def __init__(self, tau: float = 0.02) -> None:
        super().__init__()
        tau = float(tau)
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"tau must be a positive finite scalar, got {tau}")
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau), dtype=torch.float32))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def extra_repr(self) -> str:
        return f"tau={float(self.tau.detach().cpu()):.6f}"

    def forward(self, cos_scores: torch.Tensor) -> torch.Tensor:
        """Return ``p(d|q) = softmax(cos_scores / tau)`` over the last dimension."""
        if not isinstance(cos_scores, torch.Tensor):
            cos_scores = torch.as_tensor(cos_scores, dtype=torch.float32, device=self.log_tau.device)
        return torch.softmax(cos_scores / self.tau, dim=-1)

    def nll_loss(self, cos_scores: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        """Mean ``-log sum_{relevant} p(d|q)`` over queries that have a positive.

        This is the NLL minimized by ``fit_temperature`` and reported by
        ``src.eval.metrics.compute_ece_and_nll``. Queries with an all-negative
        mask are dropped: the truncated posterior assigns them probability 0.
        """
        scores, mask = self._prepare(cos_scores, pos_mask)
        valid = mask.any(dim=-1)
        if not bool(valid.any()):
            raise ValueError("NLL is undefined: pos_mask has no positive documents")
        log_probs = torch.log_softmax(scores / self.tau, dim=-1)
        masked = log_probs.masked_fill(~mask, float("-inf"))
        log_pos_mass = torch.logsumexp(masked, dim=-1)
        return -log_pos_mass[valid].mean()

    def fit_temperature(
        self,
        cos_scores,
        pos_mask,
        optimizer: str = "lbfgs",
        max_iter: int | None = None,
        lr: float | None = None,
    ) -> dict:
        """Fit ``log_tau`` on dev-split scores by minimizing NLL.

        ``optimizer`` is ``"lbfgs"`` (default; the usual temperature-scaling
        solver) or ``"adam"``. The module is updated in place. Returns the
        temperature and NLL before and after the fit.
        """
        scores, mask = self._prepare(cos_scores, pos_mask)
        valid = mask.any(dim=-1)
        n_queries = int(scores.shape[0])
        n_dropped = int((~valid).sum().item())
        if n_dropped:
            scores = scores[valid]
            mask = mask[valid]
        if scores.shape[0] == 0:
            raise ValueError("NLL is undefined: pos_mask has no positive documents")

        solver = optimizer.lower().replace("-", "").replace("_", "")
        if solver == "lbfgs":
            max_iter = 50 if max_iter is None else max_iter
            lr = 0.5 if lr is None else lr
        elif solver == "adam":
            max_iter = 200 if max_iter is None else max_iter
            lr = 0.05 if lr is None else lr
        else:
            raise ValueError("optimizer must be 'lbfgs' or 'adam'")
        if max_iter < 1:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")

        tau_before = float(self.tau.detach().cpu())
        with torch.no_grad():
            nll_before = float(self.nll_loss(scores, mask).detach().cpu())
        if not math.isfinite(nll_before):
            raise RuntimeError("NLL is non-finite before the temperature fit")

        with torch.enable_grad():
            if solver == "lbfgs":
                self._fit_lbfgs(scores, mask, lr=lr, max_iter=max_iter)
            else:
                self._fit_adam(scores, mask, lr=lr, max_iter=max_iter)

        tau_after = float(self.tau.detach().cpu())
        with torch.no_grad():
            nll_after = float(self.nll_loss(scores, mask).detach().cpu())
        if not math.isfinite(tau_after) or not math.isfinite(nll_after):
            raise RuntimeError("temperature fit produced a non-finite tau or NLL")

        return {
            "optimizer": solver,
            "tau_before": tau_before,
            "tau_after": tau_after,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "n_queries": n_queries - n_dropped,
            "n_dropped": n_dropped,
            "max_iter": int(max_iter),
            "lr": float(lr),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save ``log_tau`` so the fitted temperature can be reloaded."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"log_tau": self.log_tau.detach().cpu()}, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load ``log_tau`` from a checkpoint written by ``save_checkpoint``."""
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "log_tau" not in payload:
            raise ValueError(f"Checkpoint {path} has no log_tau")
        loaded = payload["log_tau"]
        if not torch.is_tensor(loaded):
            raise ValueError(f"Checkpoint {path} log_tau is not a tensor")
        with torch.no_grad():
            self.log_tau.copy_(
                loaded.reshape(()).to(dtype=self.log_tau.dtype, device=self.log_tau.device)
            )

    def _prepare(
        self,
        cos_scores,
        pos_mask,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(cos_scores, torch.Tensor):
            scores = cos_scores.detach().to(dtype=torch.float32, device=self.log_tau.device)
        else:
            scores = torch.tensor(
                np.asarray(cos_scores, dtype=np.float32),
                dtype=torch.float32,
                device=self.log_tau.device,
            )
        if isinstance(pos_mask, torch.Tensor):
            mask = pos_mask.detach().to(device=self.log_tau.device)
        else:
            mask = torch.tensor(np.asarray(pos_mask), device=self.log_tau.device)
        if scores.ndim != 2:
            raise ValueError(f"cos_scores must have shape [n_queries, k], got {tuple(scores.shape)}")
        if mask.shape != scores.shape:
            raise ValueError(
                f"pos_mask shape {tuple(mask.shape)} does not match cos_scores shape {tuple(scores.shape)}"
            )
        if not torch.isfinite(scores).all():
            raise ValueError("cos_scores must be finite")
        if mask.dtype != torch.bool:
            mask = mask > 0
        return scores, mask

    def _fit_lbfgs(
        self,
        cos_scores: torch.Tensor,
        pos_mask: torch.Tensor,
        lr: float,
        max_iter: int,
    ) -> None:
        optimizer = torch.optim.LBFGS(
            [self.log_tau],
            lr=lr,
            max_iter=max_iter,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = self.nll_loss(cos_scores, pos_mask)
            loss.backward()
            return loss

        optimizer.step(closure)

    def _fit_adam(
        self,
        cos_scores: torch.Tensor,
        pos_mask: torch.Tensor,
        lr: float,
        max_iter: int,
    ) -> None:
        optimizer = torch.optim.Adam([self.log_tau], lr=lr)
        for _ in range(max_iter):
            optimizer.zero_grad()
            loss = self.nll_loss(cos_scores, pos_mask)
            loss.backward()
            optimizer.step()
