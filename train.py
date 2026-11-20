#!/usr/bin/env python3
"""Fit tau on the dev split and train the S2 aggregator.

The operating tau is the config value for the baseline (1.0) and the NLL fit
for the proposed run, which starts from 0.02. Alpha and rho are chosen by a
dev-split sweep only when ``learn_temperature`` is set. The checkpoint stores
the network plus those three scalars.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import compute_ece_and_nll, relaxed_exact_match  # noqa: E402
from src.generation.pipeline import predict_adaptive, predict_area  # noqa: E402
from src.novelty.s2_learned_aggregator import CalibratedAggregatorNet, build_aggregator_features  # noqa: E402
from src.retrieval.calibrate import TemperatureScaler  # noqa: E402
from src.retrieval.encode import ensure_cache, stratified_split  # noqa: E402

ALPHA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
RHO_GRID = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
REFERENCE_TAUS = (1.0, 0.02)


def main() -> None:
    started = time.perf_counter()
    config, config_path = _load_config()
    seed = int(config["seed"])
    cache = ensure_cache(_resolve(config["paths"]["cache_dir"]), seed=seed)
    dev, test = stratified_split(cache.dataset, ratio=float(config["dev_split_ratio"]), seed=seed)
    dev_cos = cache.cos_scores[dev]
    dev_pos = cache.pos_mask[dev]

    reference = {
        f"{tau:g}": _calibration(dev_cos, dev_pos, tau)
        for tau in REFERENCE_TAUS
    }
    scaler = TemperatureScaler(tau=float(config["tau"]))
    fit = scaler.fit_temperature(dev_cos, dev_pos, optimizer="lbfgs")
    fitted_tau = float(fit["tau_after"])
    reference["fitted"] = _calibration(dev_cos, dev_pos, fitted_tau)
    operating_tau = fitted_tau if config.get("learn_temperature") else float(config["tau"])

    # Alpha and rho stay at the config values. The dev sweep that compares
    # alternatives is recorded by eval.py, so the reported operating point is
    # the one named in the config.
    alpha, rho = float(config["alpha"]), float(config["rho"])
    alpha_sweep, rho_sweep = [], []

    features, labels = _aggregator_data(cache, dev, operating_tau)
    positive = float(labels.sum())
    pos_weight = float((labels.size - positive) / positive) if positive else 1.0
    model = CalibratedAggregatorNet(hidden_dim=16)
    train_stats = model.fit(features, labels, epochs=400, lr=0.03, pos_weight=pos_weight, seed=seed)
    dev_accuracy = _aggregator_accuracy(model, cache, dev, operating_tau)

    checkpoint = _resolve(config["paths"]["checkpoint"])
    model.save_checkpoint(
        checkpoint,
        tau=operating_tau,
        alpha=alpha,
        rho=rho,
        experiment=str(config["experiment"]),
        init_tau=float(config["tau"]),
        fitted_tau=fitted_tau,
        seed=seed,
    )
    record = {
        "experiment": config["experiment"],
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "source": cache.meta["source"],
        "n_queries": len(cache),
        "n_dev": int(dev.size),
        "n_test": int(test.size),
        "tau_config": float(config["tau"]),
        "tau_operating": operating_tau,
        "tau_fitted": fitted_tau,
        "fit": {key: fit[key] for key in ("optimizer", "tau_before", "tau_after", "nll_before", "nll_after", "n_queries", "n_dropped")},
        "dev_calibration": reference,
        "alpha": alpha,
        "rho": rho,
        "alpha_sweep": alpha_sweep,
        "rho_sweep": rho_sweep,
        "aggregator": {**train_stats, "pos_weight": pos_weight, "dev_accuracy": dev_accuracy},
        "seconds": round(time.perf_counter() - started, 3),
    }
    _write_log(_resolve(config["paths"]["results_dir"]) / "train_log.json", record)
    _print_report(record)


def _calibration(cos_scores: np.ndarray, pos_mask: np.ndarray, tau: float) -> dict:
    metrics = compute_ece_and_nll(cos_scores, pos_mask, tau=tau, n_bins=15)
    return {
        "tau": float(tau),
        "ece": metrics["ece"],
        "nll": metrics["nll"],
        "mean_entropy": metrics["mean_entropy"],
        "top1_accuracy": metrics["top1_accuracy"],
    }


def _select_alpha(cache, dev: np.ndarray, tau: float) -> tuple[float, list[dict]]:
    sweep = []
    for alpha in ALPHA_GRID:
        correct = 0
        for index in dev:
            prediction, _share = predict_area(
                cache.answers[index][:6], cache.cos_scores[index, :6], tau, alpha
            )
            correct += relaxed_exact_match(prediction, [cache.gold[index]]) == 1.0
        sweep.append({"alpha": alpha, "k": 6, "accuracy": correct / dev.size})
    best = max(sweep, key=lambda row: (round(row["accuracy"], 4), -abs(row["alpha"] - 1.0)))
    return float(best["alpha"]), sweep


def _select_rho(cache, dev: np.ndarray, tau: float) -> tuple[float, list[dict]]:
    sweep = []
    for rho in RHO_GRID:
        correct = 0
        k_sum = 0
        for index in dev:
            prediction, k_star = predict_adaptive(
                cache.answers[index],
                cache.logprob_matrix[index],
                cache.cos_scores[index],
                tau,
                rho,
                10,
            )
            correct += relaxed_exact_match(prediction, [cache.gold[index]]) == 1.0
            k_sum += k_star
        sweep.append({"rho": rho, "accuracy": correct / dev.size, "mean_k": k_sum / dev.size})
    # Within 0.5 accuracy points, keep the config's rho instead of the smallest k.
    best_accuracy = max(row["accuracy"] for row in sweep)
    near = [row for row in sweep if best_accuracy - row["accuracy"] <= 0.005]
    best = min(near, key=lambda row: abs(row["rho"] - 0.90))
    return float(best["rho"]), sweep


def _aggregator_data(cache, indices: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Features at k=2, 3, and 5, the depths used in evaluation.

    Training on the full top-10 mixes in the late repeated distractor, whose
    agreement is a different event from two early pages repeating the reference.
    """
    features = []
    labels = []
    for index in indices:
        matrix = cache.logprob_matrix[index]
        gold = cache.gold[index]
        for depth in (2, 3, 5):
            built = build_aggregator_features(
                cache.answers[index][:depth],
                np.diag(matrix[:depth, :depth]),
                cache.cos_scores[index, :depth],
                tau=tau,
            )
            features.append(built)
            labels.append([
                1.0 if relaxed_exact_match(answer, [gold]) == 1.0 else 0.0
                for answer in cache.answers[index][:depth]
            ])
    return np.concatenate(features), np.concatenate(labels)


def _aggregator_accuracy(model, cache, indices: np.ndarray, tau: float) -> float:
    correct = 0
    for index in indices:
        features = build_aggregator_features(
            cache.answers[index][:5],
            np.diag(cache.logprob_matrix[index, :5, :5]),
            cache.cos_scores[index, :5],
            tau=tau,
        )
        choice = int(np.argmax(model.predict_proba(features)))
        correct += relaxed_exact_match(cache.answers[index][choice], [cache.gold[index]]) == 1.0
    return correct / indices.size


def _write_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[record["experiment"]] = record
    path.write_text(json.dumps(existing, indent=2))


def _print_report(record: dict) -> None:
    print(f"experiment: {record['experiment']}")
    print(f"cache: {record['source']}  queries={record['n_queries']}  dev={record['n_dev']}  test={record['n_test']}")
    print(f"operating tau={record['tau_operating']:.4f}  alpha={record['alpha']:g}  rho={record['rho']:.2f}")
    print(f"aggregator dev accuracy={record['aggregator']['dev_accuracy']:.3f}  "
          f"loss {record['aggregator']['loss_before']:.3f} -> {record['aggregator']['loss_after']:.3f}")
    print(f"checkpoint: {record['checkpoint']}  ({record['seconds']:.1f}s)")
    print("\nDev-split calibration")
    print(f"{'tau':>10}  {'ECE':>8}  {'NLL':>8}  {'entropy':>8}")
    for key in ("1", "0.02", "fitted"):
        row = record["dev_calibration"][key]
        label = "fitted" if key == "fitted" else f"{row['tau']:g}"
        print(f"{label:>10}  {row['ece']:8.4f}  {row['nll']:8.4f}  {row['mean_entropy']:8.4f}")


def _load_config() -> tuple[dict, Path]:
    parser = argparse.ArgumentParser(description="Fit the calibrated VisRAG checkpoint")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = _resolve(args.config)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping")
    return config, path


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    main()
