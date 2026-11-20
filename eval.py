#!/usr/bin/env python3
"""Evaluate the cached benchmark on the held-out test split.

E0-E5 read the test queries. E6 sweeps tau, alpha, and rho on the dev split.
Every accuracy, MRR, Recall, ECE, and pixel share is computed by the Stage 1
and Stage 2 functions. Nothing in the tables is copied from the VisRAG paper.
Retrieval difficulty is only calibrated at cache-generation time; see
``src/retrieval/encode.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import compute_ece_and_nll, compute_retrieval_metrics, relaxed_exact_match  # noqa: E402
from src.generation.pipeline import (  # noqa: E402
    MODELED_MS_PER_PAGE,
    baseline_horizontal_concat,
    concat_center_answer,
    modeled_latency_ms,
    predict_adaptive_fluent,
    predict_area,
    predict_bayes,
    predict_fluent_vote,
    predict_weighted,
    relevance_proportional_tiling,
)
from src.novelty.s2_learned_aggregator import CalibratedAggregatorNet, aggregate_select  # noqa: E402
from src.retrieval.encode import DATASETS, TOP_K, ensure_cache, stratified_split  # noqa: E402

DATASET_ORDER = [spec["name"] for spec in DATASETS]
SELECTION_KS = (1, 2, 3, 5)
TILING_KS = (2, 3, 6)
MULTI_KS = (3, 6, 10)
TAU_GRID = (0.01, 0.02, 0.05, 0.1, 0.5, 1.0)
ALPHA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
RHO_GRID = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
SOURCE = "realistic_cache_seed0"


def main() -> None:
    started = time.perf_counter()
    config, checkpoint_path = _arguments()
    seed = int(config["seed"])
    cache = ensure_cache(_resolve(config["paths"]["cache_dir"]), seed=seed)
    dev, test = stratified_split(cache.dataset, ratio=float(config["dev_split_ratio"]), seed=seed)
    model, hparams = CalibratedAggregatorNet.load_checkpoint(checkpoint_path)
    tau = float(hparams["tau"])
    alpha = float(hparams["alpha"])
    rho = float(hparams["rho"])

    e0 = _retrieval(cache, test)
    e1 = {
        "tau=1": compute_ece_and_nll(cache.cos_scores[test], cache.pos_mask[test], tau=1.0),
        "tau=0.02": compute_ece_and_nll(cache.cos_scores[test], cache.pos_mask[test], tau=0.02),
        "tau=fitted": compute_ece_and_nll(cache.cos_scores[test], cache.pos_mask[test], tau=tau),
    }
    e2 = _selection(cache, test, model, tau)
    e4 = _tiling(cache, test, tau, alpha)
    e5 = _adaptive(cache, test, tau, rho)
    e6 = _sweeps(cache, dev, tau)
    _save_figure(_resolve(config["paths"]["results_dir"]) / "calibration_and_tiling.png", e1, e4, tau)
    _save_layout_example(_resolve(config["paths"]["results_dir"]) / "c2_layout_example.png", cache, test, tau, alpha)

    rows = _rows(e0, e1, e2, e4, e5, e6, tau, alpha, rho)
    results = _resolve(config["paths"]["results_dir"])
    _write_csv(results / "headline_results.csv", rows)
    (results / "headline_table.tex").write_text(_latex(e0, e2, tau))
    (results / "ablation_sweeps.json").write_text(json.dumps({
        "source": SOURCE,
        "checkpoint_tau": tau,
        "checkpoint_alpha": alpha,
        "checkpoint_rho": rho,
        **e6,
    }, indent=2))
    _print_all(e0, e1, e2, e4, e5, e6, tau, alpha, rho, time.perf_counter() - started)


def _retrieval(cache, indices: np.ndarray) -> dict:
    scores = {}
    for name in DATASET_ORDER:
        chosen = indices[cache.dataset[indices] == name]
        metrics = _retrieval_block(cache, chosen)
        scores[name] = {"MRR@10": metrics["MRR@10"], "Recall@10": metrics["Recall@10"], "n": int(chosen.size)}
    scores["Average"] = {
        "MRR@10": float(np.mean([scores[name]["MRR@10"] for name in DATASET_ORDER])),
        "Recall@10": float(np.mean([scores[name]["Recall@10"] for name in DATASET_ORDER])),
        "n": int(indices.size),
    }
    return scores


def _retrieval_block(cache, indices: np.ndarray) -> dict:
    cos = cache.cos_scores[indices]
    positive = cache.pos_mask[indices]
    missing = cache.n_positives[indices] - positive.sum(axis=1)
    extra = int(missing.max())
    sim = np.full((indices.size, TOP_K + extra), -2.0, dtype=np.float64)
    sim[:, :TOP_K] = cos
    doc_ids = [f"r{page}" for page in range(TOP_K)] + [f"m{page}" for page in range(extra)]
    positives = []
    for row in range(indices.size):
        ids = [f"r{page}" for page in range(TOP_K) if positive[row, page]]
        ids.extend(f"m{page}" for page in range(int(missing[row])))
        positives.append(ids)
    return compute_retrieval_metrics(sim, positives, doc_ids, k=10)


def _selection(cache, indices: np.ndarray, model, tau: float) -> dict:
    methods = {
        "Weighted tau=1": lambda index, k: predict_weighted(
            cache.answers[index][:k], np.diag(cache.logprob_matrix[index, :k, :k]), cache.cos_scores[index, :k], 1.0
        ),
        "Weighted tau=0.02": lambda index, k: predict_weighted(
            cache.answers[index][:k], np.diag(cache.logprob_matrix[index, :k, :k]), cache.cos_scores[index, :k], 0.02
        ),
        "Bayes": lambda index, k: predict_bayes(
            cache.answers[index][:k], cache.logprob_matrix[index, :k, :k], cache.cos_scores[index, :k], tau
        ),
        "S2 aggregator": lambda index, k: aggregate_select(
            model,
            cache.answers[index][:k],
            np.diag(cache.logprob_matrix[index, :k, :k]),
            cache.cos_scores[index, :k],
            tau=tau,
        )[0],
    }
    return {
        name: {k: _accuracy(cache, indices, lambda index, k=k, fn=fn: fn(index, k)) for k in SELECTION_KS}
        for name, fn in methods.items()
    }


def _tiling(cache, indices: np.ndarray, tau: float, alpha: float) -> dict:
    out = {"Horizontal concat": {}, "Relevance tiling": {}}
    for k in TILING_KS:
        concat_hit = np.zeros(indices.size, dtype=bool)
        tiled_hit = np.zeros(indices.size, dtype=bool)
        uniform_share = np.zeros(indices.size, dtype=np.float64)
        tiled_share = np.zeros(indices.size, dtype=np.float64)
        for row, index in enumerate(indices):
            pages = cache.answers[index][:k]
            scores = cache.cos_scores[index, :k]
            concat_pred = concat_center_answer(pages, cache.query_id[index])
            _, uniform = predict_area(pages, scores, tau, 0.0)
            tiled_pred, tiled = predict_area(pages, scores, tau, alpha)
            gold = [cache.gold[index]]
            concat_hit[row] = relaxed_exact_match(concat_pred, gold) == 1.0
            tiled_hit[row] = relaxed_exact_match(tiled_pred, gold) == 1.0
            uniform_share[row] = uniform
            tiled_share[row] = tiled
        out["Horizontal concat"][k] = {**_by_dataset(cache, indices, concat_hit), "pixel_share": float(uniform_share.mean())}
        out["Relevance tiling"][k] = {**_by_dataset(cache, indices, tiled_hit), "pixel_share": float(tiled_share.mean())}
    return out


def _adaptive(cache, indices: np.ndarray, tau: float, rho: float) -> dict:
    out = {}
    for k in MULTI_KS:
        correct = np.zeros(indices.size, dtype=bool)
        for row, index in enumerate(indices):
            prediction = predict_fluent_vote(cache.answers[index][:k], cache.logprob_matrix[index, :k, :k])
            correct[row] = relaxed_exact_match(prediction, [cache.gold[index]]) == 1.0
        out[f"Fixed k={k}"] = {
            **_by_dataset(cache, indices, correct),
            "mean_k": float(k),
            "latency_ms": modeled_latency_ms(k),
            "passes": float(k),
        }
    correct = np.zeros(indices.size, dtype=bool)
    chosen_k = np.zeros(indices.size, dtype=np.float64)
    for row, index in enumerate(indices):
        prediction, k_star = predict_adaptive_fluent(
            cache.answers[index], cache.logprob_matrix[index], cache.cos_scores[index], tau, rho, TOP_K
        )
        correct[row] = relaxed_exact_match(prediction, [cache.gold[index]]) == 1.0
        chosen_k[row] = k_star
    out["Adaptive k*"] = {
        **_by_dataset(cache, indices, correct),
        "mean_k": float(chosen_k.mean()),
        "latency_ms": float(chosen_k.mean() * MODELED_MS_PER_PAGE),
        "passes": float(chosen_k.mean()),
        "passes_saved_vs_10": float(TOP_K - chosen_k.mean()),
    }
    return out


def _sweeps(cache, indices: np.ndarray, tau: float) -> dict:
    tau_rows = []
    for value in TAU_GRID:
        metrics = compute_ece_and_nll(cache.cos_scores[indices], cache.pos_mask[indices], tau=value)
        correct = 0
        for index in indices:
            prediction = predict_bayes(
                cache.answers[index][:5], cache.logprob_matrix[index, :5, :5], cache.cos_scores[index, :5], value
            )
            correct += relaxed_exact_match(prediction, [cache.gold[index]]) == 1.0
        tau_rows.append({
            "tau": value,
            "ece": metrics["ece"],
            "nll": metrics["nll"],
            "mean_entropy": metrics["mean_entropy"],
            "bayes_accuracy_k5": correct / indices.size,
        })
    alpha_rows = []
    for value in ALPHA_GRID:
        correct = 0
        share = 0.0
        for index in indices:
            prediction, pixel_share = predict_area(cache.answers[index][:6], cache.cos_scores[index, :6], tau, value)
            correct += relaxed_exact_match(prediction, [cache.gold[index]]) == 1.0
            share += pixel_share
        alpha_rows.append({"alpha": value, "accuracy_k6": correct / indices.size, "pixel_share": share / indices.size})
    rho_rows = []
    for value in RHO_GRID:
        correct = 0
        k_sum = 0
        for index in indices:
            prediction, k_star = predict_adaptive_fluent(
                cache.answers[index], cache.logprob_matrix[index], cache.cos_scores[index], tau, value, TOP_K
            )
            correct += relaxed_exact_match(prediction, [cache.gold[index]]) == 1.0
            k_sum += k_star
        rho_rows.append({"rho": value, "accuracy": correct / indices.size, "mean_k": k_sum / indices.size})
    return {"tau": tau_rows, "alpha": alpha_rows, "rho": rho_rows}


def _accuracy(cache, indices: np.ndarray, predict) -> dict:
    correct = np.zeros(indices.size, dtype=bool)
    for row, index in enumerate(indices):
        correct[row] = relaxed_exact_match(predict(index), [cache.gold[index]]) == 1.0
    return _by_dataset(cache, indices, correct)


def _by_dataset(cache, indices: np.ndarray, correct: np.ndarray) -> dict:
    per = {}
    for name in DATASET_ORDER:
        mask = cache.dataset[indices] == name
        per[name] = float(correct[mask].mean())
    per["Average"] = float(np.mean([per[name] for name in DATASET_ORDER]))
    per["micro"] = float(correct.mean())
    return per


def _rows(e0, e1, e2, e4, e5, e6, tau, alpha, rho) -> list[dict]:
    rows = []
    for name, metrics in e0.items():
        for metric in ("MRR@10", "Recall@10"):
            rows.append(_row("E0", name, "VisRAG-Ret cache", 10, metric, metrics[metric]))
    for label, metrics in e1.items():
        for metric in ("ece", "nll", "mean_entropy", "top1_accuracy"):
            rows.append(_row("E1", "all-test", label, 10, metric, metrics[metric]))
    for method, by_k in e2.items():
        for k, scores in by_k.items():
            for name, value in scores.items():
                rows.append(_row("E2", name, method, k, "accuracy", value))
    for method, by_k in e4.items():
        for k, scores in by_k.items():
            for name, value in scores.items():
                metric = "pixel_share" if name == "pixel_share" else "accuracy"
                dataset = "all-test" if name == "pixel_share" else name
                rows.append(_row("E4", dataset, method, k, metric, value))
    for method, scores in e5.items():
        for key in ("Average", "mean_k", "latency_ms", "passes", "passes_saved_vs_10"):
            if key in scores:
                rows.append(_row("E5", "Average" if key == "Average" else "all-test", method, "", key, scores[key]))
    for family, items in e6.items():
        for item in items:
            for key, value in item.items():
                if key != family:
                    rows.append(_row("E6", "dev", family, item[family], key, value))
    rows.append(_row("checkpoint", "dev-fit", "operating point", "", "tau", tau))
    rows.append(_row("checkpoint", "dev-fit", "operating point", "", "alpha", alpha))
    rows.append(_row("checkpoint", "dev-fit", "operating point", "", "rho", rho))
    return rows


def _row(section, dataset, method, k, metric, value) -> dict:
    return {
        "section": section,
        "dataset": dataset,
        "method": method,
        "k": k,
        "metric": metric,
        "value": float(value),
        "source": SOURCE,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex(e0, e2, tau: float) -> str:
    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}"

    lines = [
        "% Computed from the realistic cache by compute_retrieval_metrics and relaxed_exact_match.",
        "% Not a live VisRAG-Ret / MiniCPM-V run and not copied from the paper.",
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lrr}",
        "\\hline",
        "Dataset & MRR@10 & Recall@10 \\\\",
        "\\hline",
    ]
    for name in DATASET_ORDER + ["Average"]:
        lines.append(f"{name} & {pct(e0[name]['MRR@10'])} & {pct(e0[name]['Recall@10'])} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    lines += [
        "\\begin{table}[t]",
        "\\centering",
        f"% Bayes and S2 use the checkpoint tau ({tau:.4f}).",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Method & k=1 & k=2 & k=3 & k=5 \\\\",
        "\\hline",
    ]
    for method, by_k in e2.items():
        cells = " & ".join(pct(by_k[k]["Average"]) for k in SELECTION_KS)
        lines.append(f"{method} & {cells} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _save_figure(path: Path, e1: dict, e4: dict, tau: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for label, metrics in e1.items():
        bins = metrics["bins"]
        occupied = bins["counts"] > 0
        axes[0].plot(bins["confidence"][occupied], bins["accuracy"][occupied], marker="o", label=label)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    axes[0].set_xlabel("Top-1 confidence")
    axes[0].set_ylabel("Accuracy in bin")
    axes[0].set_title("Reliability of p(d | q)")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)

    ks = list(TILING_KS)
    uniform = [e4["Horizontal concat"][k]["pixel_share"] for k in ks]
    tiled = [e4["Relevance tiling"][k]["pixel_share"] for k in ks]
    positions = np.arange(len(ks))
    axes[1].bar(positions - 0.18, uniform, width=0.36, label="Uniform (concat)")
    axes[1].bar(positions + 0.18, tiled, width=0.36, label=f"alpha, tau={tau:.3f}")
    axes[1].set_xticks(positions, [f"k={k}" for k in ks])
    axes[1].set_ylabel("Expected relevant-page pixel share")
    axes[1].set_title("C2 pixel share")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_layout_example(path: Path, cache, test: np.ndarray, tau: float, alpha: float) -> None:
    from PIL import Image

    colors = [(220, 20, 60), (30, 144, 255), (50, 205, 50), (255, 165, 0)]
    pages = [Image.new("RGB", size, color) for size, color in zip([(520, 680), (680, 520), (600, 600), (480, 720)], colors)]
    scores = cache.cos_scores[int(test[0]), :4]
    tiled, _fractions = relevance_proportional_tiling(pages, scores, tau=tau, alpha=alpha, canvas_size=(672, 672))
    concat = baseline_horizontal_concat(pages, total_budget_px=672 * 672)
    concat = concat.resize((672, max(1, round(672 * concat.height / concat.width))))
    canvas = Image.new("RGB", (672, tiled.height + concat.height + 8), (255, 255, 255))
    canvas.paste(tiled, (0, 0))
    canvas.paste(concat, (0, tiled.height + 8))
    canvas.save(path)


def _print_all(e0, e1, e2, e4, e5, e6, tau, alpha, rho, seconds: float) -> None:
    print("Source: realistic cache (seed 0), metrics computed by the project code.")
    print("Not a live VisRAG-Ret or VLM run.")
    _table(
        "E0  Retrieval on the test split",
        ["Dataset", "n", "MRR@10", "Recall@10"],
        [[name, e0[name]["n"], _pct(e0[name]["MRR@10"]), _pct(e0[name]["Recall@10"])] for name in DATASET_ORDER + ["Average"]],
    )
    _table(
        "E1  Calibration on the test split",
        ["tau", "ECE", "NLL", "entropy", "top-1 acc"],
        [[label, f"{metrics['ece']:.4f}", f"{metrics['nll']:.4f}", f"{metrics['mean_entropy']:.4f}", _pct(metrics["top1_accuracy"])] for label, metrics in e1.items()],
    )
    _table(
        f"E2/E3  Selection accuracy, macro average (Bayes and S2 at tau={tau:.4f})",
        ["Method", *[f"k={k}" for k in SELECTION_KS]],
        [[method, *[_pct(by_k[k]["Average"]) for k in SELECTION_KS]] for method, by_k in e2.items()],
    )
    _table(
        "E4  Tiling, macro accuracy and expected relevant-page pixel share",
        ["Method", *[f"k={k} acc" for k in TILING_KS], *[f"k={k} share" for k in TILING_KS]],
        [[
            method,
            *[_pct(by_k[k]["Average"]) for k in TILING_KS],
            *[f"{by_k[k]['pixel_share']:.3f}" for k in TILING_KS],
        ] for method, by_k in e4.items()],
    )
    _table(
        "E5  Multi-image fixed k vs adaptive k* (180 ms/page)",
        ["Method", "accuracy", "mean k", "ms/query", "passes saved vs 10"],
        [[
            method,
            _pct(scores["Average"]),
            f"{scores['mean_k']:.2f}",
            f"{scores['latency_ms']:.0f}",
            f"{scores.get('passes_saved_vs_10', TOP_K - scores['mean_k']):.2f}",
        ] for method, scores in e5.items()],
    )
    best_tau = min(e6["tau"], key=lambda row: row["nll"])
    best_alpha = max(e6["alpha"], key=lambda row: row["accuracy_k6"])
    best_rho = max(e6["rho"], key=lambda row: (row["accuracy"], -row["mean_k"]))
    print("\nE6  Dev sweep")
    print(f"  lowest NLL at tau={best_tau['tau']:g} (NLL {best_tau['nll']:.4f}, Bayes@5 {_pct(best_tau['bayes_accuracy_k5'])})")
    print(f"  best area-vote accuracy at alpha={best_alpha['alpha']:g} ({_pct(best_alpha['accuracy_k6'])}, share {best_alpha['pixel_share']:.3f})")
    print(f"  best adaptive accuracy at rho={best_rho['rho']:.2f} ({_pct(best_rho['accuracy'])}, mean k {best_rho['mean_k']:.2f})")
    print(f"\ncheckpoint operating point: tau={tau:.4f}  alpha={alpha:g}  rho={rho:.2f}")
    print(f"wrote results/headline_results.csv, headline_table.tex, ablation_sweeps.json, calibration_and_tiling.png")
    print(f"eval finished in {seconds:.1f}s")


def _table(title: str, headers: list[str], rows: list[list]) -> None:
    text = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(header), *(len(row[column]) for row in text)) for column, header in enumerate(headers)]
    def format_row(row):
        return "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row))
    print(f"\n{title}")
    print(format_row(headers))
    print(format_row(["-" * width for width in widths]) )
    for row in text:
        print(format_row(row))


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _arguments() -> tuple[dict, Path]:
    parser = argparse.ArgumentParser(description="Evaluate calibrated VisRAG on the cached test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint = _resolve(args.checkpoint or config["paths"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}. Run train.py first.")
    return config, checkpoint


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    main()
