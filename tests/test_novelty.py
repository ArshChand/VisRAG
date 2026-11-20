"""Unit tests for the C1, C2, C3, and S2 novelty modules.

Run from the repository root:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.novelty.c1_marginalization import (  # noqa: E402
    answers_equivalent,
    baseline_weighted_selection,
    bayes_marginalized_selection,
    calibrated_weighted_selection,
    retrieval_posterior,
)
from src.novelty.c2_tiling import (  # noqa: E402
    baseline_horizontal_concat,
    compute_tiling_layout,
    relevance_proportional_tiling,
    relevance_tiling_weights,
)
from src.novelty.c3_adaptive_k import adaptive_k_indices, select_adaptive_k  # noqa: E402
from src.novelty.s2_learned_aggregator import (  # noqa: E402
    CalibratedAggregatorNet,
    aggregate_select,
    build_aggregator_features,
)

COLORS = [(220, 20, 60), (30, 144, 255), (50, 205, 50), (255, 165, 0), (148, 0, 211)]


def scores_for_posterior(p, tau):
    """Cosine scores whose softmax at ``tau`` is exactly ``p``."""
    return tau * np.log(np.asarray(p, dtype=np.float64))


def solid(size, color):
    return Image.new("RGB", size, color)


def color_mask(image, color):
    return np.all(np.asarray(image) == np.array(color, dtype=np.uint8), axis=-1)


def roc_auc(labels, scores):
    labels = np.asarray(labels).astype(bool)
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="mergesort")] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class StrictWarnings(unittest.TestCase):
    def setUp(self):
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("error", RuntimeWarning)

    def tearDown(self):
        self._warnings.__exit__(None, None, None)


class TestC1Marginalization(StrictWarnings):
    def test_posterior_is_tempered_softmax(self):
        scores = np.array([0.80, 0.78, 0.60])
        for tau in (1.0, 0.02):
            logits = scores / tau
            expected = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
            np.testing.assert_allclose(retrieval_posterior(scores, tau), expected, rtol=0, atol=1e-15)

    def test_raw_softmax_and_calibrated_posterior_pick_different_pages(self):
        candidates = ["page0", "page1", "page2"]
        gen_logprobs = np.log([0.50, 0.90, 0.95])
        scores = np.array([0.80, 0.78, 0.60])

        lam = np.exp(scores) / np.exp(scores).sum()
        expected_baseline = candidates[int(np.argmax(np.exp(gen_logprobs) * lam))]
        p = retrieval_posterior(scores, 0.02)
        expected_calibrated = candidates[int(np.argmax(np.exp(gen_logprobs) * p))]

        self.assertEqual(expected_baseline, "page1")
        self.assertEqual(expected_calibrated, "page0")
        self.assertEqual(baseline_weighted_selection(candidates, gen_logprobs, scores), expected_baseline)
        self.assertEqual(calibrated_weighted_selection(candidates, gen_logprobs, scores), expected_calibrated)
        self.assertEqual(
            calibrated_weighted_selection(candidates, gen_logprobs, scores, tau=1.0),
            baseline_weighted_selection(candidates, gen_logprobs, scores),
        )

    def test_marginal_matches_formula_and_grouping_merges_mass(self):
        candidates = ["Paris", "Lyon", "lyon."]
        p = np.array([0.50, 0.30, 0.20])
        scores = scores_for_posterior(p, 0.02)
        probs = np.array(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.80, 0.40],
                [0.05, 0.40, 0.80],
            ]
        )
        marginal = probs @ p
        self.assertEqual(int(np.argmax(marginal)), 0)
        self.assertGreater(marginal[1] + marginal[2], marginal[0])

        answer, info = bayes_marginalized_selection(candidates, np.log(probs), scores, group_equivalent=False)
        self.assertEqual(answer, "Paris")
        np.testing.assert_allclose(info["candidate_scores"], marginal, rtol=1e-12)
        np.testing.assert_allclose(info["posterior"], p, rtol=1e-12)

        answer, info = bayes_marginalized_selection(candidates, np.log(probs), scores)
        self.assertEqual(answer, "Lyon")
        self.assertEqual(info["groups"], [[0], [1, 2]])
        self.assertEqual(info["best_group"], 1)
        np.testing.assert_allclose(info["group_scores"], [marginal[0], marginal[1] + marginal[2]], rtol=1e-12)
        np.testing.assert_allclose(info["group_posterior"].sum(), 1.0, rtol=1e-12)

    def test_identical_strings_are_one_event(self):
        candidates = ["42", "42", "Paris"]
        p = np.array([0.4, 0.4, 0.2])
        probs = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.1, 0.1, 0.9]])
        _, info = bayes_marginalized_selection(candidates, np.log(probs), scores_for_posterior(p, 0.02))
        self.assertEqual(info["groups"][0], [0, 1])
        self.assertAlmostEqual(float(info["group_scores"][0]), 0.5, places=12)

    def test_numeric_grouping_and_no_chaining(self):
        self.assertTrue(answers_equivalent("42", "42.0"))
        self.assertTrue(answers_equivalent("100", "104"))
        self.assertTrue(answers_equivalent("104", "108"))
        self.assertFalse(answers_equivalent("100", "108"))
        candidates = ["100", "104", "108"]
        probs = np.log(np.array([[0.6], [0.3], [0.2]]))
        _, info = bayes_marginalized_selection(candidates, probs, np.array([0.5]))
        self.assertEqual(info["groups"], [[0, 1], [2]])

    def test_unscored_pairs_and_validation(self):
        probs = np.array([[np.log(0.9), -np.inf], [-np.inf, np.log(0.8)]])
        answer, info = bayes_marginalized_selection(["a", "b"], probs, scores_for_posterior([0.3, 0.7], 0.02))
        self.assertEqual(answer, "b")
        np.testing.assert_allclose(info["candidate_scores"], [0.27, 0.56], rtol=1e-12)
        with self.assertRaises(ValueError):
            bayes_marginalized_selection(["a"], np.zeros((1, 3)), np.zeros(2))
        with self.assertRaises(ValueError):
            baseline_weighted_selection(["a", "b"], np.array([0.5, -1.0]), np.zeros(2))
        with self.assertRaises(TypeError):
            baseline_weighted_selection("ab", np.zeros(2), np.zeros(2))


class TestC2Tiling(StrictWarnings):
    def test_baseline_concat_equal_width_columns_within_budget(self):
        sizes = [(800, 1000), (600, 900), (1000, 800)]
        pages = [solid(size, color) for size, color in zip(sizes, COLORS)]
        budget = 1344 * 1344
        out = baseline_horizontal_concat(pages, total_budget_px=budget)
        self.assertLessEqual(out.width * out.height, budget)
        self.assertEqual(out.width % 3, 0)
        cell = out.width // 3
        for index, ((w, h), color) in enumerate(zip(sizes, COLORS)):
            column = color_mask(out, color)[:, index * cell : (index + 1) * cell]
            self.assertTrue(column.all(axis=0).sum() == 0 or column.any())
            self.assertEqual(int(column.any(axis=1).sum()), round(cell * h / w))
            self.assertEqual(int(column.any(axis=0).sum()), cell)

    def test_baseline_concat_does_not_upscale_small_pages(self):
        out = baseline_horizontal_concat([solid((100, 100), COLORS[0]), solid((100, 100), COLORS[1])])
        self.assertEqual(out.size, (200, 100))

    def test_weights_budgets_and_cauchy_schwarz(self):
        p = np.array([0.6, 0.25, 0.1, 0.05])
        info = relevance_tiling_weights(scores_for_posterior(p, 0.02), tau=0.02, alpha=1.0, total_budget_px=1000)
        np.testing.assert_allclose(info["weights"], p, rtol=1e-12)
        np.testing.assert_allclose(info["budgets_px"], 1000 * p, rtol=1e-12)
        self.assertAlmostEqual(info["expected_true_page_ratio"], float(np.sum(p**2)), places=12)
        self.assertTrue(info["cauchy_schwarz_holds"])
        self.assertGreater(info["expected_true_page_ratio"], info["uniform_ratio"])

        uniform = relevance_tiling_weights(scores_for_posterior(p, 0.02), alpha=0.0)
        np.testing.assert_allclose(uniform["weights"], np.full(4, 0.25), rtol=1e-12)
        self.assertAlmostEqual(uniform["expected_true_page_ratio"], 0.25, places=12)

        half = relevance_tiling_weights(scores_for_posterior(p, 0.02), alpha=0.5)
        expected_w = np.sqrt(p) / np.sqrt(p).sum()
        np.testing.assert_allclose(half["weights"], expected_w, rtol=1e-12)
        self.assertGreaterEqual(half["expected_true_page_ratio"], 0.25)

        rng = np.random.default_rng(0)
        for _ in range(200):
            scores = rng.normal(0.5, 0.1, size=int(rng.integers(2, 8)))
            for alpha in (0.0, 0.5, 1.0, 2.0):
                ratio = relevance_tiling_weights(scores, alpha=alpha)
                self.assertGreaterEqual(ratio["expected_true_page_ratio"], ratio["uniform_ratio"] - 1e-12)
                self.assertTrue(ratio["cauchy_schwarz_holds"])

    def test_layout_tiles_canvas_exactly_with_sane_aspect_ratios(self):
        canvas = (1344, 1344)
        weights = np.array([0.1, 0.6, 0.05, 0.25])
        rects = compute_tiling_layout(weights, canvas)
        coverage = np.zeros(canvas[::-1], dtype=np.int32)
        for x0, y0, x1, y1 in rects:
            coverage[y0:y1, x0:x1] += 1
        self.assertTrue((coverage == 1).all(), "cells must cover the canvas without gaps or overlaps")

        areas = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects], dtype=np.float64)
        tolerance = canvas[0] + canvas[1]
        np.testing.assert_array_less(np.abs(areas - weights * canvas[0] * canvas[1]), tolerance)
        self.assertEqual(rects[1][:2], (0, 0), "the top page takes the top-left primary region")
        self.assertEqual(int(np.argmax(areas)), 1)
        for x0, y0, x1, y1 in rects:
            w, h = x1 - x0, y1 - y0
            self.assertLessEqual(max(w / h, h / w), 3.0)

    def test_tiling_composite_is_proportional_and_undistorted(self):
        sizes = [(850, 1100), (1100, 850), (900, 900), (600, 1000)]
        pages = [solid(size, color) for size, color in zip(sizes, COLORS)]
        p = np.array([0.55, 0.25, 0.15, 0.05])
        scores = scores_for_posterior(p, 0.02)
        composite, fractions = relevance_proportional_tiling(pages, scores, tau=0.02, alpha=1.0)

        self.assertEqual(composite.size, (1344, 1344))
        self.assertAlmostEqual(sum(fractions), 1.0, places=12)
        np.testing.assert_allclose(fractions, p, atol=0.01)
        self.assertEqual(list(np.argsort(-np.array(fractions))), [0, 1, 2, 3])

        pixel_share = []
        for (w, h), color in zip(sizes, COLORS):
            mask = color_mask(composite, color)
            ys, xs = np.nonzero(mask)
            box_w, box_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            self.assertEqual(int(mask.sum()), int(box_w * box_h), "each page is one solid rectangle")
            self.assertAlmostEqual((box_w / box_h) / (w / h), 1.0, delta=0.02)
            pixel_share.append(mask.sum() / (1344 * 1344))
        self.assertEqual(int(np.argmax(pixel_share)), 0)
        self.assertGreater(pixel_share[0], 1 / len(pages))

    def test_tiling_alpha_zero_is_uniform_and_lengths_must_match(self):
        pages = [solid((500, 500), color) for color in COLORS[:4]]
        _, fractions = relevance_proportional_tiling(pages, np.array([0.9, 0.5, 0.4, 0.1]), alpha=0.0)
        np.testing.assert_allclose(fractions, np.full(4, 0.25), atol=1e-3)
        with self.assertRaises(ValueError):
            relevance_proportional_tiling(pages, np.zeros(3))


class TestC3AdaptiveK(StrictWarnings):
    def test_prefix_mass_rule(self):
        p = np.array([0.5, 0.3, 0.15, 0.05])
        scores = scores_for_posterior(p, 0.02)
        k, kept = select_adaptive_k(scores, tau=0.02, rho=0.90, k_max=6)
        self.assertEqual(k, 3)
        np.testing.assert_allclose(kept, p[:3] / p[:3].sum(), rtol=1e-12)
        self.assertEqual(select_adaptive_k(scores, rho=0.80)[0], 2)
        self.assertEqual(select_adaptive_k(scores, rho=0.50)[0], 1)
        self.assertEqual(select_adaptive_k(scores, rho=1.00)[0], 4)
        self.assertEqual(select_adaptive_k(scores, rho=0.99, k_max=2)[0], 2)

    def test_sharp_posterior_gives_one_flat_gives_k_max(self):
        scores = np.array([0.90, 0.80, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45])
        k_sharp, kept = select_adaptive_k(scores, tau=0.02)
        self.assertEqual(k_sharp, 1)
        np.testing.assert_allclose(kept, [1.0])
        self.assertEqual(select_adaptive_k(scores, tau=1.0, k_max=6)[0], 6)

    def test_unsorted_scores_and_raw_prefix(self):
        p = np.array([0.15, 0.5, 0.05, 0.3])
        indices, kept = adaptive_k_indices(scores_for_posterior(p, 0.02), rho=0.9, renormalize=False)
        self.assertEqual(indices.tolist(), [1, 3, 0])
        np.testing.assert_allclose(kept, [0.5, 0.3, 0.15], rtol=1e-12)
        for bad in ({"rho": 0.0}, {"rho": 1.5}, {"k_max": 0}):
            with self.assertRaises(ValueError):
                select_adaptive_k(np.zeros(3), **bad)


class TestS2LearnedAggregator(StrictWarnings):
    def test_features_by_hand(self):
        p = np.array([0.6, 0.3, 0.1])
        features = build_aggregator_features(
            ["Paris", "paris", "Lyon"], np.log([0.9, 0.6, 0.7]), scores_for_posterior(p, 0.02)
        )
        expected = np.array(
            [
                [0.6, 0.9, 0.3, 0.0],
                [0.3, 0.6, 0.6, 0.5],
                [0.1, 0.7, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(features, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(build_aggregator_features(["x"], np.log([0.5]), np.array([0.3]))[0, 3], 0.0)

    def _draw_features(self, n, seed):
        rng = np.random.default_rng(seed)
        features = rng.uniform(0.0, 1.0, size=(n, 4))
        logit = 6.0 * (0.7 * features[:, 0] + 0.5 * features[:, 2] + 0.3 * features[:, 1] - 0.3 * features[:, 3] - 0.45)
        labels = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float64)
        return features, labels

    def test_training_learns_and_checkpoint_round_trips(self):
        train_x, train_y = self._draw_features(3000, seed=1)
        test_x, test_y = self._draw_features(1000, seed=2)
        for hidden_dim in (0, 16):
            model = CalibratedAggregatorNet(hidden_dim=hidden_dim)
            stats = model.fit(train_x, train_y, epochs=400, lr=0.03)
            self.assertLess(stats["loss_after"], stats["loss_before"] - 0.05)
            auc = roc_auc(test_y, model.predict_proba(test_x))
            self.assertGreater(auc, 0.80, f"hidden_dim={hidden_dim} held-out AUC {auc:.3f}")

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "calibrated_model.pt"
                model.save_checkpoint(path, tau=0.031, alpha=0.8, rho=0.9, dev_auc=float(auc))
                restored, hparams = CalibratedAggregatorNet.load_checkpoint(path)
            np.testing.assert_allclose(restored.predict_proba(test_x), model.predict_proba(test_x), rtol=0, atol=0)
            self.assertEqual({k: hparams[k] for k in ("tau", "alpha", "rho")}, {"tau": 0.031, "alpha": 0.8, "rho": 0.9})

    def test_fit_is_reproducible_and_aggregate_select_uses_argmax(self):
        x, y = self._draw_features(500, seed=3)
        a, b = CalibratedAggregatorNet(), CalibratedAggregatorNet()
        a.fit(x, y, epochs=50, seed=7)
        b.fit(x, y, epochs=50, seed=7)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

        candidates = ["Paris", "paris", "Lyon"]
        logprobs = np.log([0.9, 0.6, 0.7])
        scores = scores_for_posterior([0.6, 0.3, 0.1], 0.02)
        answer, probs = aggregate_select(a, candidates, logprobs, scores)
        self.assertEqual(answer, candidates[int(np.argmax(probs))])
        np.testing.assert_allclose(probs, a.predict_proba(build_aggregator_features(candidates, logprobs, scores)))

    def test_validation(self):
        model = CalibratedAggregatorNet()
        with self.assertRaises(ValueError):
            model.fit(np.zeros((4, 4)), np.array([0, 1, 2, 0]))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            CalibratedAggregatorNet(hidden_dim=-1)


if __name__ == "__main__":
    unittest.main()
