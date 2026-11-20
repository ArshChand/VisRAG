# visrag-calibrated

Course project for DA 6021. The starting point is VisRAG (Yu et al., ICLR 2025): a retriever embeds each document page as an image, and a generator answers from the retrieved pages.

Team Visionaries: Aaryan Bondekar (230101001), Anirudh Bhardwaj (230101010), Arsh Chand (230101017), Bhavya Bharti (230122015), Jinay Mehta (230122021).

## What is added

The retriever is unchanged. Four pieces sit on top of its cosine scores.

- C1. Marginalize generator probabilities over pages that produce the same answer, using the calibrated posterior `p(d|q) = softmax(cosine / tau)`.
- C2. Give each page a canvas area proportional to that posterior, instead of an equal-width strip.
- C3. Keep the shortest prefix of the ranked list whose posterior mass reaches `rho`.
- S2. A small network scores each page from `[posterior, answer confidence, cross-page agreement, rank]` and is trained with binary cross-entropy.

## Evaluation setup: controlled simulation

We study the selectors in a controlled simulation of the VisRAG pipeline. `generate_realistic_visrag_cache()` in `src/retrieval/encode.py` builds `results/cache/` from a seeded random generator, calibrated to the paper's published statistics.

- **Sizes:** the six sets follow VisRAG Table 1: ArxivQA (816), ChartQA (63), MP-DocVQA (591), InfoVQA (718), PlotQA (863) and SlideVQA (556), 3,607 queries in total.
- **Retrieval:** the rank of each positive page is sampled so that per-dataset MRR@10 and Recall@10 match VisRAG-Ret's in-domain numbers (VisRAG Tables 2 and 6). Cosine scores and the 16-dim embeddings are synthetic.
- **Generation:** candidate-answer log-probabilities are drawn from fixed ranges (true pages −0.55 to −0.35, fluent distractors −0.25 to −0.12). Distractor and rescue rates are set by the constants at the top of `encode.py`. `exp(log-probability)` stands in for the reciprocal perplexity in VisRAG equation (4).
- **Latency:** modeled at 180 ms per page (`MODELED_MS_PER_PAGE` in `src/generation/pipeline.py`), not measured.

The simulation isolates each selector's mechanism: it shows how calibration, marginalization, tiling and adaptive k behave when the retriever's scores carry the paper's published level of signal. Running the selectors on real VisRAG-Ret and generator outputs is a natural extension; the model-loading code in `src/generation/pipeline.py` and `src/retrieval/encode.py` is the starting point.

The selectors are scored on the seed-0 20/80 dev/test split.

## Run

From this directory, with PyTorch, NumPy, Pillow, PyYAML, and Matplotlib installed:

```bash
python3 train.py --config configs/baseline.yaml
python3 train.py --config configs/proposed.yaml
python3 eval.py --config configs/proposed.yaml --checkpoint checkpoints/best_calibrated_visrag.pt
```

`configs/baseline.yaml` keeps `tau = 1`. `configs/proposed.yaml` starts at `tau = 0.02` and fits `tau` on the dev split. Alpha stays 0.5 and rho stays 0.90. The checkpoint stores the network and the operating `(tau, alpha, rho)`.

## Outputs

- `results/headline_results.csv` and `results/headline_table.tex`
- `results/ablation_sweeps.json` for the dev sweeps over tau, alpha, and rho
- `results/calibration_and_tiling.png`
- `results/train_log.json`
- `checkpoints/best_calibrated_visrag.pt`
- `report/Visionaries_26_DLCV_Project_Report.tex`

Tests for the selectors:

```bash
python3 -m unittest discover -s tests -q
```
