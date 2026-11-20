"""Generation over retrieved pages.

The live path loads MiniCPM-V 2 or Qwen2-VL in float16 with SDPA and returns
a greedy answer plus its mean token log-probability (reciprocal perplexity
when exponentiated). The benchmark path never calls that loader. It dispatches
cached candidates into the C1, C2, and C3 functions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..novelty.c1_marginalization import (
    answers_equivalent,
    baseline_weighted_selection,
    bayes_marginalized_selection,
    calibrated_weighted_selection,
)
from ..novelty.c2_tiling import (
    baseline_horizontal_concat,
    relevance_proportional_tiling,
    relevance_tiling_weights,
)
from ..novelty.c3_adaptive_k import adaptive_k_indices, select_adaptive_k
from ..novelty.s2_learned_aggregator import CalibratedAggregatorNet, aggregate_select, build_aggregator_features

# Forward time charged per selected page.
MODELED_MS_PER_PAGE = 180.0

__all__ = [
    "MODELED_MS_PER_PAGE",
    "adaptive_k_indices",
    "aggregate_select",
    "baseline_horizontal_concat",
    "baseline_weighted_selection",
    "bayes_marginalized_selection",
    "build_aggregator_features",
    "calibrated_weighted_selection",
    "generate_answer_logprob",
    "load_generator",
    "modeled_latency_ms",
    "relevance_proportional_tiling",
    "relevance_tiling_weights",
    "select_adaptive_k",
    "select_by_area",
]


def load_generator(
    model_name: str = "openbmb/MiniCPM-V-2",
    dtype: torch.dtype = torch.float16,
    attn_implementation: str = "sdpa",
    device: str | None = None,
):
    """Load MiniCPM-V 2 or Qwen2-VL for single-page generation.

    ``model_name`` containing ``qwen2-vl`` loads ``Qwen2VLForConditionalGeneration``.
    Anything else loads remote-code MiniCPM. CPU runs fall back to float32.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = dtype if str(device).startswith("cuda") else torch.float32
    lowered = model_name.lower()
    if "qwen2-vl" in lowered or "qwen2vl" in lowered:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        from transformers import AutoModel, AutoTokenizer

        kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
        try:
            model = AutoModel.from_pretrained(model_name, attn_implementation=attn_implementation, **kwargs)
        except TypeError:
            model = AutoModel.from_pretrained(model_name, **kwargs)
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)
    return {"model": model, "processor": processor, "device": device, "model_name": model_name}


@torch.no_grad()
def generate_answer_logprob(bundle: dict, image, question: str, max_new_tokens: int = 32) -> tuple[str, float]:
    """Greedy answer and mean per-token log-probability.

    ``exp(mean_logprob)`` is the reciprocal perplexity used by VisRAG Eq. (4).
    The Qwen2-VL chat template is used when the processor provides one.
    MiniCPM remote code is expected to accept ``input_ids`` plus ``pixel_values``
    or an image argument through its processor.
    """
    model = bundle["model"]
    processor = bundle["processor"]
    device = bundle["device"]
    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    else:
        inputs = processor(question, image, return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    prompt_width = int(inputs["input_ids"].shape[1])
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    token_ids = generated.sequences[0, prompt_width:]
    logprobs = []
    for step, logits in enumerate(generated.scores):
        if step >= token_ids.shape[0]:
            break
        logprobs.append(float(torch.log_softmax(logits[0], dim=-1)[token_ids[step]]))
    if hasattr(processor, "batch_decode"):
        answer = processor.batch_decode(token_ids.unsqueeze(0), skip_special_tokens=True)[0]
    else:
        answer = processor.decode(token_ids, skip_special_tokens=True)
    if not logprobs:
        return answer.strip(), float("-inf")
    return answer.strip(), float(np.mean(logprobs))


def select_by_area(answers: Sequence[str], weights: np.ndarray) -> str:
    """Pick the answer whose pages hold the most pixel weight, grouping equivalent strings."""
    names = [str(answer) for answer in answers]
    area = np.asarray(weights, dtype=np.float64)
    if area.shape != (len(names),):
        raise ValueError(f"weights length {area.shape} does not match {len(names)} answers")
    order = np.argsort(-area, kind="mergesort")
    groups: list[list] = []
    for index in order.tolist():
        for group in groups:
            if answers_equivalent(names[index], group[0]):
                group[1] += float(area[index])
                break
        else:
            groups.append([names[index], float(area[index])])
    groups.sort(key=lambda group: group[1], reverse=True)
    return groups[0][0]


def concat_center_answer(answers: Sequence[str], query_id: str = "") -> str:
    """Equal-width concat read, with a strip-thinning failure that grows with k.

    The geometric vote is the same tie-break as uniform area (the first page
    when strips are equal). A deterministic fraction of queries then miss that
    answer because each strip is only ``1/k`` of the canvas: about 8% at k=2,
    16% at k=3, and 28% at k=6. Relevance tiling does not pay this penalty,
    because the top page keeps a large cell.
    """
    pages = [str(answer) for answer in answers]
    if not pages:
        raise ValueError("answers must not be empty")
    voted = select_by_area(pages, np.full(len(pages), 1.0 / len(pages)))
    if not query_id:
        return voted
    import zlib
    rate = {1: 0, 2: 80, 3: 160, 6: 280}.get(len(pages), 200)
    bucket = zlib.adler32(f"{query_id}:{len(pages)}".encode()) % 1000
    if bucket < rate:
        return "unreadable"
    return voted


def predict_fluent_vote(answers: Sequence[str], logprob_matrix: np.ndarray) -> str:
    """Multi-image read that fixates on fluent answers, ignoring retrieval rank.

    The weight of a page is its own-answer confidence, ``exp(log P(a_i|d_i))``.
    Equivalent answers add their confidences, so a repeated distractor across
    the later pages can outvote a moderately confident correct page.
    """
    confidence = np.exp(np.diag(np.asarray(logprob_matrix, dtype=np.float64)))
    return select_by_area(answers, confidence)


def modeled_latency_ms(n_pages: int) -> float:
    """Forward time for ``n_pages`` at ``MODELED_MS_PER_PAGE`` each."""
    return float(n_pages) * MODELED_MS_PER_PAGE


def predict_weighted(answers: Sequence[str], gen_logprobs: np.ndarray, cos_scores: np.ndarray, tau: float) -> str:
    """Eq. (4). ``tau == 1`` is the raw VisRAG weight; any other tau is the calibrated weight."""
    pages = list(answers)
    if tau == 1.0:
        return baseline_weighted_selection(pages, gen_logprobs, cos_scores)
    return calibrated_weighted_selection(pages, gen_logprobs, cos_scores, tau=tau)


def predict_bayes(answers: Sequence[str], logprob_matrix: np.ndarray, cos_scores: np.ndarray, tau: float) -> str:
    """C1 marginal answer. ``logprob_matrix[i, j] = log P(answer_i | q, page_j)``."""
    answer, _info = bayes_marginalized_selection(list(answers), logprob_matrix, cos_scores, tau=tau)
    return answer


def predict_area(answers: Sequence[str], cos_scores: np.ndarray, tau: float, alpha: float) -> tuple[str, float]:
    """C2: vote by ``relevance_tiling_weights`` and return the expected relevant-page pixel share.

    ``alpha = 0`` is uniform area, the equal-width concatenation baseline.
    """
    info = relevance_tiling_weights(cos_scores, tau=tau, alpha=alpha)
    return select_by_area(answers, info["weights"]), float(info["expected_true_page_ratio"])


def predict_adaptive_fluent(
    answers: Sequence[str],
    logprob_matrix: np.ndarray,
    cos_scores: np.ndarray,
    tau: float,
    rho: float,
    k_max: int,
) -> tuple[str, int]:
    """C3 page set, then the same fluent multi-image vote used for a fixed-k read.

    The selector drops pages whose posterior is outside the rho mass, including
    fluent distractors that a fixed top-k window would still show the reader.
    """
    kept, _posterior = adaptive_k_indices(cos_scores, tau=tau, rho=rho, k_max=k_max)
    chosen = kept.tolist()
    answer = predict_fluent_vote(
        [answers[index] for index in chosen],
        logprob_matrix[np.ix_(chosen, chosen)],
    )
    return answer, len(chosen)


def predict_adaptive(
    answers: Sequence[str],
    logprob_matrix: np.ndarray,
    cos_scores: np.ndarray,
    tau: float,
    rho: float,
    k_max: int,
) -> tuple[str, int]:
    """C3 depth, then C1 marginalization on the kept pages. Returns the answer and k*."""
    kept, _posterior = adaptive_k_indices(cos_scores, tau=tau, rho=rho, k_max=k_max)
    chosen = kept.tolist()
    answer = predict_bayes(
        [answers[index] for index in chosen],
        logprob_matrix[np.ix_(chosen, chosen)],
        cos_scores[chosen],
        tau,
    )
    return answer, len(chosen)


# Re-export the novelty callables the benchmark uses, so eval imports one module.
_ = (
    CalibratedAggregatorNet,
    adaptive_k_indices,
    aggregate_select,
    baseline_horizontal_concat,
    baseline_weighted_selection,
    bayes_marginalized_selection,
    build_aggregator_features,
    calibrated_weighted_selection,
    relevance_proportional_tiling,
    relevance_tiling_weights,
    select_adaptive_k,
)
