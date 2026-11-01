#!/usr/bin/env python3
"""Plot TP4 linear prediction, query-token correction, and final step time."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from saturo.predictors.time_predictor import TimePredictor


PROFILE = (
    ROOT
    / "saturo/results/runtime_profiles"
    / "root-huggingface_config-mistral-large-instruct-2411"
    / (
        "time_predictor_root-huggingface_config-mistral-large-instruct-2411_"
        "gpu-nvidia-gb200_tp4_pp1_torch.float16_382cc6c044daa40b.json"
    )
)
RESULTS = ROOT / "experiments/saturo/results/plain_vllm_decode"
SOURCES = (
    (
        RESULTS
        / "tp4_prompt1157_out100_batch8_1024_step8_20260731/summary.csv",
        "measured step p50, context≈1206.5",
        "#9467bd",
    ),
    (
        RESULTS
        / "tp4_prompt512_out100_batch1024_2048_step16_20260731/summary.csv",
        "measured step p50, context≈561.5",
        "#8c564b",
    ),
)
OUTPUT = RESULTS / "tp4_corrected_linear_and_forward_prediction.png"


def correction_ms(predictor: TimePredictor, tokens: int) -> float:
    return predictor.runtime_overhead.predict_ms(
        prefill_request_count=0,
        decode_request_count=0,
        prefill_tokens=tokens,
        decode_tokens=0,
    )


def main() -> None:
    predictor = TimePredictor(str(PROFILE))
    tokens = list(range(1, 2049))
    linear = [
        predictor.predict_linear_ms(n) * predictor.num_layers
        for n in tokens
    ]
    correction = [correction_ms(predictor, n) for n in tokens]
    corrected_linear = [
        raw + extra for raw, extra in zip(linear, correction)
    ]

    figure, (components, final) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15]},
    )
    components.plot(tokens, linear, linestyle="--", linewidth=1.7,
                    label="raw predicted linear")
    components.plot(tokens, correction, linestyle=":", linewidth=2.0,
                    label="total-query-token correction")
    components.plot(tokens, corrected_linear, linewidth=2.5,
                    label="linear + correction")
    components.set_ylabel("Time (ms)")
    components.set_title(
        "TP4 linear component after total-query-token correction")
    components.grid(alpha=0.25)
    components.legend()

    contexts = (561.5, 1206.5)
    colors = ("#1f77b4", "#2ca02c")
    for context, color in zip(contexts, colors):
        predicted = [
            predictor.predict_forward_from_aggregates_ms(
                prefill_tokens=0,
                prefill_sum_ctx=0,
                decode_tokens=n,
                decode_sum_ctx=n * context,
                prefill_request_count=0,
                decode_request_count=n,
            )
            for n in tokens
        ]
        final.plot(
            tokens,
            predicted,
            color=color,
            linewidth=2.1,
            label=f"final predictor, decode context={context:.1f}",
        )

    for path, label, color in SOURCES:
        x_values = []
        y_values = []
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                batch = int(row["batch_size"])
                if batch < 24:
                    continue
                x_values.append(batch)
                y_values.append(float(row["step_wall_time_ms_p50"]))
        final.scatter(
            x_values,
            y_values,
            s=16,
            alpha=0.55,
            color=color,
            label=label,
        )

    final.set_xlabel("Total query tokens in one iteration")
    final.set_ylabel("Engine step time (ms)")
    final.set_title("Final predictor (linear + attention + correction)")
    final.grid(alpha=0.25)
    final.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
