#!/usr/bin/env python3
"""Trigger time_predictor profiling for one model + TP config (no validation).

Building the offline engine with no cached profile makes LLMEngine.__init__ run
the live 3-pass time_predictor profiling and write the fitted profile to disk.
This driver just builds the engine, confirms the predictor is available, and
prints the saved profile path. Use it to batch-profile many model/TP configs.
"""
import argparse
import os

# Ensure live profiling is NOT skipped.
os.environ["SATURO_SKIP_TIME_PREDICTOR_PROFILE"] = "0"

from vllm import LLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--load-format", default="dummy")
    ap.add_argument("--max-model-len", type=int, default=8448)
    args = ap.parse_args()

    print(f"[profile] building engine model={args.model} "
          f"tp={args.tensor_parallel_size} dtype={args.dtype} "
          f"(this triggers time_predictor profiling)", flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        load_format=args.load_format,
        skip_tokenizer_init=True,
    )

    predictor = llm.llm_engine.time_predictor
    if predictor is None or not getattr(predictor, "available", False):
        raise SystemExit("[profile] ERROR: time_predictor unavailable after "
                         "profiling.")
    print(f"[profile] OK  model={args.model} tp={args.tensor_parallel_size}",
          flush=True)
    print(f"[profile] saved profile: {predictor.profile_path}", flush=True)


if __name__ == "__main__":
    main()
