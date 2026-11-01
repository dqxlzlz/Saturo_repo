#!/usr/bin/env python3
"""Probe profiler-path stability for adjacent small token counts.

Directly drives the same measurement path the time_predictor uses
(profile_qwen2_module_times -> CUDA-event per-layer module timing) for a set of
num_token values, repeating each many times, and prints every repeat so we can
see whether adjacent token counts land close to each other or are jittery.
"""
import argparse
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--load-format", default="dummy")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=8448)
    ap.add_argument("--tokens", default="64,66,68,70,72,74")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--warmup-per-config", default="0",
                    help="comma list of warmup_per_config values to compare")
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--group", default="stateless_linear")
    args = ap.parse_args()

    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        load_format=args.load_format,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )

    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    configs = [("prefill", n, 1) for n in tokens]
    wpc_list = [int(w) for w in str(args.warmup_per_config).split(",") if w.strip()]

    executor = llm.llm_engine.model_executor
    g = args.group

    for wpc in wpc_list:
        res = executor.profile_qwen2_module_times(
            warmup=args.warmup,
            repeat=args.repeat,
            warmup_per_config=wpc,
            configs=configs,
            profile_num_layers=args.num_layers,
            profile_layer_start=0,
            profile_kv_cache_fraction=0.9,
        )
        rows = {int(r["n_tokens"]): r for r in (res or {}).get("rows", [])}

        print(f"\n=== profiler-path probe: model={args.model} "
              f"TP={args.tensor_parallel_size} group={g} "
              f"repeat={args.repeat} warmup={args.warmup} "
              f"warmup_per_config={wpc} num_layers={args.num_layers} ===")
        print(f"{'ntok':>5} | {'mid3avg':>8} {'median':>8} {'mean':>8} {'std':>7} "
              f"{'cv%':>6} {'min':>8} {'max':>8}  (per-layer ms)")
        print("-" * 82)
        for n in tokens:
            r = rows.get(n)
            if r is None:
                print(f"{n:>5} | (no row)")
                continue
            s = r["group_per_layer_ms"].get(g, {})
            each = [v for v in s.get("each", []) if isinstance(v, (int, float))]
            if not each:
                print(f"{n:>5} | (no data)")
                continue
            sv = sorted(each)
            lo = (len(sv) - 3) // 2
            mid3 = sv[lo:lo + 3]
            mid3avg = statistics.mean(mid3)
            med = statistics.median(each)
            mean = statistics.mean(each)
            std = statistics.pstdev(each)
            cv = 100.0 * std / mean if mean else 0.0
            print(f"{n:>5} | {mid3avg:>8.4f} {med:>8.4f} {mean:>8.4f} {std:>7.4f} "
                  f"{cv:>6.2f} {min(each):>8.4f} {max(each):>8.4f}")
            print("      each: " + " ".join(f"{v:.4f}" for v in each))
            print("   sorted: " + " ".join(f"{v:.4f}" for v in sv)
                  + f"   -> mid3={[round(x,4) for x in mid3]} avg={mid3avg:.4f}")


if __name__ == "__main__":
    main()
