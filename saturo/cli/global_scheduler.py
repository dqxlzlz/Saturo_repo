#!/usr/bin/env python3
"""CLI entrypoint for the Saturo global scheduler."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from saturo.runtime.workload import DECODE, PREFILL, TPUT, repo_root


if hasattr(argparse, "BooleanOptionalAction"):
    BooleanOptionalAction = argparse.BooleanOptionalAction
else:
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            options = []
            for option in option_strings:
                options.append(option)
                if option.startswith("--"):
                    options.append("--no-" + option[2:])
            super().__init__(
                option_strings=options,
                dest=dest,
                nargs=0,
                default=default,
                **kwargs,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest,
                    not str(option_string).startswith("--no-"))


def build_parser(*, include_worker: bool = True) -> argparse.ArgumentParser:
    """Single definition of scheduler and e2e experiment options."""
    parser = argparse.ArgumentParser()
    if include_worker:
        worker = parser.add_argument_group("Single-worker compatibility mode")
        worker.add_argument(
            "--worker",
            action="store_true",
            help="Run a single Saturo worker process controlled by JSON stdin.")
        worker.add_argument("--worker-id", type=int, default=0)
        worker.add_argument("--worker-role",
                            choices=(PREFILL, DECODE, TPUT),
                            default=DECODE)
        worker.add_argument("--tensor-parallel-size", type=int, default=1)
        worker.add_argument("--pipeline-parallel-size", type=int, default=1)
        worker.add_argument("--expert-parallel-size", type=int, default=0)
        worker.add_argument("--p-pool-be-chunk-ms", type=float, default=100.0)

    experiment = parser.add_argument_group('Experiment and output')
    experiment.add_argument("--model", default="")
    experiment.add_argument(
        "--output-dir",
        default="saturo/results/global_scheduler")
    experiment.add_argument("--num-req", type=int, default=None)
    experiment.add_argument("--target-rps", type=float, default=None)
    experiment.add_argument("--max-wall-s", type=float, default=180.0)
    experiment.add_argument("--seed", type=int, default=11)

    placement = parser.add_argument_group('Ray and GPU placement')
    placement.add_argument(
        "--total-gpus",
        type=int,
        default=0,
        help="Total GPUs available for allocation. 0 = auto-detect from Ray.")
    placement.add_argument("--ray-address", default="auto")
    placement.add_argument(
        "--gpu-ids",
        default="",
        help=(
            "Comma-separated local GPU ids used to annotate worker placement, "
            "for example 2,3."
        ),
    )
    placement.add_argument("--dry-run-worker-plan",
                        action=BooleanOptionalAction,
                        default=False,
                        help="Build and print the worker plan without starting workers.")

    workload = parser.add_argument_group('RT and BE workloads')
    workload.add_argument("--rt-workload-csv", default="")
    workload.add_argument("--be-workload-csv", default="")
    workload.add_argument("--kv-handoff", choices=("dummy", "real"), default="dummy",
                          help="P-to-D RT KV path; real uses native V0 shared-storage KV transfer.")
    workload.add_argument("--kv-transfer-dir", default="",
                          help="Real mode: absolute shared directory visible at the same path on all nodes.")
    workload.add_argument("--kv-transfer-timeout-s", type=float, default=30.0)
    workload.add_argument(
        "--be-input-drop-frac", type=float, default=0.0,
        help="Drop the shortest-input fraction of BE requests (0,1).")
    workload.add_argument(
        "--be-output-keep-frac", type=float, default=None,
        help="Keep only the shortest-output fraction of BE requests (0,1]. "
             "Lowers avg_out to raise the optimal P/D balance chunk.")
    workload.add_argument(
        "--be-output-max-len", type=int, default=None,
        help="Keep only BE requests with output_len <= this threshold "
             "(takes precedence over --be-output-keep-frac).")

    profiles = parser.add_argument_group('Profiles and resource allocation')
    profiles.add_argument("--resource-allocation-policy", choices=("saturation_aware", "demand_based"),
                          default="saturation_aware")
    profiles.add_argument("--resource-profile", default="")
    profiles.add_argument("--resource-config-filter", default="")
    profiles.add_argument("--resource-profile-max-candidates",
                        type=int,
                        default=64)
    profiles.add_argument("--resource-memory-watermark", type=float, default=0.9)
    profiles.add_argument("--resource-max-prefill-chunk", type=int, default=8192)
    profiles.add_argument("--resource-prefill-search-granularity",
                        type=int,
                        default=1)
    profiles.add_argument("--resource-decode-warmup-s", type=float, default=0.0)
    profiles.add_argument("--resource-decode-lookahead-window-s",
                        type=float,
                        default=None)
    profiles.add_argument("--resource-decode-model-len-quantile",
                        type=float,
                        default=0.95)
    profiles.add_argument("--resource-decode-min-interval-ms",
                        type=float,
                        default=5.0)
    profiles.add_argument("--resource-decode-chunk-sizes", default="64,128")
    profiles.add_argument("--resource-decode-strategies", default="progressive")
    profiles.add_argument("--resource-tput-chunk-size-candidates",
                        default=",".join(
                            str(x) for x in range(128, 24577, 128)),
                        help=("Total per-iteration token budget candidates "
                              "(max_num_batched_tokens) for T pool. "
                              "Steady-state prefill tokens = chunk_size - "
                              "decode_batch_size."))
    profiles.add_argument("--runtime-profile",
                        default="saturo/results/runtime_profiles",
                        help="Directory containing runtime profiles.")
    profiles.add_argument("--model-len-predictor", default="")

    saturation = parser.add_argument_group('SLO and saturation')
    saturation.add_argument("--enable-prefill-saturation", action=BooleanOptionalAction,
                            default=True, help="Allow BE saturation on P; disabled P workers serve RT only.")
    saturation.add_argument("--rt-ttft-slo-ms", type=float, default=2000.0)
    saturation.add_argument("--rt-tbt-slo-ms", type=float, default=100.0)
    saturation.add_argument(
        "--rt-max-violation-rate",
        type=float,
        default=0.05,
        help="P-pool BE saturation is skipped (set to 0) when the pure-RT "
             "TTFT-SLO violation rate exceeds this fraction. Range [0, 1].")
    saturation.add_argument("--rt-prefill-deadline-s", type=float, default=None,
                            help="Optional tighter local deadline; ingress TTFT remains the upper bound.")
    saturation.add_argument(
        "--enable-decode-saturation",
        action=BooleanOptionalAction,
        default=True,
        help="Enable Saturo's BE-filling decode saturation policy on D-pool "
             "workers. Use --no-enable-decode-saturation for an RT-only "
             "decode baseline.")
    saturation.add_argument(
        "--decode-saturation-lookahead-s",
        type=float,
        default=None,
        help="Future RT horizon used by QRF decode-pool saturation.")
    saturation.add_argument("--p-pool-preemption-margin", type=float, default=0.0)

    updates = parser.add_argument_group('Runtime saturation and allocation updates')
    updates.add_argument("--saturation-update-interval-s", type=float, default=None,
                         help="Shared P/D update interval Delta (default 40s).")
    updates.add_argument("--saturation-update-timeout-s", type=float, default=30.0)
    updates.add_argument("--saturation-history-window-s", type=float, default=None,
                         help="Observed live history window K (default 600s).")
    updates.add_argument("--decode-target-async", action=BooleanOptionalAction,
                         default=True, help="Optimize D targets off the native scheduling path.")
    updates.add_argument("--periodic-reallocation", action=BooleanOptionalAction,
                         default=True)
    updates.add_argument("--resource-reallocation-interval-s", type=float, default=1800.0)
    updates.add_argument("--resource-reallocation-build-timeout-s", type=float, default=300.0)
    updates.add_argument("--resource-reallocation-drain-timeout-s", type=float, default=180.0)
    updates.add_argument("--resource-reallocation-history-window-s", type=float, default=1800.0)
    updates.add_argument("--resource-reallocation-min-completed", type=int, default=8)

    roles = parser.add_argument_group('Adaptive roles')
    roles.add_argument("--adaptive-role-switching",
                        action=BooleanOptionalAction, default=True,
                        help="Borrow T workers for RT bursts and reclaim "
                             "RT-idle P/D workers without changing TP/PP.")
    roles.add_argument("--role-switch-interval-s", type=float, default=60.0)
    roles.add_argument("--role-switch-window-s", type=float, default=60.0)
    roles.add_argument("--role-switch-timeout-s", type=float, default=10.0)

    engine = parser.add_argument_group('Model execution')
    engine.add_argument("--max-model-len", type=int, default=None)
    engine.add_argument("--max-num-seqs", type=int, default=None)
    engine.add_argument("--max-num-batched-tokens", type=int, default=None)
    engine.add_argument("--swap-space", type=float, default=4.0,
                        help="Native vLLM CPU KV swap space, GiB per GPU.")
    engine.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    engine.add_argument("--load-format", default="auto")
    engine.add_argument("--dtype", default="float16")
    engine.add_argument("--disable-inductor",
                        action=BooleanOptionalAction,
                        default=True)

    startup = parser.add_argument_group('Worker startup and warmup')
    startup.add_argument("--worker-start-timeout-s", type=float, default=600.0)
    startup.add_argument("--worker-warmup",
                        action=BooleanOptionalAction,
                        default=True)
    startup.add_argument("--worker-warmup-prompt-len", type=int, default=16)
    startup.add_argument("--worker-warmup-output-len", type=int, default=2)
    startup.add_argument("--worker-warmup-timeout-s", type=float, default=600.0)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.kv_handoff == "real":
        if not args.kv_transfer_dir or not Path(args.kv_transfer_dir).expanduser().is_absolute():
            parser.error("--kv-handoff real requires an absolute --kv-transfer-dir")
        args.kv_transfer_dir = str(Path(args.kv_transfer_dir).expanduser().resolve())
        if args.load_format == "dummy":
            parser.error("--kv-handoff real requires real model weights, not --load-format dummy")
    if not math.isfinite(args.kv_transfer_timeout_s) or args.kv_transfer_timeout_s <= 0:
        parser.error("--kv-transfer-timeout-s must be finite and positive")
    env = dict(os.environ)
    for key, value in getattr(args, "env", {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    if args.saturation_update_interval_s is None:
        args.saturation_update_interval_s = float(env.get("SATURO_HISTORY_REFRESH_S", "40"))
    if args.saturation_history_window_s is None:
        args.saturation_history_window_s = float(env.get("SATURO_RUNTIME_HISTORY_WINDOW_S", "600"))
    if args.decode_saturation_lookahead_s is None:
        args.decode_saturation_lookahead_s = args.saturation_update_interval_s
    if args.resource_decode_lookahead_window_s is None:
        args.resource_decode_lookahead_window_s = args.saturation_update_interval_s
    if args.rt_prefill_deadline_s is None:
        args.rt_prefill_deadline_s = args.rt_ttft_slo_ms / 1000.0
    for name in ("saturation_update_interval_s", "saturation_update_timeout_s",
                 "saturation_history_window_s", "resource_reallocation_interval_s",
                 "resource_reallocation_build_timeout_s", "resource_reallocation_drain_timeout_s",
                 "resource_reallocation_history_window_s", "decode_saturation_lookahead_s",
                 "resource_decode_lookahead_window_s", "rt_prefill_deadline_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error("--" + name.replace("_", "-") + " must be finite and positive")
    if args.resource_reallocation_min_completed < 1:
        parser.error("--resource-reallocation-min-completed must be positive")
    if min(args.role_switch_interval_s, args.role_switch_window_s,
           args.role_switch_timeout_s) <= 0:
        parser.error("role-switch intervals and timeout must be positive")
    if not getattr(args, "worker", False):
        if (args.num_req is None or args.target_rps is None
                or args.num_req <= 0 or args.target_rps <= 0):
            parser.error("--num-req and --target-rps must be positive")
        from saturo.tools.trace_scaler import fixed_num_window_s
        args.window_s = fixed_num_window_s(
            args.num_req, args.target_rps)


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        from saturo.runtime.worker import worker_main
        worker_main(args)
        return
    from saturo.runtime.workload import preflight_workload_manifests
    preflight_workload_manifests(args)
    if args.model_len_predictor:
        os.environ["SATURO_HISTORY_LENGTH_PREDICTOR"] = args.model_len_predictor
    if args.enable_decode_saturation or getattr(args, "adaptive_role_switching", False):
        if args.enable_decode_saturation and "progressive" not in args.resource_decode_strategies.split(","):
            raise ValueError("Enabled decode saturation requires progressive QRF resource profiling")
        from saturo.predictors.qrf import load_required_qrf, validate_qrf_workload
        predictor = load_required_qrf()  # Before resource profiling as well as serving.
        validate_qrf_workload(predictor, args)
    from saturo.runtime.scheduler import SaturoGlobalScheduler
    scheduler = SaturoGlobalScheduler(args)
    if args.dry_run_worker_plan:
        print(json.dumps({
            "worker_plan": [
                {
                    "role": item.role,
                    "host": item.host,
                    "gpu_id": item.gpu_id,
                    "config_name": item.config_name,
                    "tensor_parallel_size": item.tensor_parallel_size,
                    "pipeline_parallel_size": item.pipeline_parallel_size,
                }
                for item in scheduler.worker_plan
            ]
        }, indent=2, sort_keys=True))
        return
    summary = scheduler.run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
