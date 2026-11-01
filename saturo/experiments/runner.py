"""Build reproducible per-figure cases and launch explicitly selected backends."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from saturo.experiments.catalog import BACKENDS, METHODS, NATIVE_PAPER_GAPS, STUDIES

ROOT = Path(__file__).resolve().parents[2]


def repo_path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_config(path, mode):
    payload = json.loads(repo_path(path).read_text())
    if "base" not in payload:
        if mode != "smoke":
            raise ValueError("A flat e2e config is a smoke input; use --mode smoke")
        return {"base": payload, "models": {"smoke": {}},
                "workloads": {"smoke": {}}, "_flat": True,
                "duration_s": payload["num_req"] / payload["target_rps"]}
    if not isinstance(payload.get("base"), dict):
        raise ValueError("base must be an object")
    for backend, command in payload.get("backend_commands", {}).items():
        if not isinstance(command, list) or not command or not all(isinstance(t, str) for t in command):
            raise ValueError(f"backend_commands.{backend} must be a nonempty argv list")
    return payload


def _id(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "_")


def generate_cases(study, config, *, mode="paper", rps=None, methods=None,
                   models=None, workloads=None, values=None):
    if study not in STUDIES:
        raise ValueError("Unknown study: " + study)
    spec = copy.deepcopy(STUDIES[study])
    if values is not None:
        if "values" not in spec:
            raise ValueError("This study has no scalar --values grid; use --rps or edit the configuration")
        if study == "fig14_quantile" and any(v > 1 for v in values):
            raise ValueError("QRF quantiles must be in (0, 1]")
        if study in ("fig10_cluster_size", "fig13_allocator_overhead", "fig13_d_overhead"):
            if any(v != int(v) for v in values):
                raise ValueError("GPU and batch counts must be integers")
            values = [int(v) for v in values]
        spec["values"] = values
    c = config
    if rps and len(rps) > 1 and study not in ("fig6_e2e", "fig11_saturation", "fig12_allocation"):
        raise ValueError("This study accepts one --rps value; use separate invocations for additional rates")
    flat = c.get("_flat", False)
    model_keys = models or (["smoke"] if flat else
                           ["llama", "mistral"] if study == "fig6_e2e" else ["mistral"])
    workload_keys = workloads or (["smoke"] if flat else
                                  ["conv", "code"] if study in ("fig5_workload_cdf", "fig6_e2e")
                                  else ["conv"])
    methods = methods or METHODS
    for label, selected, available in (
            ("model", model_keys, c.get("models", {})),
            ("workload", workload_keys, c.get("workloads", {}))):
        unknown = set(selected) - set(available)
        if unknown:
            raise ValueError(f"Unknown {label}: {sorted(unknown)}")
    unknown = set(methods) - set(METHODS) - set(c.get("backend_commands", {}))
    if unknown:
        raise ValueError(f"Unknown method: {sorted(unknown)}")
    cases = []

    def add(model, workload, variant, backend, *, rate=None, overrides=None, params=None):
        values = copy.deepcopy(c["base"])
        def merge(layer):
            env = dict(values.get("env", {}))
            env.update(layer.get("env", {}))
            values.update(copy.deepcopy(layer))
            if env:
                values["env"] = env
        merge(c.get("models", {}).get(model, {}))
        merge(c.get("workloads", {}).get(workload, {}))
        history = values.pop("history_csv", c.get("history_csv", ""))
        pool_profile = values.pop("pool_runtime_profile", None)
        if rate is not None:
            values["target_rps"] = rate
            values["num_req"] = max(1, round(rate * c.get("duration_s", 400)))
        if overrides:
            merge(overrides)
        case_id = "_".join(map(_id, [study, model, workload, variant]
                              + ([rate] if rate is not None else [])))
        for key, value in values.get("env", {}).items():
            if isinstance(value, str):
                for name, replacement in {"model": model, "workload": workload, "case": case_id}.items():
                    value = value.replace("{" + name + "}", replacement)
                values["env"][key] = value
        params = dict(params or {})
        params["pool"] = copy.deepcopy(c.get("pool", {}))
        substitutions = {"model": model, "workload": workload, "case": case_id}
        def expand(value):
            for name, replacement in substitutions.items():
                value = value.replace("{" + name + "}", replacement)
            return value
        for key in ("rt_workload_csv", "be_workload_csv", "runtime_profile", "resource_profile", "model_len_predictor"):
            if isinstance(values.get(key), str):
                values[key] = expand(values[key])
        if history:
            params["history_csv"] = expand(history)
        if pool_profile:
            params["pool_runtime_profile"] = expand(pool_profile)
        cases.append({
            "id": case_id, "study": study, "figure": spec["figure"],
            "model": model, "workload": workload, "variant": variant,
            "backend": backend, "mode": mode, "config": values,
            "study_parameters": params, "grid_note": spec.get("grid_note"),
        })

    if study == "fig6_e2e":
        for model, workload in itertools.product(model_keys, workload_keys):
            rates = rps or ([c["base"]["target_rps"]] if flat else
                            list(range(12, 45, 4)) if model == "llama" else list(range(4, 29, 4)))
            for rate, method in itertools.product(rates, methods):
                add(model, workload, method, method, rate=rate)
    elif study in ("fig8_timeline", "fig9_slo_scale", "fig10_cluster_size"):
        for model, workload, method in itertools.product(model_keys, workload_keys, methods):
            values = spec.get("values", [None])
            for value in values:
                rate = (rps or [20 if study == "fig8_timeline" else
                                16 if study == "fig9_slo_scale" else 12])[0]
                changes, params = {}, {}
                if study == "fig9_slo_scale":
                    changes = {"rt_ttft_slo_ms": 1000 * value,
                               "rt_tbt_slo_ms": 150 * value}
                    params["slo_scale"] = value
                elif study == "fig10_cluster_size":
                    changes["total_gpus"] = value
                    # Local placement labels must not masquerade as multi-node placement.
                    changes["gpu_ids"] = ""
                    params["gpu_budget"] = value
                add(model, workload, f"{method}_{value}", method,
                    rate=rate, overrides=changes, params=params)
    elif study == "fig11_saturation":
        rates = rps or sorted(set(spec["panel_a_rps"] + spec["panel_b_rps"]))
        for model, workload, rate in itertools.product(model_keys, workload_keys, rates):
            for variant, (p, d) in spec["variants"].items():
                add(model, workload, variant, "saturo", rate=rate,
                    overrides={"enable_prefill_saturation": p,
                               "enable_decode_saturation": d,
                               "adaptive_role_switching": False,
                               "periodic_reallocation": False},
                    params={"panel_a": rate in spec["panel_a_rps"],
                            "panel_b": rate in spec["panel_b_rps"],
                            "roles_fixed_for_controlled_ablation": True})
    elif study == "fig12_allocation":
        for model, workload, rate, policy in itertools.product(
                model_keys, workload_keys, rps or spec["values"],
                ["saturation_aware", "demand_based"]):
            add(model, workload, policy, "saturo", rate=rate,
                overrides={"resource_allocation_policy": policy,
                           "enable_prefill_saturation": True,
                           "enable_decode_saturation": True,
                           "adaptive_role_switching": False,
                           "periodic_reallocation": False})
    elif study in ("fig13_p_overhead", "fig13_d_overhead", "fig13_allocator_overhead"):
        backend = {"fig13_p_overhead": "p_overhead",
                   "fig13_d_overhead": "d_sync_overhead",
                   "fig13_allocator_overhead": "allocator_overhead"}[study]
        parameter = {"fig13_p_overhead": "history_window_s",
                     "fig13_d_overhead": "batch_size",
                     "fig13_allocator_overhead": "gpu_budget"}[study]
        for value in spec["values"]:
            add(model_keys[0], workload_keys[0], str(value), backend,
                params={parameter: value})
            if study == "fig13_d_overhead":
                add(model_keys[0], workload_keys[0], f"async_{value}", "d_async_overhead",
                    params={parameter: value})
    elif study in ("fig14_history_window", "fig14_quantile", "fig14_update_interval"):
        for value in spec["values"]:
            backend = {"fig14_history_window": "p_history",
                       "fig14_quantile": "d_quantile",
                       "fig14_update_interval": "update_interval"}[study]
            parameter = {"fig14_history_window": "history_window_s",
                         "fig14_quantile": "quantile",
                         "fig14_update_interval": "update_interval_s"}[study]
            changes = {}
            if study == "fig14_quantile":
                changes["resource_decode_model_len_quantile"] = value
                changes["env"] = {"VLLM_DECODE_SATURATION_MODEL_LEN_ETA": str(value)}
            elif study == "fig14_update_interval":
                changes["saturation_update_interval_s"] = value
                changes["decode_target_async"] = True
                # The interval is an actual P/D publication cadence. The
                # paper also uses this Delta as the future prediction window.
                changes["decode_saturation_lookahead_s"] = value
                changes["resource_decode_lookahead_window_s"] = value
            add(model_keys[0], workload_keys[0], str(value), backend,
                rate=(rps or [c["base"].get("target_rps", 20)])[0],
                overrides=changes, params={parameter: value,
                    "pool_rps_not_specified_in_paper": study != "fig14_update_interval"})
    else:
        for backend in spec["backends"]:
            for model, workload in itertools.product(model_keys[:1], workload_keys):
                duration = {"fig1_fragmentation": 200, "fig2_colocation": 100}.get(study)
                overrides = ({"num_req": round(c["base"].get("target_rps", 20) * duration)}
                             if duration else None)
                add(model, workload, backend, backend, overrides=overrides,
                    params={"measurement_duration_s": duration} if duration else None)
    return cases


def blockers(case, config):
    backend = case["backend"]
    if backend in config.get("backend_commands", {}):
        return []  # User-supplied adapters must still satisfy result contracts.
    if backend in ("saturo", "update_interval"):
        return list(NATIVE_PAPER_GAPS) if case["mode"] == "paper" else []
    if backend in ("p_overhead", "d_sync_overhead", "allocator_overhead",
                   "workload_stats", "p_history", "d_quantile"):
        return []
    return [BACKENDS.get(backend, {}).get("gap", "No configured execution backend")]


def prepare_case(case, config, output_root):
    case = copy.deepcopy(case)
    directory = repo_path(output_root) / case["id"]
    case["directory"] = str(directory)
    values = case["config"]
    values["output_dir"] = str(directory)
    # Counterfactual SLO/workload cases must not silently share old mu/phi tables.
    if (case["backend"] in (*METHODS, "update_interval") and not config.get("_flat")
            and not config.get("reuse_allocation_profiles", False)):
        table = (repo_path(output_root) / "_profiles" / _id(case["model"])
                 / _id(case["workload"]) / _id(values.get("target_rps", "base"))
                 / f"slo_{_id(values.get('rt_ttft_slo_ms'))}_{_id(values.get('rt_tbt_slo_ms'))}"
                 / "allocation_profiles.json")
        if case["backend"] == "update_interval":
            table = table.parent / ("delta_" + _id(values["saturation_update_interval_s"])) / table.name
        values["resource_profile"] = str(table)
    native = [sys.executable, str(ROOT / "saturo/run_saturo_endtoend_experiment.py"),
              "--config", str(directory / "config.json")]
    backend = case["backend"]
    if backend in ("saturo", "update_interval"):
        command = native
    elif backend in ("p_overhead", "d_sync_overhead", "allocator_overhead"):
        kind = {"p_overhead": "p", "d_sync_overhead": "d",
                "allocator_overhead": "allocator"}[backend]
        command = [sys.executable, str(ROOT / "saturo/experiments/overhead/measure.py"),
                   "--kind", kind, "--case", str(directory / "case.json"),
                   "--repeats", str(config.get("measurement_repeats", 30))]
    elif backend == "workload_stats":
        command = [sys.executable, str(ROOT / "saturo/experiments/workloads/inspect_workloads.py"),
                   "--case", str(directory / "case.json")]
    elif backend in ("p_history", "d_quantile"):
        entry = "run_history_window.py" if backend == "p_history" else "run_quantile.py"
        command = [sys.executable, str(ROOT / "saturo/experiments/pool_validation" / entry),
                   "--case", str(directory / "case.json"), "--execute"]
    else:
        command = None
    if backend in config.get("backend_commands", {}):
        substitutions = {"python": sys.executable, "repo": str(ROOT),
                         "config": str(directory / "config.json"),
                         "case": str(directory / "case.json"), "output_dir": str(directory)}
        command = [token.format(**substitutions)
                   for token in config["backend_commands"][backend]]
    case["external_backend"] = backend in config.get("backend_commands", {})
    case["command"] = command
    case["blockers"] = blockers(case, config)
    case["result_kind"] = ("cpu_measurement" if "overhead" in backend else
                           "dataset_statistics" if backend == "workload_stats" else
                           "smoke_runtime" if case["mode"] == "smoke" else
                           "requires_paper_result_validation")
    return case


def validate_execution_inputs(case):
    if case.get("external_backend"):
        return
    backend, values = case["backend"], case["config"]
    required = ["rt_workload_csv", "be_workload_csv"]
    if backend in ("saturo", "update_interval", "p_history", "d_quantile"):
        if values.get("load_format", "auto") == "dummy":
            raise ValueError("Canonical launchers require real weights (load_format=auto)")
        required += ["model", "runtime_profile", "model_len_predictor"]
    elif backend in ("p_overhead", "d_sync_overhead"):
        required += ["runtime_profile"]
        if backend == "d_sync_overhead":
            required += ["model_len_predictor"]
    elif backend == "allocator_overhead":
        required = ["resource_profile"]
    for key in required:
        if not values.get(key) or not repo_path(values[key]).exists():
            raise ValueError(f"{case['id']}: missing {key}: {values.get(key)}")
    if backend in ("p_history", "p_overhead", "d_sync_overhead"):
        history = case["study_parameters"].get("history_csv")
        if not history or not repo_path(history).is_file():
            raise ValueError("Independent history_csv is required for history-based experiments")
        if repo_path(history) == repo_path(values["rt_workload_csv"]):
            raise ValueError("history_csv must be separate from the evaluation trace")


def execute_cases(cases):
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs would overwrite experiment results")
    unavailable = [c for c in cases if c["blockers"] or not c["command"]]
    if unavailable:
        details = "\n".join(c["id"] + ": " + "; ".join(c["blockers"]) for c in unavailable)
        raise ValueError("Execution stopped before starting any case:\n" + details)
    for case in cases:
        validate_execution_inputs(case)
        directory = Path(case["directory"])
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Refuse to overwrite existing experiment: " + str(directory))
    for case in cases:
        directory = Path(case["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(case["config"], indent=2))
        (directory / "case.json").write_text(json.dumps(case, indent=2))
        env = os.environ.copy()
        env.setdefault("VLLM_USE_V1", "0")
        env.setdefault("MPLBACKEND", "Agg")
        for key, value in case["config"].get("env", {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        with (directory / "launcher.log").open("w") as log:
            proc = subprocess.run(case["command"], cwd=str(ROOT), env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
        (directory / "launcher_status.json").write_text(json.dumps({
            "returncode": proc.returncode, "result_kind": case["result_kind"]}, indent=2))
        if not proc.returncode and case["result_kind"] == "requires_paper_result_validation":
            status = {"status": "unvalidated", "reason": "Run analysis/summarize.py on per-request measurements; joint TTFT/TBT is mandatory."}
            (directory / "result_validation.json").write_text(json.dumps(status, indent=2))
        if proc.returncode:
            raise RuntimeError(f"{case['id']} failed; see {directory / 'launcher.log'}")


def main(default_study=None):
    parser = argparse.ArgumentParser(description="Paper experiment matrix; planning is the default.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--study", choices=sorted(STUDIES), default=default_study)
    parser.add_argument("--config", default="saturo/experiments/e2e/paper.example.json")
    parser.add_argument("--mode", choices=("paper", "smoke"), default="paper")
    parser.add_argument("--methods", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--rps", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backends", default="", help="Optional backend filter, e.g. d_sync_overhead")
    parser.add_argument("--values", default="", help="Override the scalar sensitivity/overhead grid")
    parser.add_argument("--output-dir", default="saturo/results/paper_experiments")
    parser.add_argument("--print-plan", action="store_true",
                        help="Print only; this is also the default without --execute.")
    parser.add_argument("--write-plan", default="", help="Optional plan JSON output, no experiment launch.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(STUDIES, indent=2, ensure_ascii=False)); return
    if not args.study:
        parser.error("--study is required; see --list")
    if args.print_plan and args.execute:
        parser.error("--print-plan and --execute are mutually exclusive")
    try:
        config = load_config(args.config, args.mode)
        rates = [float(v) for v in args.rps.split(",") if v] or None
        if rates and any(not math.isfinite(v) or v <= 0 for v in rates):
            raise ValueError("--rps must be finite and positive")
        values = [float(v) for v in args.values.split(",") if v] or None
        if values and any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("--values must be finite and positive")
        selected = generate_cases(
            args.study, config, mode=args.mode, rps=rates,
            methods=args.methods.split(",") if args.methods else None,
            models=args.models.split(",") if args.models else None,
            workloads=args.workloads.split(",") if args.workloads else None, values=values)
        if args.backends:
            filters = set(args.backends.split(","))
            unknown = filters - {c["backend"] for c in selected}
            if unknown:
                raise ValueError("Unknown/unavailable backend filter: " + str(sorted(unknown)))
            selected = [c for c in selected if c["backend"] in filters]
        total = len(selected)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            selected = selected[:args.limit]
        cases = [prepare_case(c, config, args.output_dir) for c in selected]
        plan = {"study": args.study, "source": "quan_xing (18).pdf",
                "mode": args.mode, "grid": STUDIES[args.study],
                "grid_override_values": values,
                "total_cases_in_grid": total, "selected_cases": len(cases),
                "cases": cases}
        if args.write_plan:
            path = repo_path(args.write_plan); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        if args.execute:
            execute_cases(cases)
        else:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, str(exc) + "\n")

