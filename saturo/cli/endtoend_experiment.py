#!/usr/bin/env python3
"""Configure and supervise one Saturo e2e experiment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from saturo.cli.global_scheduler import (
    BooleanOptionalAction, build_parser as scheduler_parser, validate_args)

PATH_OPTIONS = (
    "output_dir", "rt_workload_csv", "be_workload_csv",
    "resource_profile", "runtime_profile", "model_len_predictor", "kv_transfer_dir",
)
STAGES = (
    "load_or_collect_profiles", "allocate_p_d_t", "launch_workers",
    "warmup_if_enabled", "preload_be", "replay_rt_p_to_reserved_d",
    "collect_results_and_shutdown",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo_root() / path).resolve()


def _normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    for name in PATH_OPTIONS:
        if values.get(name):
            values[name] = str(_repo_path(values[name]))
    model = values.get("model", "")
    if model and (Path(model).expanduser().is_absolute()
                  or (repo_root() / model).exists()):
        values["model"] = str(_repo_path(model))
    return argparse.Namespace(**values)


def workload_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.rt_workload_csv or not args.be_workload_csv:
        raise ValueError("Both --rt-workload-csv and --be-workload-csv are required.")
    return _repo_path(args.rt_workload_csv), _repo_path(args.be_workload_csv)


def build_command(args: argparse.Namespace) -> list[str]:
    """Forward every shared option, including explicit false booleans."""
    args = _normalize_paths(args)
    cmd = [sys.executable, str(repo_root() / "saturo/run_saturo_global_scheduler.py")]
    for action in scheduler_parser(include_worker=False)._actions:
        if not action.option_strings or action.dest == "help":
            continue
        value = getattr(args, action.dest, action.default)
        if value is None:
            continue
        if isinstance(action, BooleanOptionalAction):
            option = next(option for option in action.option_strings
                          if option.startswith("--no-") == (not value))
            cmd.append(option)
        elif value != "":
            cmd.extend([action.option_strings[0], str(value)])
    return cmd


def describe_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args = _normalize_paths(args)
    names = [action.dest for action in scheduler_parser(include_worker=False)._actions
             if action.dest != "help"]
    return {
        "config_file": getattr(args, "config", "") or None,
        "parameters": {name: getattr(args, name) for name in names},
        "window_s": args.window_s,
        "require_complete": args.require_complete,
        "environment": dict(getattr(args, "env", {})),
        "command": build_command(args),
        "stages": list(STAGES[:2] if args.dry_run_worker_plan else STAGES),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = _normalize_paths(args)
    workload_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = describe_experiment(args)
    cmd = manifest["command"]
    (output_dir / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "runner_command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    for name, value in manifest["environment"].items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    print("[endtoend] " + shlex.join(cmd), flush=True)
    with (output_dir / "runner_stdout.log").open("w") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(repo_root()), env=env, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    if args.dry_run_worker_plan and proc.returncode == 0:
        plan_path = output_dir / "resource_worker_plan.json"
        if not plan_path.exists():
            raise RuntimeError(f"Scheduler did not produce {plan_path}")
        summary = {"status": "planned", "plan": json.loads(plan_path.read_text())}
    else:
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["runner_returncode"] = proc.returncode
    (output_dir / "runner_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Saturo scheduler failed with code {proc.returncode}; see "
            f"{output_dir / 'runner_stdout.log'}")
    if (args.require_complete and not args.dry_run_worker_plan
            and summary.get("status") != "completed"):
        raise RuntimeError(
            f"Incomplete e2e run: {summary.get('stop_reason', 'missing summary')}; "
            f"see {output_dir / 'runner_summary.json'}")
    return summary


def _config_defaults(parser: argparse.ArgumentParser, path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read --config {path}: {exc}")
    if not isinstance(payload, dict):
        parser.error("--config must contain a JSON object")
    actions = {action.dest: action for action in parser._actions
               if action.dest not in ("help", "config", "print_command")}
    unknown = set(payload) - set(actions) - {"env"}
    if unknown:
        parser.error("Unknown config keys: " + ", ".join(sorted(unknown)))
    defaults = {}
    for name, value in payload.items():
        if name == "env":
            continue
        action = actions[name]
        if value is None and action.default is None:
            defaults[name] = None
            continue
        if isinstance(action, BooleanOptionalAction):
            if not isinstance(value, bool):
                parser.error(f"Config {name} must be a JSON boolean")
        elif action.type is not None:
            if isinstance(value, bool):
                parser.error(f"Config {name} must be a number")
            try:
                converted = action.type(value)
            except (ValueError, TypeError):
                parser.error(f"Invalid config value for {name}: {value!r}")
            if action.type is int and isinstance(value, float) and value != converted:
                parser.error(f"Config {name} must be an integer")
            value = converted
        elif not isinstance(value, str):
            parser.error(f"Config {name} must be a string")
        if action.choices is not None and value not in action.choices:
            parser.error(f"Invalid config choice for {name}: {value!r}")
        defaults[name] = value
    environment = payload.get("env", {})
    if (not isinstance(environment, dict)
            or any(not isinstance(name, str) or not name or "=" in name
                   or (value is not None and not isinstance(value, str))
                   for name, value in environment.items())):
        parser.error("Config env must map environment names to strings or null")
    defaults["env"] = environment
    return defaults


def parse_args(argv=None) -> argparse.Namespace:
    parser = scheduler_parser(include_worker=False)
    parser.description = "Configure and supervise a Saturo e2e experiment."
    # Preserve the compatibility entrypoint's existing run limits.
    parser.set_defaults(output_dir="saturo/results/endtoend",
                        max_wall_s=240.0, worker_start_timeout_s=180.0, env={})
    runner = parser.add_argument_group("Experiment runner")
    runner.add_argument("--config", default="",
                        help="JSON config; CLI flags override it. Local paths are repo-relative.")
    runner.add_argument("--print-command", action="store_true",
                        help="Print resolved config and command; start no profiling or workers.")
    runner.add_argument("--require-complete", action=BooleanOptionalAction, default=True,
                        help="Fail on unfinished requests or request errors.")
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default="")
    config_path = preliminary.parse_known_args(argv)[0].config
    if config_path:
        config_path = _repo_path(config_path)
        parser.set_defaults(**_config_defaults(parser, config_path))
    args = parser.parse_args(argv)
    args.config = str(config_path) if config_path else ""
    validate_args(args, parser)
    for name in ("model", "resource_config_filter", "rt_workload_csv", "be_workload_csv"):
        if not getattr(args, name):
            parser.error("--" + name.replace("_", "-") + " is required")
    return _normalize_paths(args)


def main() -> None:
    args = parse_args()
    result = describe_experiment(args) if args.print_command else run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
