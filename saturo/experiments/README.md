# Experiment launchers

Run commands from the repository root with the installed environment. Start with
the [end-to-end guide](e2e/README.md) and the
[workload construction guide](workloads/README.md).

| Directory | Purpose |
| --- | --- |
| `e2e/` | RT load sweeps and request timelines |
| `ablation/` | P/D saturation and resource allocation ablations |
| `sensitivity/` | SLOs, GPU counts, history windows, quantiles, update intervals |
| `overhead/` | P budget, D target, and allocator CPU measurements |
| `workloads/` | Real-content dataset construction and workload statistics |
| `setup/` | Content QRF training, profile collection, allocation planning |
| `pool_validation/` | P/D/T instance probes |
| `analysis/` | Result summaries and SLO analysis |

List studies and preview commands without starting GPUs:

```bash
.venv/bin/python saturo/experiments/run.py --list
.venv/bin/python saturo/experiments/ablation/run_saturation.py --print-plan
.venv/bin/python saturo/experiments/e2e/run_load_sweep.py \
  --config saturo/experiments/e2e/paper.example.json \
  --models llama,mistral --workloads conv,code --methods saturo --rps 20 --print-plan
```

The matrix launchers default to planning; `--execute` starts selected cases.
Replace every `/path/to/...` input before execution. Case directories contain the
resolved configuration, command, logs, and completion status; nonempty result
directories are not overwritten. Use `--help` on individual launchers for filters.

`e2e/paper.example.json` is the paper-scale planning template. It requires the
corresponding weights, tokenizer-specific CSVs, trained predictors, and measured
profiles.
Paper-mode execution retains review blockers for incomplete experiment paths.

For saturation ablations, a disabled P/D pool serves RT only. Allocation
ablations keep the selected topology fixed by disabling adaptive roles and
periodic reallocation. Use independent history for predictor training and
history-based measurements. Canonical experiment launchers require real weights.

`analysis/summarize.py` reports missing per-request TBT explicitly; a successful
process exit alone does not establish joint TTFT/TBT attainment. See the root
[README](../../README.md) for the project structure and serving quickstart.
