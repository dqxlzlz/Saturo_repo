# End-to-end serving

Install the project as described in the root [README](../../../README.md).
The entrypoint is `saturo/run_saturo_endtoend_experiment.py`; configuration and
process management live in `saturo/cli/endtoend_experiment.py`, and the global
scheduler starts the Ray serving workers.

## Models and configurations

| Paper model | Single-run configuration |
| --- | --- |
| Llama-3.3-70B-Instruct | [llama3.3-70b.example.json](llama3.3-70b.example.json) |
| Mistral-Large-Instruct-2411 (123B) | [mistral-large-123b.example.json](mistral-large-123b.example.json) |

Both configurations use the paper setup: 32 GPUs, 8,000 RT requests, a 1,000 ms
TTFT SLO, a 150 ms TBT SLO, and a 65,536-token context limit. The default workload
is Conv-ShareGPT with the LongBench BE mix. For Code-LMSYS, select its RT CSV and
the corresponding `code_tfidf_qrf.joblib` predictor.

Copy the selected configuration outside the repository and replace every
`/path/to/...` value with local model weights, model-specific RT/BE CSVs, a
compatible content QRF artifact, measured execution-time profiles, and an
allocation profile for the model/GPU/TP/PP configuration. No weights are
downloaded by these templates.

The [workload guide](../workloads/README.md) covers real token IDs and content QRF
training. The serving model, workloads, and predictor must use the same tokenizer.
Preparation entrypoints are under `saturo/experiments/setup/`.

## Preview and run

Run from the repository root. JSON paths are resolved relative to this root;
CLI options override JSON values. For Llama:

```bash
cp saturo/experiments/e2e/llama3.3-70b.example.json /tmp/saturo-run.json
```

For Mistral, copy `saturo/experiments/e2e/mistral-large-123b.example.json` to the
same local configuration path instead. After filling the paths and services:

```bash
.venv/bin/python saturo/run_saturo_endtoend_experiment.py \
  --config /tmp/saturo-run.json --print-command

.venv/bin/python saturo/run_saturo_endtoend_experiment.py \
  --config /tmp/saturo-run.json --output-dir /path/to/new_run
```

`--print-command` does not collect profiles or start workers.
`--dry-run-worker-plan` invokes the allocator and can collect missing profiles;
provide measured profiles for planning without collection.

The templates use an existing Ray cluster (`ray_address=auto`). All participating
hosts must access the same absolute KV transfer directory. The KV backend uses
CPU/shared-filesystem transfer with full-attention, unquantized FLASH_ATTN NHD
caches. A synthetic `--kv-handoff dummy` mode is also available with compatible
length-only predictors.

## History, controls, and output

Install `saturo/requirements-traces.txt` and configure an existing Redis service
for the templates' Redis history settings. Set `env.SATURO_REDIS_URL` to an
address reachable by all workers; the localhost URL is an example for a local
service. History and trace keys are separated by model/workload.

The templates refresh P/D saturation controls every 40 seconds, use 600 seconds
of history, enable adaptive roles, and enable full resource replanning every
1,800 seconds. `decode_target_async` selects asynchronous D-target updates.
Use `--help` for shared scheduler parameters.

The runner writes `experiment_config.json`, `runner_command.json`,
`runner_stdout.log`, and `runner_summary.json`. The scheduler additionally writes
`resource_worker_plan.json`, `requests.csv`, `summary.json`, and worker logs.
Request completion and latency/SLO analysis are separate; see the
[experiment index](../README.md) for the analysis entrypoints.

The [paper matrix](paper.example.json) supplies both models to load sweeps,
ablations, and sensitivity studies. Matrix launchers plan by default and require
`--execute` to run selected cases.
