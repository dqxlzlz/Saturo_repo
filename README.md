# Saturo

Saturo is an LLM serving system for colocating real-time (RT) and best-effort
(BE) workloads. Built on vLLM V0 and Ray, it uses latency and memory headroom
to run batch work alongside online requests.

Serving instances are organized into three pools:

- **P:** RT prefill and BE execution within the prefill latency budget.
- **D:** RT decode and BE execution within the token latency and memory budgets.
- **T:** Dedicated BE execution.

The global scheduler routes requests across these pools. A resource allocator
selects pool sizes and tensor/pipeline parallelism using measured capacity
profiles. Remaining-length and execution-time predictors support saturation
control and admission decisions.

## Repository structure

| Path | Contents |
| --- | --- |
| `saturo/cli/` | Serving and end-to-end experiment entrypoints |
| `saturo/runtime/` | Global scheduler, Ray workers, role management, and KV transfer |
| `saturo/resource_allocation/` | Capacity profiles and resource allocation |
| `saturo/predictors/` | Remaining-length QRF and execution-time prediction |
| `saturo/saturation/` | Saturation control and request forecasts |
| `saturo/simulators/` | Pool-level simulation tools |
| `saturo/experiments/` | Data preparation, training, experiment launchers, and analysis |
| `vllm/` | Inference runtime |
| `csrc/`, `cmake/` | Native kernels and build configuration |
| `requirements/` | Dependencies and environment constraints |
| `tests/` | Unit and component tests |

## Installation

Use Linux, NVIDIA GPUs, and a CUDA toolkit/compiler compatible with PyTorch.
The reference environment uses Python 3.12, PyTorch 2.7.0, and CUDA 12.6.
Run the following from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements/saturo-tested.txt -e .
python -m pip install -c requirements/saturo-tested.txt scikit-learn joblib

export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

The constraints file records key dependencies for the reference environment,
including the scikit-learn version used to train the QRF artifacts.

## Prepare inputs

A serving run requires:

1. Local model weights and the corresponding tokenizer.
2. RT and BE workload CSVs containing tokenized prompts.
3. A content QRF artifact compatible with the serving tokenizer.
4. Measured execution-time and allocation profiles for the model, GPU, and TP/PP configuration.

The [workload guide](saturo/experiments/workloads/README.md) describes how to
construct Conv-ShareGPT, Code-LMSYS, and the LongBench BE mix, and how to train
the content predictor. Training and profiling entrypoints are under
[`saturo/experiments/setup/`](saturo/experiments/setup/).

Workload token IDs, the predictor, and the serving model must use the same
tokenizer. Set the context limit to cover the selected inputs; the paper BE
workload requires `max_model_len=65536`.

## Run Saturo

The e2e configurations use the models evaluated in the paper:

| Model | Configuration |
| --- | --- |
| Llama-3.3-70B-Instruct | [llama3.3-70b.example.json](saturo/experiments/e2e/llama3.3-70b.example.json) |
| Mistral-Large-Instruct-2411 (123B) | [mistral-large-123b.example.json](saturo/experiments/e2e/mistral-large-123b.example.json) |

For Llama, copy the configuration to a local run file:

```bash
cp saturo/experiments/e2e/llama3.3-70b.example.json /tmp/saturo-run.json
```

For Mistral, copy `saturo/experiments/e2e/mistral-large-123b.example.json` to
`/tmp/saturo-run.json` instead.

Edit the copied configuration and replace every `/path/to/...` value:

| Setting | Purpose |
| --- | --- |
| `model` | Local model directory |
| `rt_workload_csv`, `be_workload_csv` | Tokenized request inputs |
| `model_len_predictor` | Content QRF artifact |
| `runtime_profile`, `resource_profile` | Measured timing and allocation profiles |
| `total_gpus`, `env.CUDA_VISIBLE_DEVICES` | GPU resources for the run |
| `kv_transfer_dir` | KV transfer directory accessible to P/D workers |
| `max_model_len` | Context limit for the model and inputs |

Preview the resolved command, then start serving:

```bash
python saturo/run_saturo_endtoend_experiment.py \
  --config /tmp/saturo-run.json --print-command

python saturo/run_saturo_endtoend_experiment.py \
  --config /tmp/saturo-run.json \
  --output-dir saturo/results/my_run
```

Both templates use 32 GPUs, 8,000 RT requests, a 1,000 ms TTFT SLO, a 150 ms
TBT SLO, and a 65,536-token context limit. The default workload is Conv-ShareGPT
with the LongBench BE mix. For Code-LMSYS, select its RT CSV and the matching
`code_tfidf_qrf.joblib` predictor. Use a shared Ray cluster, a Redis history
service reachable by all workers, and the same absolute KV transfer directory
on every participating host. The KV backend uses CPU/shared-filesystem
transfer with full-attention models and unquantized FLASH_ATTN NHD KV caches.

Each run writes configuration and command records, request results, summaries,
and worker logs under the output directory. See the
[end-to-end guide](saturo/experiments/e2e/README.md) for runtime controls and
output files.

## Experiment launchers

```bash
python saturo/experiments/run.py --list
```

The [experiment index](saturo/experiments/README.md) covers load sweeps,
ablations, sensitivity studies, and overhead measurements. Matrix launchers
generate plans by default; `--execute` starts the selected cases.
[`paper.example.json`](saturo/experiments/e2e/paper.example.json) provides the
model and workload settings for paper-scale experiment planning.

## Acknowledgments and license

Saturo builds on [vLLM](https://github.com/vllm-project/vllm). Upstream source
copyright and license notices are retained. See [LICENSE](LICENSE) for the
repository license.
