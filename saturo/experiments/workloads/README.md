# Paper workload construction

Build real-content CSVs for **Conv-ShareGPT**, **Code-LMSYS**, and a common
**BE mix**. This is a CPU-only data preparation step; model weights, vLLM
workers, and GPUs are not needed.

## Inputs and defaults

[paper.json](paper.json) fixes the source filenames, model revisions, and seed.
Pass a local `--data-root` with this layout:

```text
<data-root>/
  AzureLLMInferenceTrace_conv_2024.csv
  AzureLLMInferenceTrace_code_2024.csv
  sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
  lmsys_chat_1m/data/*.parquet
  longbench_v1/data/gov_report.jsonl
  longbench_v1/data/multi_news.jsonl
  longbench_v1/data/qmsum.jsonl
```

The development server has these inputs under `/root/datasets`.

| Workload | Arrival/length targets | Real payload source | Requests per model |
| --- | --- | --- | ---: |
| Conv-ShareGPT | Azure conversation 2024, first 8,000 rows | ShareGPT V3 cleaned-split | 8,000 |
| Code-LMSYS | Azure code 2024, first 8,000 rows | All six LMSYS-Chat-1M shards | 8,000 |
| BE mix | All requests ready at time zero | LongBench v1 GovReport, MultiNews, QMSum | 600 (200 per task) |

The BE sources are **LongBench subsets**, not the full original datasets.
This choice and the split should be stated in the paper/artifact description.
No `_e` files or LongBench v2 files are mixed in.

## Build

Run commands from the repository root, using Python 3.12. The recorded dependency
versions are in [requirements.txt](requirements.txt):

```bash
python -m pip install -r saturo/experiments/workloads/requirements.txt

python -m saturo.experiments.workloads.build_paper \
  --data-root /root/datasets \
  --tokenizer-root data/tokenizers \
  --output-dir data/workloads/paper_20261101 \
  --download-tokenizers
```

Downloads use the official Hugging Face repositories at pinned commits and an
allowlist of tokenizer/config files. No weight files are downloaded. Llama
requires an account with access to `meta-llama/Llama-3.3-70B-Instruct`; authenticate
using `hf auth login` first. The script does not print credentials or use mirrors.

To build one model, add `--models mistral-large-123b` or `--models llama3.3-70b`.
The two models have different token IDs and require separate workload files.

For manually downloaded tokenizers, omit `--download-tokenizers` and put files in:

```text
data/tokenizers/llama3.3-70b/
data/tokenizers/mistral-large-123b/
```

Each directory needs `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, and `config.json`. Use the revisions in `paper.json`
for exact reproduction. Mistral's optional `tokenizer.model`/`tokenizer.model.v7`
can also be present. The suite builder verifies local/manual files against
official Hub file metadata recorded in [tokenizer_files.json](tokenizer_files.json)
at the pinned commits, then records SHA256s in the manifest. A standalone
`prepare` invocation accepts any local tokenizer and records its hashes.

Re-running into existing outputs fails. Add `--resume` to verify and reuse
completed CSVs; a partial CSV without its manifest must be moved aside first.
Use a new output directory for a different configuration. To change request
counts, the trace start row, candidate sample size, or seed, copy `paper.json`,
edit it, and pass `--config /path/to/custom.json`. RT candidate limit `0` uses
all available pairs and needs more CPU time/memory. `--threads` defaults to 8.

## Construction policy

1. Extract adjacent user/assistant pairs from every source conversation. Each
   user message is a standalone prompt; preceding conversation turns are not
   concatenated. Retain source shard, row, pair index, and original ID.
2. Uniformly reservoir-sample **100,000 valid pairs per RT source**, seed
   **20261101**, across the entire source (including all six LMSYS shards).
   The candidate sample is identical across model builds before tokenization.
3. Tokenize actual prompt and reference output text for the selected model with
   `add_special_tokens=False`, no chat template, and no truncation. Lengths are
   the lengths of these real token arrays; text is never padded to a target.
4. For each Azure row, find the nearest candidate length bin by Euclidean
   distance in `log1p(input_length), log1p(output_length)` space. Break bin ties
   deterministically and use the seeded RNG to select a real pair within the
   bin. Pairs can be reused across RT requests. This is an explicit matching
   policy; the paper does not specify an exact matching algorithm.
5. Preserve the first 8,000 Azure rows' relative arrivals at microsecond
   precision. Files retain **native** arrival offsets. Saturo's RT loader scales
   gaps by `native_rps / target_rps`; for example, 8,000 requests at 20 req/s
   occupy a 400-second sending window. Do not independently rescale the CSV.
6. Use all 200 examples per BE task, without replacement; save one seed-shuffled
   order for all experiments/models. BE prompts use the explicit task templates
   in `prepare.py` (also saved in manifests), including the QMSum query. Reference
   output is `answers[0]`. No text/token truncation is applied.

The 600-row BE file is a finite input set. Continuous BE backlog in a long run
requires the serving driver to keep supplying work after that set is exhausted.

## Files and validation

Each model directory contains:

```text
conv_sharegpt.csv
code_lmsys.csv
be_mix.csv
<each CSV>.manifest.json
build_manifest.json
```

CSV fields include `request_id`, `prompt`, `output`, `prompt_token_ids`,
`output_token_ids`, `prompt_len`, `output_len`, `arrival_offset_s`, and provenance.
Token arrays are JSON arrays inside quoted CSV fields. RT rows additionally keep
`azure_timestamp`, `azure_row`, `azure_prompt_len`, `azure_output_len`, matching
distance, and actual/target length ratios. The original Azure target lengths
and matched payload lengths may differ; manifests report their distributions.

`prompt_token_ids` are consumed directly by Saturo. `output_token_ids` preserve
the source reference response; serving generates its own response up to the
specified `output_len`, rather than teacher-forcing those reference tokens.

Manifests include source/tokenizer SHA256s, output SHA256, versions, seed,
candidate counts, length/matching statistics, maximum total token count, and
builder hash. Set serving `--max-model-len` large enough for the reported
`max_total_tokens`; the builder does not silently truncate long requests.
For the recorded Mistral build, the BE maximum is **59,239 total tokens**;
use `--max-model-len 65536`. The paper planning template's older 32,768 setting
must be overridden when using these untruncated inputs.

Verify every saved token array against its text/tokenizer, input/output hashes,
BE balance/uniqueness, arrival order, and the actual Saturo CSV loader:

```bash
python -m saturo.experiments.workloads.validate \
  data/workloads/paper_20261101/mistral-large-123b/conv_sharegpt.csv \
  data/workloads/paper_20261101/mistral-large-123b/code_lmsys.csv \
  data/workloads/paper_20261101/mistral-large-123b/be_mix.csv \
  --report data/workloads/paper_20261101/mistral-large-123b/validation.json
```

For a relocated build, pass `--tokenizer-dir` and, if raw inputs are unavailable,
`--skip-input-hashes`. Build in another directory with the same configuration
and compare CSV SHA256s to check reproduction. Runtime prompt reshaping should
be disabled (`RT_LEN_SCALE=1`, `RT_TRIM_TOP_FRAC=0`) for these content workloads.

For serving, use `--rt-workload-csv <model>/conv_sharegpt.csv` (or `code_lmsys.csv`)
and `--be-workload-csv <model>/be_mix.csv`, together with the matching model
weights and `--kv-handoff real` for real P-to-D inference.

## Runtime reader

The scheduler reads prepared CSVs directly through `saturo/runtime/workload.py`.
Declared prompt/reference token arrays must be valid JSON integer arrays with
lengths matching `prompt_len`/`output_len`. Missing IDs, duplicate request IDs,
invalid arrivals, and requests exceeding the context limit fail explicitly.
Azure target lengths remain provenance; the real payload lengths drive serving.
`RT_LEN_SCALE` cannot reshape tokenized prompts. Old length-only CSVs remain
supported for profiling, including dummy KV mode.

For prepared CSVs with a `.manifest.json`, the global CLI checks the CSV SHA256,
workload kind, and model tokenizer fingerprints before resource profiling.
Tokenizer files must match locally beside the model weights (or in the local
Hub cache); this check downloads nothing. The BE context requirement is also
checked before profiling; RT checks the context of its selected replay prefix.
Without a manifest, basic row/ID validation still runs, but there is no model
tokenizer fingerprint to compare.

The reader retains prompt IDs and small provenance fields, rather than retaining
full prompt/reference text or reference token arrays. Reference IDs validate the
output budget and are never sent to the worker. Results record `source_id`,
`dataset`, and Azure target lengths in `requests.csv`, and distinguish tokenized
prompt replay from legacy length-only replay in `summary.json`.

## Content-aware remaining-length prediction

Paper runs use **TF-IDF + QRF**, with token content from the real prompt and the
already-generated response prefix, plus five length features. This is a CPU
content representation, not an additional LLM/embedding service. The online
predictor never receives the complete reference response or a future suffix.
Missing prompt/prefix IDs fail explicitly; there is no zero-vector fallback.
The same content features reach synchronous admission and asynchronous target
optimization, and the resource simulator now reads token arrays from CSV too.

First prepare independent training pools. The following commands reproduce the
Mistral build with 6,000 unique prompts per source, excluding token-identical
and text-identical prompts from **both** RT evaluation workloads:

```bash
SATURO_PAPER_DATA=data/workloads/paper_20261101/mistral-large-123b
for workload in conv code; do
  if [ "$workload" = conv ]; then
    dataset=sharegpt
    source_path=/root/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
  else
    dataset=lmsys
    source_path=/root/datasets/lmsys_chat_1m/data
  fi
  python -m saturo.experiments.setup.prepare_content_history \
    --dataset "$dataset" --source "$source_path" \
    --model data/tokenizers/mistral-large-123b \
    --exclude-workload "$SATURO_PAPER_DATA/conv_sharegpt.csv" "$SATURO_PAPER_DATA/code_lmsys.csv" \
    --output "$SATURO_PAPER_DATA/predictors/train_$workload.csv" --count 6000
  python -m saturo.experiments.setup.train_qrf \
    --history "$SATURO_PAPER_DATA/predictors/train_$workload.csv" \
    --output "$SATURO_PAPER_DATA/predictors/${workload}_tfidf_qrf.joblib" \
    --feature-mode tfidf --trees 32 --max-steps 8 --max-features 2048 --jobs 4
done
```

Training partitions requests by complete prompt-token hash (80/20, seed
20261101) **before** expanding them into prefix samples. The TF-IDF vocabulary
fits only the training partition. Each artifact has a `.report.json` recording
tokenizer/source hashes, label source, feature dimensions, grouped validation
metrics, and a length-only QRF comparison; improvement is measured, not assumed.

The current artifacts use **dataset reference responses as development labels**.
They are not trained on observed Mistral generation. For paper-quality
model-specific prediction evaluation, collect independent completed requests
from the target model, then join their actual generated token arrays to the
corresponding historical input CSV:

```bash
python -m saturo.experiments.setup.train_qrf \
  --history /path/to/independent_historical_inputs.csv \
  --observed-outputs /path/to/completed_run/requests.csv \
  --output /path/to/observed_content_qrf.joblib --feature-mode tfidf
```

Online history updates retain arrival/length statistics; they do not retrain or
replace this static content model. Unknown future arrivals have no prompt yet,
so forecasts use observed historical averages, or training mean output length
during cold start. They are never assigned invented token content.

For Conv, pass `--model-len-predictor .../predictors/conv_tfidf_qrf.joblib`; for
Code, use `code_tfidf_qrf.joblib`, together with `--kv-handoff real`. Prepared
real-KV workloads reject length-only QRF artifacts and mismatched tokenizer
provenance before profiling. Worker info records `feature_mode: tfidf` so the
actual configuration is visible. The paper experiment template selects these
per-workload artifacts and real KV mode.
Changing the predictor changes progressive saturation estimates; regenerate
allocation profiles rather than reusing tables built with a length-only QRF.

Global D admission/role selection also uses QRF-predicted remaining tokens.
Its memory estimate is current prompt+observed output length plus the prediction;
`output_len`/the reference answer length is never an admission forecast or a
prediction fallback. It remains the generation budget for trace replay. Changing
that budget alone does not change the admission estimate for the same observed
prompt/prefix. Unchanged observations are cached with bounded storage.
Adaptive role selection therefore needs a QRF even if D-pool BE saturation is
disabled. `summary.json` exposes `global_admission.length_source`, feature mode,
prediction/cache counts, and `uses_trace_output_len: false` for review.

Legacy length-only/dummy profiling remains available by training with
`--feature-mode len` and running `--kv-handoff dummy`. A content predictor is
rejected in dummy D-pool mode because that path uses synthetic decode prompts.

Keep raw data, downloaded tokenizers, generated workloads, and build receipts
outside the published repository; publish the scripts, configuration, and guide.
This builder does not download or redistribute the source datasets.

For a single workload or custom source paths, use:

```bash
python -m saturo.experiments.workloads.prepare --help
python -m pytest --confcutdir=tests/saturo -q tests/saturo/test_workload_preparation.py
```
