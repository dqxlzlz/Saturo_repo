#!/usr/bin/env python3
"""Download candidate RT/online datasets for Azure-distribution comparison.

Datasets (all contain real token content -> usable for token_ids):
  - Azure   : AzureLLMInferenceTrace_conv.csv (reference distribution)
  - ShareGPT: anon8231489123/ShareGPT_Vicuna_unfiltered  (JSON, public)
  - WildChat: allenai/WildChat-1M                          (parquet, gated)
  - LMSYS   : lmsys/lmsys-chat-1m                          (parquet, gated)

Gated datasets (WildChat / LMSYS) require:
  1. Accepting the license on the HuggingFace dataset page.
  2. A token exported via env HF_TOKEN (or `huggingface-cli login`).

Local layout produced (under --data-root, default ``data``):
  data/datasets/AzureLLMInferenceTrace_conv.csv
  data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
  data/wildchat/0000.parquet
  data/lmsys/0000.parquet

Usage:
  # download everything (needs HF_TOKEN for the gated ones)
  python saturo/tools/download_candidate_datasets.py --datasets all

  # only the public ones
  python saturo/tools/download_candidate_datasets.py --datasets azure sharegpt

  # custom root + token
  HF_TOKEN=hf_xxx python saturo/tools/download_candidate_datasets.py \
      --datasets all --data-root /mnt/data
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

AZURE_URL = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/"
    "master/data/AzureLLMInferenceTrace_conv.csv"
)

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

WILDCHAT_REPO = "allenai/WildChat-1M"
LMSYS_REPO = "lmsys/lmsys-chat-1m"

ALL_DATASETS = ["azure", "sharegpt", "wildchat", "lmsys"]


def _hf_token(explicit: Optional[str]) -> Optional[str]:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")


def download_azure(data_root: Path) -> Path:
    """Download the Azure conv trace CSV from the Azure public GitHub repo."""
    out_dir = data_root / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "AzureLLMInferenceTrace_conv.csv"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[azure] already present: {out_path}")
        return out_path
    print(f"[azure] downloading {AZURE_URL}")
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with urllib.request.urlopen(AZURE_URL) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(out_path)
    print(f"[azure] saved -> {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def download_sharegpt(data_root: Path, token: Optional[str]) -> Path:
    """Download the public ShareGPT unfiltered JSON."""
    from huggingface_hub import hf_hub_download

    out_dir = data_root / "sharegpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / SHAREGPT_FILE
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[sharegpt] already present: {out_path}")
        return out_path
    print(f"[sharegpt] downloading {SHAREGPT_REPO}:{SHAREGPT_FILE}")
    cached = hf_hub_download(
        repo_id=SHAREGPT_REPO,
        filename=SHAREGPT_FILE,
        repo_type="dataset",
        token=token,
    )
    shutil.copyfile(cached, out_path)
    print(f"[sharegpt] saved -> {out_path}")
    return out_path


def _download_first_parquet(
    repo: str,
    out_dir: Path,
    token: Optional[str],
    num_shards: int = 1,
) -> List[Path]:
    """Download the first ``num_shards`` parquet shards from a HF dataset repo."""
    from huggingface_hub import HfApi, hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset", token=token)
    parquets = sorted(f for f in files if f.endswith(".parquet"))
    if not parquets:
        raise RuntimeError(f"no parquet files found in {repo}")
    selected = parquets[:max(1, num_shards)]
    saved: List[Path] = []
    for idx, remote in enumerate(selected):
        out_path = out_dir / f"{idx:04d}.parquet"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{repo}] already present: {out_path}")
            saved.append(out_path)
            continue
        print(f"[{repo}] downloading shard {remote}")
        cached = hf_hub_download(
            repo_id=repo,
            filename=remote,
            repo_type="dataset",
            token=token,
        )
        shutil.copyfile(cached, out_path)
        print(f"[{repo}] saved -> {out_path}")
        saved.append(out_path)
    return saved


def download_wildchat(data_root: Path, token: Optional[str],
                      num_shards: int) -> List[Path]:
    if token is None:
        print("[wildchat] WARNING: gated dataset; set HF_TOKEN after accepting "
              "the license at https://huggingface.co/datasets/allenai/WildChat-1M")
    return _download_first_parquet(
        WILDCHAT_REPO, data_root / "wildchat", token, num_shards)


def download_lmsys(data_root: Path, token: Optional[str],
                   num_shards: int) -> List[Path]:
    if token is None:
        print("[lmsys] WARNING: gated dataset; set HF_TOKEN after accepting "
              "the license at https://huggingface.co/datasets/lmsys/lmsys-chat-1m")
    return _download_first_parquet(
        LMSYS_REPO, data_root / "lmsys", token, num_shards)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download candidate online/RT datasets for comparison.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=f"Subset of {ALL_DATASETS} or 'all'.",
    )
    parser.add_argument("--data-root", default="data",
                        help="Root directory for downloaded datasets.")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (else uses HF_TOKEN env).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Parquet shards to fetch for wildchat/lmsys.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    token = _hf_token(args.hf_token)

    requested = ALL_DATASETS if "all" in args.datasets else args.datasets
    unknown = [d for d in requested if d not in ALL_DATASETS]
    if unknown:
        print(f"ERROR: unknown datasets {unknown}; valid: {ALL_DATASETS}")
        sys.exit(1)

    failures = []
    for name in requested:
        try:
            if name == "azure":
                download_azure(data_root)
            elif name == "sharegpt":
                download_sharegpt(data_root, token)
            elif name == "wildchat":
                download_wildchat(data_root, token, args.num_shards)
            elif name == "lmsys":
                download_lmsys(data_root, token, args.num_shards)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}")
            failures.append(name)

    print("\n[download] done.")
    if failures:
        print(f"[download] failed datasets: {failures}")
        print("[download] gated sets (wildchat/lmsys) need license + HF_TOKEN.")
        sys.exit(2)


if __name__ == "__main__":
    main()
