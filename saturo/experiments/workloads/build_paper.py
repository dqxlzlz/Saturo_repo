"""Build all three workload types for each selected paper tokenizer, without weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .prepare import build_workload, log, sha256_file
from . import prepare


TOKENIZER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "tokenizer.model", "tokenizer.model.v7"]
REQUIRED_TOKENIZER_FILES = TOKENIZER_FILES[:4]


def verify_tokenizer_files(directory, spec, expected):
    """Verify local/manual files against official Hub metadata at the pinned commit."""
    if any(spec[key] != expected[key] for key in ("repo_id", "revision")):
        raise ValueError("Tokenizer revision differs from tokenizer_files.json; update both explicitly")
    for name in REQUIRED_TOKENIZER_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"Missing tokenizer file: {directory / name}")
    for name, entry in expected["files"].items():
        path = directory / name
        if not path.is_file():  # Optional SentencePiece files.
            continue
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Tokenizer file size differs from pinned commit: {path}")
        if "sha256" in entry:
            valid = sha256_file(path) == entry["sha256"]
        else:
            data = path.read_bytes()
            valid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["git_blob_sha1"]
        if not valid:
            raise ValueError(f"Tokenizer file hash differs from pinned commit: {path}")


def download_tokenizer(spec, directory):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                      allow_patterns=TOKENIZER_FILES, local_dir=str(directory), max_workers=3,
                      endpoint="https://huggingface.co")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("paper.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, default=Path("data/tokenizers"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="Default: every model in config")
    parser.add_argument("--download-tokenizers", action="store_true", help="Only the six named tokenizer/config files")
    parser.add_argument("--resume", action="store_true", help="Skip existing CSVs only if manifests and inputs verify")
    parser.add_argument("--threads", type=int, default=8, help="Rayon tokenizer CPU threads; no GPU use")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    config = json.loads(args.config.read_text())
    expected_files = json.loads(Path(__file__).with_name("tokenizer_files.json").read_text())
    names = args.models or list(config["models"])
    for name in names:
        if name not in config["models"]:
            parser.error(f"Unknown model: {name}")
    sources = {k: str(args.data_root / value) for k, value in config["sources"].items()}
    # Import after thread configuration; no vLLM, Ray or model loading.
    from transformers import AutoTokenizer
    from .validate import validate_workload
    from .prepare import source_files

    for name in names:
        spec = config["models"][name]
        tokenizer_dir = args.tokenizer_root / name
        if args.download_tokenizers:
            download_tokenizer(spec, tokenizer_dir)
        verify_tokenizer_files(tokenizer_dir, spec, expected_files[name])
        # Fail before expensive dataset processing if access/download is incomplete.
        AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
        origin = {**spec, "download_verified": args.download_tokenizers,
                  "files_verified_against_pinned_revision": True}
        built = []
        for kind, filename in (("conv", "conv_sharegpt.csv"), ("code", "code_lmsys.csv"), ("be", "be_mix.csv")):
            output = args.output_dir / name / filename
            manifest_file = Path(str(output) + ".manifest.json")
            if kind == "be":
                inputs = {k: sources[k] for k in ("govreport", "multinews", "qmsum")}
                kwargs = dict(inputs, num_req=config["be_requests"])
                expected_sources = [Path(inputs[k]) for k in ("govreport", "multinews", "qmsum")]
            else:
                source = sources["sharegpt" if kind == "conv" else "lmsys"]
                trace = sources[kind + "_trace"]
                kwargs = dict(source=source, azure_trace=trace, num_req=config["rt_requests"],
                              candidate_limit=config["candidate_limit"], start_row=config["rt_start_row"],
                              timestamp_unit="iso")
                expected_sources = source_files(source) + [Path(trace)]
            if args.resume and output.is_file() and manifest_file.is_file():
                manifest = json.loads(manifest_file.read_text())
                if (manifest["seed"] != config["seed"] or manifest["kind"] != kind
                        or manifest["count"] != kwargs["num_req"]
                        or manifest["builder_sha256"] != sha256_file(prepare.__file__)
                        or manifest["tokenizer_origin"]["repo_id"] != spec["repo_id"]
                        or manifest["tokenizer_origin"]["revision"] != spec["revision"]):
                    raise ValueError(f"Resume configuration differs: {manifest_file}")
                if kind != "be" and (manifest["start_row"] != kwargs["start_row"]
                                      or manifest["sampling"]["candidate_limit"] != kwargs["candidate_limit"]):
                    raise ValueError(f"Resume RT window/candidate sample differs: {manifest_file}")
                saved = {r["name"]: r["sha256"] for r in manifest["inputs"]}
                if saved != {p.name: sha256_file(p) for p in expected_sources}:
                    raise ValueError(f"Resume source hashes differ: {manifest_file}")
                validate_workload(output, tokenizer_dir=tokenizer_dir, check_inputs=False)
                log(f"Verified existing {output}")
            else:
                manifest = build_workload(kind=kind, model=tokenizer_dir, output=output,
                                          seed=config["seed"], tokenizer_origin=origin, **kwargs)
            built.append({"kind": kind, "file": filename, "count": manifest["count"],
                          "sha256": manifest["output"]["sha256"]})
        summary = {"config": config, "config_sha256": sha256_file(args.config), "model": name,
                   "tokenizer": spec, "workloads": built}
        (args.output_dir / name / "build_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"model": name, "workloads": built}, indent=2), flush=True)


if __name__ == "__main__":
    main()
