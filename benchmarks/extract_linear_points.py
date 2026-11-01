#!/usr/bin/env python3
"""Extract compact stateless-linear points + fitted curve from a profile.

Reads a fitted time_predictor JSON and its .raw_profile.json, then writes a
small JSON with {label, xs, ys, fit} so the (large) raw profile never has to
leave the machine it was produced on. Uses only the stdlib.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitted", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fitted = json.load(open(args.fitted))
    lin_fit = fitted.get("time_predictor", {}).get("linear", {})

    raw = json.load(open(args.raw))
    rows = [r for r in raw.get("rows", [])
            if r.get("profile_pass") == "linear"
            and "stateless_linear" in r.get("group_per_layer_ms", {})]
    rows.sort(key=lambda r: r["n_tokens"])
    xs = [int(r["n_tokens"]) for r in rows]
    ys = [float(r["group_per_layer_ms"]["stateless_linear"]["median"])
          for r in rows]

    with open(args.out, "w") as f:
        json.dump({"label": args.label, "xs": xs, "ys": ys, "fit": lin_fit}, f)
    print(f"[extract] {args.label}: {len(xs)} pts -> {args.out}")


if __name__ == "__main__":
    main()
