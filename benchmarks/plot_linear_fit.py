#!/usr/bin/env python3
"""Plot the stateless-linear profile: raw data vs fitted piecewise curve.

For each TP, x = num_token, y = per-layer stateless_linear forward time (ms).
Reads the raw profile (linear-pass rows) and the fitted model, overlays the
piecewise fit on the raw scatter, and writes one PNG per TP.
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROFILE_DIR = ("saturo/results/runtime_profiles/"
               "root-models-qwen-qwen2.5-72b-instruct")
DIGESTS = {1: "7cf6d80ff959485c", 2: "830481095bd45b35", 4: "bc11c59e400ce84e"}


def _find(tp, digest, raw):
    suffix = ".raw_profile.json" if raw else ".json"
    pat = (f"{PROFILE_DIR}/time_predictor_*_tp{tp}_pp1_torch.float16_"
           f"{digest}.json" + ("" if not raw else ""))
    pat = (f"{PROFILE_DIR}/*tp{tp}_pp1_torch.float16_{digest}"
           f"{'.json.raw_profile.json' if raw else '.json'}")
    hits = [h for h in glob.glob(pat)
            if h.endswith(suffix) and (raw or not h.endswith(".raw_profile.json"))]
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


class Piecewise:
    def __init__(self, data):
        self.width = int(data.get("width", 32))
        self.coef = {int(k): (float(v[0]), float(v[1]))
                     for k, v in data.get("coef", {}).items()}
        fb = data.get("fallback", [0.0, 0.0])
        self.fallback = (float(fb[0]), float(fb[1]))
        self._repair()

    def _repair(self):
        seg0, seg1 = self.coef.get(0), self.coef.get(1)
        if seg0 is None or seg1 is None:
            return
        w = float(self.width)
        if seg0[0] * (w - 1.0) + seg0[1] > seg1[0] * w + seg1[1]:
            self.coef[0] = seg1

    def predict(self, n):
        sid = int(math.floor(n / self.width))
        a, b = self.coef.get(sid, self.fallback)
        return max(0.0, a * n + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/admin/Transfer-station")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    written = []
    for tp, digest in DIGESTS.items():
        raw = json.load(open(_find(tp, digest, raw=True)))
        fitted = json.load(open(_find(tp, digest, raw=False)))
        lin_model = Piecewise(fitted["time_predictor"]["linear"])

        rows = [r for r in raw["rows"] if r.get("profile_pass") == "linear"]
        rows.sort(key=lambda r: r["n_tokens"])
        xs = [r["n_tokens"] for r in rows]
        ys = [r["group_per_layer_ms"]["stateless_linear"]["median"]
              for r in rows]

        fit_x = list(range(2, 8193, 2))
        fit_y = [lin_model.predict(x) for x in fit_x]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(xs, ys, s=4, alpha=0.4, color="#1f77b4",
                   label="raw profile (per-layer median)")
        ax.plot(fit_x, fit_y, color="#d62728", lw=1.5,
                label=f"piecewise fit (width={lin_model.width})")
        ax.set_xlabel("num_token")
        ax.set_ylabel("stateless_linear per-layer forward time (ms)")
        ax.set_title(f"TP-{tp}  stateless-linear: raw vs fitted")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"linear_fit_tp{tp}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        written.append(out)
        print(f"[plot] TP-{tp}: {len(xs)} pts -> {out}")

    for w in written:
        print("wrote", w)


if __name__ == "__main__":
    main()
