#!/usr/bin/env python3
"""Plot stateless-linear raw points vs fitted piecewise curve from compact JSON.

Each input JSON (see extract_linear_points.py) has {label, xs, ys, fit}. One
PNG is written per input: x = num_token, y = per-layer stateless_linear
forward time (ms), with the raw scatter and the piecewise fit overlaid.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
    ap.add_argument("--points", nargs="+", required=True)
    ap.add_argument("--out-dir", default="/home/admin/Transfer-station")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.points:
        d = json.load(open(path))
        label = d["label"]
        xs, ys = d["xs"], d["ys"]
        model = Piecewise(d.get("fit", {}))
        lo = min(xs) if xs else 2
        hi = max(xs) if xs else 8192
        fit_x = list(range(int(lo), int(hi) + 1, 2))
        fit_y = [model.predict(x) for x in fit_x]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(xs, ys, s=4, alpha=0.4, color="#1f77b4",
                   label="raw profile (per-layer median)")
        ax.plot(fit_x, fit_y, color="#d62728", lw=1.5,
                label=f"piecewise fit (width={model.width})")
        ax.set_xlabel("num_token")
        ax.set_ylabel("stateless_linear per-layer forward time (ms)")
        ax.set_title(f"{label}  stateless-linear: raw vs fitted")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"linear_fit_{label}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[plot] {label}: {len(xs)} pts -> {out}")


if __name__ == "__main__":
    main()
