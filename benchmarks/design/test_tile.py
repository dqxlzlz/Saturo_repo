import json
import argparse
import math
import matplotlib.pyplot as plt


def load_and_preprocess(json_path: str, max_y: float):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "linear_ms" not in data or not isinstance(data["linear_ms"], list):
        raise KeyError("JSON 中找不到 'linear_ms'，或它不是一个 list。")

    # 1) 过滤 y>max_y
    # 2) 对同一个 num_token 取 ms_mean 最小值（点取最小）
    min_by_token = {}
    removed = 0
    total = 0
    for item in data["linear_ms"]:
        if not isinstance(item, dict):
            continue
        if "num_token" not in item or "ms_mean" not in item:
            continue

        total += 1
        x = int(item["num_token"])
        y = float(item["ms_mean"])

        if y > max_y:
            removed += 1
            continue

        if x not in min_by_token or y < min_by_token[x]:
            min_by_token[x] = y

    if not min_by_token:
        raise ValueError(f"过滤后没有数据点（可能所有 ms_mean 都 > {max_y}）。")

    # 排序成列表
    xs = sorted(min_by_token.keys())
    ys = [min_by_token[x] for x in xs]

    print(f"Total valid items read: {total}")
    print(f"Filtered out (ms_mean > {max_y}): {removed}")
    print(f"Unique num_token after min-agg: {len(xs)}")
    return xs, ys


def plot_ranges(xs, ys, out_prefix: str, show: bool, chunk_size: int, num_plots: int):
    max_token = xs[-1]
    end_cover = max(chunk_size * num_plots, int(math.ceil(max_token / chunk_size) * chunk_size))

    # 生成分段边界（前 num_plots-1 段固定，最后一段覆盖到 end_cover）
    ranges = []
    start = 1
    for i in range(num_plots):
        end = start + chunk_size - 1
        ranges.append((start, end))
        start = end + 1

    # 如果 max_token 超过默认最后上界，把最后一段扩展
    last_start, last_end = ranges[-1]
    if max_token > last_end:
        ranges[-1] = (last_start, end_cover)

    # 画每一段
    for (lo, hi) in ranges:
        x_seg = []
        y_seg = []
        for x, y in zip(xs, ys):
            if lo <= x <= hi:
                x_seg.append(x)
                y_seg.append(y)

        if not x_seg:
            print(f"Range {lo}-{hi}: no points, skip.")
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(
            x_seg, y_seg,
            marker="o",
            linewidth=1,
            markersize=2,          # 点更小：1~3
            markeredgewidth=0,     # 去掉点边框
            alpha=0.8              # 稍微透明一点（可选）
        )

        plt.xlabel("num_token")
        plt.ylabel("ms_mean (ms)")
        plt.title(f"num_token vs ms_mean (min per token), range {lo}-{hi}")
        plt.grid(True, linestyle="--", linewidth=0.5)

        out_path = f"{out_prefix}_{lo}_{hi}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        if show:
            plt.show()
        plt.close()

        print(f"Saved: {out_path} (points={len(x_seg)})")


def main():
    parser = argparse.ArgumentParser(description="Plot num_token vs ms_mean from JSON['linear_ms'] in 4 ranges")
    parser.add_argument("--json_path", required=True, help="Path to the input JSON file")
    parser.add_argument("--out-prefix", default="linear_ms", help="Output filename prefix (default: linear_ms)")
    parser.add_argument("--max-y", type=float, default=40.0, help="Filter out points with ms_mean > max-y (default: 40)")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Token range size per plot (default: 1024)")
    parser.add_argument("--num-plots", type=int, default=4, help="Number of plots (default: 4)")
    parser.add_argument("--no-show", action="store_true", help="Do not display plot windows")
    args = parser.parse_args()

    xs, ys = load_and_preprocess(args.json_path, max_y=args.max_y)
    plot_ranges(xs, ys, out_prefix=args.out_prefix, show=not args.no_show,
                chunk_size=args.chunk_size, num_plots=args.num_plots)


if __name__ == "__main__":
    main()
