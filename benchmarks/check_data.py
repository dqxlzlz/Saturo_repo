import numpy as np

def debug_bad_y(y, name="y", topk=20):
    y = np.asarray(y)
    print(f"[{name}] shape={y.shape}, dtype={y.dtype}")

    # 找非有限值：nan/inf/-inf
    bad = ~np.isfinite(y)
    n_bad = int(bad.sum())
    print(f"[{name}] non-finite count = {n_bad}")

    if n_bad > 0:
        idx = np.where(bad)[0]
        print(f"[{name}] first {min(topk, len(idx))} bad indices:", idx[:topk])
        print(f"[{name}] first bad values:", y[idx[:topk]])
        # 如果你有对应的样本ID/行号，也在这里打印出来

    # 看看是不是极端大值导致后面溢出（即便是finite也可能非常大）
    finite = y[np.isfinite(y)]
    if finite.size > 0:
        absmax = np.max(np.abs(finite))
        print(f"[{name}] finite min={finite.min()}, max={finite.max()}, absmax={absmax}")
        # 打印最大的几个值
        largest_idx = np.argsort(np.abs(finite))[-topk:]
        print(f"[{name}] largest |y| values:", finite[largest_idx])

# 在 fit 前调用
debug_bad_y(yp_tr, "yp_tr")
