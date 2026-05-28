import numpy as np, os

big_files = [
    # "Train.npy",
    # "Test-143.npy",
    "Test-143_unmasked.npy",
    # "TEST464.npy"
    # 把上一步 find 列出的文件都放进来
]


for f in big_files:
    print(f"Loading {f} (mmap mode)...")
    data = np.load(f, mmap_mode='r')
    out = f.replace('.npy', '.npz')
    print(f"Saving compressed {out} ...")
    np.savez_compressed(out, data)
    size_gb = os.path.getsize(out) / 1e9
    print(f"{f} -> {out}, size: {size_gb:.2f} GB")
    if size_gb > 2.0:
        print("Still too large, need to split instead.")
    else:
        os.remove(f)