import os
import numpy as np
import numpy.typing as npt

REF_DIR = os.path.dirname(os.path.abspath(__file__))

def dump(name: str, arr: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """ Write arr to ref/<name>.bin as raw fp32: no header, no shape, just values."""
    arr = np.asarray(arr, dtype=np.float32)
    path = os.path.join(REF_DIR, name + ".bin")
    arr.tofile(path)
    print(f"dumped {name}.bin shape={arr.shape} n={arr.size} bytes={arr.nbytes}")
    return arr

#------------------------------------
# stage 2: the three primitives

def rmsnorm(x, weight, eps=1e-5):
    return x / np.sqrt(np.mean(x * x) + eps) * weight

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def matmul(w, x):
    # w is (d, n) row-major, x is (n,) , result is (d,)
    return w @ x

def stage2():
    rng = np.random.default_rng(2)
    n, d = 288, 768

    # dump() returns the fp32 array, so everything below is computed from
    # exactly the bytes C will read - not the fp64 originals
    x = dump("s2_rms_x", rng.standard_normal(n))
    w = dump("s2_rms_weight", 1.0 + 0.5 * rng.standard_normal(n))
    dump("s2_rms_out", rmsnorm(x, w))

    s = dump("s2_soft_x", 3.0 * rng.standard_normal(256))
    dump("s2_soft_out", softmax(s))

    mx = dump("s2_matmul_x", rng.standard_normal(n))
    # scale weights by 1 / sqrt(n): keeps dot-product output variance ~1
    # instead of growing with n (each of the n summed terms adds variance)
    mw = dump("s2_matmul_w", rng.standard_normal((d, n)) / np.sqrt(n))

    dump("s2_matmul_out", matmul(mw, mx))

#------------------------------
# stage 3: token embeddings

CHECKPOINT = os.path.join(os.path.dirname(REF_DIR), "stories15M.bin")

def load_config(path: str) -> dict:
    """read the 7-int32 header. mirrors read_checkpoint's config in run.c"""
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
    np.fromfile(path, dtype=np.int32, count=7)
    return dict(dim=int(dim), hidden_dim=int(hidden_dim), n_layers=int(n_layers),
        n_heads=int(n_heads), n_kv_heads=int(n_kv_heads),
        vocab_size=int(vocab_size), seq_len=int(seq_len),
        shared=bool(vocab_size > 0))

def load_embedding_table(path: str = CHECKPOINT):
    """the table is the first tensor in the blob, right after the 28-byte header"""
    p = load_config(path)
    n = p["vocab_size"] * p["dim"]
    tbl = np.fromfile(path, dtype=np.float32, count=n, offset=7 * 4)
    return p, tbl.reshape(p["vocab_size"], p["dim"])

S3_TOKENS = [0, 1, 2534, 31999]

def stage3():
    p, tbl = load_embedding_table()
    print(f"config: {p}")
    print(f"embedding table : {tbl.shape}")
    for t in S3_TOKENS:
        dump(f"s3_embed_{t}", tbl[t])



if __name__ == "__main__":
    x = np.array([-3.0,-1.5, 0.0, 0.1, 1.0 / 3.0, 1.5, 3.14159265, 1e8])
    dump("stage0", x)
    print(x)
    stage2()
    stage3()
