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

#--------------------------
# stage 4: rotary position embeddings

def rope(v, head_size, pos):
    """rotate each adjacent pair of v by pos * theta_i. returns a new array"""
    v = np.array(v, dtype=np.float32)
    for i in range(0, v.shape[0], 2):
        pair = (i % head_size) // 2; # pair index within this head
        theta = 10000.0 ** (-2.0 * pair / head_size)
        a = pos * theta
        cos, sin = np.cos(a), np.sin(a)
        x0, x1 = v[i], v[i + 1] # read both before writing either
        v[i] = x0 * cos - x1 * sin
        v[i + 1] = x0 * sin + x1 * cos
    return v

S4_POSITIONS = [0, 1, 5, 42, 255]

def stage4():
    rng = np.random.default_rng(4)
    dim, head_size = 288, 48
    q = dump("s4_q", rng.standard_normal(dim))
    k = dump("s4_k", rng.standard_normal(dim))
    for pos in S4_POSITIONS:
        dump(f"s4_q_rope_{pos}", rope(q, head_size, pos))
        dump(f"s4_k_rope_{pos}", rope(k, head_size, pos))

#--------------------
# stage 5: single-head causal attention with a kv cache

def load_weights(path: str = CHECKPOINT):
    """walk the blob tensor by tensor. mirror's memory_map_weights"""
    p = load_config(path)
    p["vocab_size"] = abs(p["vocab_size"]) # the sign was the shared-classifier flag
    blob = np.fromfile(path, dtype=np.float32, offset=7 * 4)

    dim, hidden_dim, n_layers = p["dim"], p["hidden_dim"], p["n_layers"]
    head_size = dim // p["n_heads"]
    kv_dim = p["n_kv_heads"] * head_size

    w, off = {}, 0
    def take(name, *shape):
        nonlocal off
        n = int(np.prod(shape))
        w[name] = blob[off: off + n].reshape(shape)
        off += n

    take("token_embedding_table", p["vocab_size"], dim)
    take("rms_att_weight", n_layers, dim)
    take("wq", n_layers, dim, dim)
    take("wk", n_layers, kv_dim, dim)
    take("wv", n_layers, kv_dim, dim)
    take("wo", n_layers, dim, dim)
    take("rms_ffn_weight", n_layers, dim)
    take("w1", n_layers, hidden_dim, dim)
    take("w2", n_layers, dim, hidden_dim)
    take("w3", n_layers, hidden_dim, dim)
    take("rms_final_weight", dim)
    off += p["seq_len"] * head_size // 2 # skip legacy freq_cos
    off += p["seq_len"] * head_size // 2 # skip leagacy freq_sin

    if p["shared"]:
        w["wcls"] = w["token_embedding_table"]
    else:
        take("wcls", p["vocab_size"], dim)

    assert off == blob.size, f"layout mismatch: walked {off} of {blob.size} floats"
    return p, w

S5_T = 5 # sequence length for the equivalence test
S5_LAYERS = [0, 3]

def softmax_rows(m):
    """softmax along each row"""
    e = np.exp(m - np.max(m, axis=1, keepdims=True))
    return (e / np.sum(e, axis=1, keepdims=True)).astype(np.float32)

def attention_batch(x, wq, wk, wv, head_size):
    """causal attention for head 0 over a whole sequence, computed at once.
    deliberately not incremental and with no cache: it builds the full (T,T)
    score matrix and masks the future with -inf, the way training does it"""

    T = x.shape[0]
    q = (x @ wq.T).astype(np.float32)
    k = (x @ wk.T).astype(np.float32)
    v = (x @ wv.T).astype(np.float32)
    for t in range(T):
        q[t] = rope(q[t], head_size, t) # each row rotated by its own position
        k[t] = rope(k[t], head_size, t)
    q, k, v = q[:, :head_size], k[:, :head_size], v[:, :head_size] # head 0

    scores = (q @ k.T).astype(np.float32) / np.float32(np.sqrt(head_size))
    scores[np.triu(np.ones((T, T), dtype=bool), k=1)] = -np.inf # causal mask
    return (softmax_rows(scores) @ v).astype(np.float32)

def stage5():
    p, w = load_weights()
    dim = p["dim"]
    head_size = dim // p["n_heads"]

    rng = np.random.default_rng(5)
    x = dump("s5_xin", rng.standard_normal((S5_T, dim)))
    for l in S5_LAYERS:
        out = attention_batch(x, w["wq"][l], w["wk"][l], w["wv"][l], head_size)
        for t in range(S5_T):
            dump(f"s5_out_l{l}_{t}", out[t])





if __name__ == "__main__":
    x = np.array([-3.0,-1.5, 0.0, 0.1, 1.0 / 3.0, 1.5, 3.14159265, 1e8])
    dump("stage0", x)
    print(x)
    stage2()
    stage3()
    stage4()
    stage5()
