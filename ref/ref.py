import os
import numpy as np
import numpy.typing as npt
import struct
import re

REF_DIR = os.path.dirname(os.path.abspath(__file__))

def dump(name: str, arr: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Write arr to ref/<name>.bin as raw fp32: no header, no shape, just values."""
    arr = np.asarray(arr, dtype=np.float32)
    path = os.path.join(REF_DIR, name + ".bin")
    arr.tofile(path)
    print(f"dumped {name}.bin shape={arr.shape} n={arr.size} bytes={arr.nbytes}")
    return arr

def dump_bytes(name: str, b: bytes) -> bytes:
    """write raw bytes to ref/<name>.bin. no header, no nothing, pure bytes"""
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(b)
    print(f"dumped {name}.bin bytes={len(b)}")
    return b


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

#------------------------------------
# stage 6: multi-head attention with grouped-query support

def mha_batch(x, wq, wk, wv, wo, n_heads, n_kv_heads):
    """causal multi-head attention over a whole sequence, computed at once.
    same batch style as attention_batch - full (T, T) scores, -inf mask - but
    every head, with the kv sharing, plus wo.
    returns (xb, out): the concatenated head outputs, and the block output"""

    T, dim = x.shape
    head_size = dim // n_heads
    kv_mul = n_heads // n_kv_heads

    q = (x @ wq.T).astype(np.float32)
    k = (x @ wk.T).astype(np.float32)
    v = (x @ wv.T).astype(np.float32)
    for t in range(T):
        q[t] = rope(q[t], head_size, t)
        k[t] = rope(k[t], head_size, t)

    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    xb = np.zeros((T, dim), dtype=np.float32)
    for h in range(n_heads):
        kvh = h // kv_mul           #the  whole of GQA, again
        qh = q[:, h   * head_size : (h +   1) * head_size]
        kh = k[:, kvh * head_size : (kvh + 1) * head_size]
        vh = v[:, kvh * head_size : (kvh + 1) * head_size]

        s = (qh @ kh.T).astype(np.float32) / np.float32(np.sqrt(head_size))
        s[mask] = -np.inf
        xb[:, h * head_size : (h + 1) * head_size] = softmax_rows(s) @ vh

    return xb, (xb @ wo.T).astype(np.float32)

S6_GQA = dict(dim=48, hidden_dim=128, n_layers=2, n_heads=6, n_kv_heads=2,
    vocab_size=64, seq_len=16)

def write_checkpoint(path, p, seed=6):
    """write a tiny v0 checkpoint of random weights, in exactly the layout
    read_checkpoint walks. the point of it is n_kv_heads != n_heads, which
    stories15M cannot give us"""
    rng = np.random.default_rng(seed)
    head_size = p["dim"] // p["n_heads"]
    kv_dim = p["n_kv_heads"] * head_size
    n_layers, dim, hidden_dim = p["n_layers"], p["dim"], p["hidden_dim"]

    def g(*shape):
        # 1/sqrt(dim) for the same reason as the stage 2 matmul weights: keeps
        # the scores near unit spread, so softmax blends instead of saturating
        return (rng.standard_normal(shape) / np.sqrt(dim)).astype(np.float32)

    def norm(*shape):
        return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)

    tensors = [
        g(p["vocab_size"], dim),        # token_embedding_table
        norm(n_layers, dim),            # rms_att_weight
        g(n_layers, dim, dim),          # wq
        g(n_layers, kv_dim, dim),       # wk
        g(n_layers, kv_dim, dim),       # wv
        g(n_layers, dim, dim),          # wo
        norm(n_layers, dim),            # rms_ffn_weight
        g(n_layers, hidden_dim, dim),   # w1
        g(n_layers, dim, hidden_dim),   # w2
        g(n_layers, hidden_dim, dim),   # w3
        norm(dim),                      # rms_final_weight
        np.zeros((p["seq_len"], head_size // 2), np.float32), # legacy freq_cos
        np.zeros((p["seq_len"], head_size // 2), np.float32), # legacy freq_sin
    ]
    header = np.array([dim, hidden_dim, n_layers, p["n_heads"], p["n_kv_heads"],
        p["vocab_size"], p["seq_len"]], np.int32)

    with open(path, "wb") as f:
        f.write(header.tobytes())
        for t in tensors:
            f.write(t.tobytes())
    print(f"wrote {os.path.basename(path)}: {os.path.getsize(path)} bytes")

S6_LAYERS = [0, 3] # same two layers as stage 5
S6_GQA_T = 7 # sequence length for the synthetic model

def stage6():
    # part 1: stories15M, on exactly the same  bytes stage 5 used, so head 0's slice
    # of the concatenation is comparable against the s5 dumps

    p, w = load_weights()
    x = np.fromfile(os.path.join(REF_DIR, "s5_xin.bin"),
        dtype=np.float32).reshape(S5_T, p["dim"])

    for l in S6_LAYERS:
        _, out = mha_batch(x, w["wq"][l], w["wk"][l], w["wv"][l], w["wo"][l],
            p["n_heads"], p["n_kv_heads"])

        for t in range(S5_T):
            dump(f"s6_out_l{l}_{t}", out[t])

    # part 2: the synthetic model, the only place the kv mapping is not a no-op
    path = os.path.join(REF_DIR, "gqa.bin")
    write_checkpoint(path, S6_GQA)
    gp, gw = load_weights(path)
    rng = np.random.default_rng(66)
    gx = dump("s6_gqa_x", rng.standard_normal((S6_GQA_T, gp["dim"])))

    for l in range(gp["n_layers"]):
        _, out = mha_batch(gx, gw["wq"][l], gw["wk"][l], gw["wv"][l], gw["wo"][l],
            gp["n_heads"], gp["n_kv_heads"])
        for t in range(S6_GQA_T):
            dump(f"s6_gqa_out_l{l}_{t}", out[t])


    #-------------------------------------
    # stage 7: the swiglu feed-forward network

def silu(v):
    return v / (1.0 + np.exp(-v))

def ffn_batch(x, w1, w2, w3):
    """the feed-forward block for a batch of positions at once.
    every row is independent"""
    hb  = (x @ w1.T).astype(np.float32)
    hb2 = (x @ w3.T).astype(np.float32)

    return ((silu(hb) * hb2).astype(np.float32) @ w2.T).astype(np.float32)

S7_LAYERS = [0, 3]

def stage7():
    p, w = load_weights()

    # reuse stage 5's input. ffn is position-independent
    x = np.fromfile(os.path.join(REF_DIR, "s5_xin.bin"),
        dtype=np.float32).reshape(S5_T, p["dim"])

    for l in S7_LAYERS:
        out = ffn_batch(x, w["w1"][l], w["w2"][l], w["w3"][l])
        for t in range(S5_T):
            dump(f"s7_out_l{l}_{t}", out[t])

    # swap w1, w2. C output must NOT match
    swapped = ffn_batch(x, w["w3"][0], w["w2"][0], w["w1"][0])
    dump("s7_swapped_l0_0", swapped[0])

# ---------------------------------
# stage 8: the whole forward pass

def rmsnorm_rows(m, weight, eps=1e-5):
    """rmsnorm on each row independently."""
    ss = np.mean(m * m, axis=1, keepdims=True)
    return ((m / np.sqrt(ss + eps)) * weight).astype(np.float32)

def forward_batch(tokens, p, w):
    """the entire model over a whole sequence at once"""
    x = w["token_embedding_table"][tokens].astype(np.float32)

    for l in range(p["n_layers"]):
        xb = rmsnorm_rows(x, w["rms_att_weight"][l])
        _, a = mha_batch(xb, w["wq"][l], w["wk"][l], w["wv"][l], w["wo"][l],
            p["n_heads"], p["n_kv_heads"])
        x = (x + a).astype(np.float32)

        xb = rmsnorm_rows(x, w["rms_ffn_weight"][l])
        x = (x + ffn_batch(xb, w["w1"][l], w["w2"][l], w["w3"][l])).astype(np.float32)
    x = rmsnorm_rows(x, w["rms_final_weight"])
    return (x @ w["wcls"].T).astype(np.float32)

S8_TOKENS = [1, 306, 3186, 29889, 0, 31999, 450, 6635, 13, 2]

def stage8():
    p, w = load_weights()
    logits = forward_batch(S8_TOKENS, p, w)
    for t in range(len(S8_TOKENS)):
        dump(f"s8_logits_{t}", logits[t])

#-------------------------------------------------
# stage 9: sampling

MASK64 = (1 << 64) - 1

def random_u32(state):
    """xorshift64*. python ints are unbounded, so the left shift
    has to be masked back to 64 bits. C wraps for free"""
    state ^= state >> 12
    state ^= (state << 25) & MASK64
    state ^= state >> 27
    return state, ((state * 0x2545F4914F6CDD1D) & MASK64) >> 32

def random_f32(state):
    state, u = random_u32(state)
    return state, (u >> 8) / 16777216.0

S9_SEED  = 20240824
S9_N     = 64
S9_TOP_P  = 0.9
S9_DRAWS = 100000
S9_COINS = [0.0, 0.05, 0.17, 0.31, 0.42, 0.5,
            0.63, 0.755, 0.86, 0.93, 0.977, 0.995]

def sample_mult(probs, coin):
    # a binary search over the cumulative sum - not the same as the C linear walk
    cdf = np.cumsum(probs.astype(np.float64))
    return min(int(np.searchsorted(cdf, coin, side="right")), len(probs) - 1)

def nucleus(probs, top_p):
    # indices sorted by descending probabilty
    order = np.argsort(-probs, kind="stable")
    cdf = np.cumsum(probs[order].astype(np.float64))
    k = min(int(np.searchsorted(cdf, top_p, side="right")) + 1, len(probs))
    return order, cdf, k

def sample_top_p(probs, top_p, coin):
    order, cdf, k = nucleus(probs, top_p)
    r = coin * cdf[k - 1]
    j = min(int(np.searchsorted(cdf[:k], r, side="right")), k - 1)
    return int(order[j])

def stage9():
    rng = np.random.default_rng(9)
    logits = (2.2 * rng.standard_normal(S9_N)).astype(np.float32)

    # dump() returns the fp32 array, so every number below is computed from
    # exactly the bytes C will read from disk
    probs = dump("s9_probs", softmax(logits))

    # qsort is not stable and argsort is: a tie would let the two
    # implementations disagree for a reason that is not a bug
    assert len(np.unique(probs)) == S9_N, "toy probabilties contain a tie"

    state = S9_SEED
    stream = []
    for _ in range(16):
        state, f = random_f32(state)
        stream.append(f)
    dump("s9_rng", stream)

    dump("s9_mult", [sample_mult(probs, c) for c in S9_COINS])
    dump("s9_top_p", [sample_top_p(probs, S9_TOP_P, c) for c in S9_COINS])

    # truncated, renormalized distribution: zero outside the nuclues.
    # doubles as the nuclues mask for the containment property in C
    order, _, k = nucleus(probs, S9_TOP_P)
    trunc = np.zeros(S9_N)
    trunc[order[:k]] = probs[order[:k]] / probs[order[:k]].sum()
    dump("s9_trunc", trunc)
    print(f"    nucleus: {k} of {S9_N} tokens, ids {sorted(order[:k].tolist())}")

    # greedy decoding on the real model, one id per position
    p, w = load_weights()
    L = forward_batch(S8_TOKENS, p, w)
    dump("s9_argmax", [int(np.argmax(L[t])) for t in range(len(S8_TOKENS))])

#--------------------------------------------
# stage 10: tokenizer decode

TOKENIZER = os.path.join(os.path.dirname(REF_DIR), "tokenizer.bin")

def load_vocab(path=TOKENIZER, vocab_size=32000):
    """walk the tape"""
    with open(path, "rb") as f:
        blob = f.read()
    off = 0
    (max_len,) = struct.unpack_from("<i", blob, off); off += 4
    toks, scores = [], []
    for _ in range(vocab_size):
        (sc,) = struct.unpack_from("<f", blob, off); off += 4
        (ln,) = struct.unpack_from("<i", blob, off); off += 4
        toks.append(blob[off:off + ln]); off += ln
        scores.append(sc)
    assert off == len(blob), f"consumed {off} of {len(blob)} bytes - vocab_size is wrong"
    return max_len, toks, scores

BYTE_FALLBACK = re.compile(rb"^<0x([0-9A-Fa-f]{2})>$")

def decode(toks, prev_token, token):
    piece = toks[token]
    if prev_token == 1 and piece.startswith(b" "):
        piece = piece[1:]
    m = BYTE_FALLBACK.match(piece)
    return bytes([int(m.group(1), 16)]) if m else piece

S10_EMOJI = [243, 162, 155, 141] # 😊 — four byte fallbacks, none valid on its own

def stage10():
    max_len, toks, _ = load_vocab()
    dump("s10_meta", [max_len, len(toks)])

    # the whole vocab end to end
    dump_bytes("s10_vocab", b"".join(toks))
    dump("s10_lens", [len(t) for t in toks])

    # test sequence:  BOS strip, <unk>, multi-byte token, a byte fallback
    pieces = [decode(toks, S8_TOKENS[i - 1] if i else -1, S8_TOKENS[i])
        for i in range(len(S8_TOKENS))]
    dump_bytes("s10_decode", b"".join(pieces))
    dump("s10_piece_lens", [len(p) for p in pieces])

    # BOS test
    dump_bytes("s10_bos",   decode(toks, 1,     9038))
    dump_bytes("s10_nobos", decode(toks, 29889, 9038))

    # a character the vocabulary has no token for
    dump_bytes("s10_emoji", b"".join(decode(toks, -1, i) for i in S10_EMOJI))

    # stage 9's greedy ids as text
    g = np.fromfile(os.path.join(REF_DIR, "s9_argmax.bin"), dtype=np.float32).astype(int)
    greedy = b"".join(decode(toks, S8_TOKENS[t], int(g[t])) for t in range(len(g)))
    dump_bytes("s10_greedy", greedy)
    print(f"     greedy : {greedy!r}")

if __name__ == "__main__":
    x = np.array([-3.0,-1.5, 0.0, 0.1, 1.0 / 3.0, 1.5, 3.14159265, 1e8])
    dump("stage0", x)
    print(x)
    stage2()
    stage3()
    stage4()
    stage5()
    stage6()
    stage7()
    stage8()
    stage9()
    stage10()
