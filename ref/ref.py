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

if __name__ == "__main__":
    x = np.array([-3.0,-1.5, 0.0, 0.1, 1.0 / 3.0, 1.5, 3.14159265, 1e8])
    dump("stage0", x)
    print(x)
    stage2()
