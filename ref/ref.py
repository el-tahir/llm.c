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

if __name__ == "__main__":
    x = np.array([-3.0,-1.5, 0.0, 0.1, 1.0 / 3.0, 1.5, 3.14159265, 1e8])
    dump("stage0", x)
    print(x)
