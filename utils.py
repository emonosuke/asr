from struct import unpack
import numpy as np


def load_htk(filepath: str) -> np.ndarray:
    fh = open(filepath, "rb")
    spam = fh.read(12)
    _, _, samp_size, _ = unpack(">IIHH", spam)
    veclen = int(samp_size / 4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat
