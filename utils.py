from struct import unpack
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def to_onehot(label: torch.tensor, num_classes: int) -> torch.tensor:
    """ (batch, seq_len) -> (batch, seq_len, num_classes)
    """
    return torch.eye(num_classes)[label].to(DEVICE)
