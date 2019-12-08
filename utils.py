from struct import unpack
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return torch.eye(num_classes)[label].to(device)


def to_onehot_ls(labels: torch.tensor, num_classes: int, ls_prob: float = 0.9) -> torch.tensor:
    onehot = to_onehot(labels, num_classes)
    onehot_ls = ls_prob * onehot + ((1 - ls_prob) / (num_classes - 1)) * (1 - onehot)

    return onehot_ls

def subword_to_word(subwords: list):
    """ ["幾@@", "星@@", "霜", "の", "年月"]　-> ["幾星霜", "の", "年月"]
    """
    tmp = ""
    words = []
    for subword in subwords:
        if subword[-2:] == "@@":
            tmp += subword[:-2]
        else:
            words.append(tmp + subword)
            tmp = ""
    return words
