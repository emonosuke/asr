import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import load_htk
from frontend import frame_stacking

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpeechDataset(Dataset):
    def __init__(self, script_path, lmfb_dim=40, num_framestack=3, no_label=False):
        self.lmfb_dim = lmfb_dim
        self.num_framestack = num_framestack
        self.no_label = no_label

        with open(script_path) as f:
            lines = [line.strip() for line in f.readlines()]

        self.dat = lines

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        if self.no_label:
            xpath = self.dat[idx]
        else:
            xpath, label = self.dat[idx].split(" ", 1)

        x = load_htk(xpath)[:, :self.lmfb_dim]

        if self.num_framestack > 1:
            x = frame_stacking(x, self.num_framestack)

        x_tensor = torch.tensor(x)

        seq_len = x.shape[0]

        ret = (x_tensor, seq_len,)

        if not self.no_label:
            label = torch.tensor(list(map(int, label.split(" "))))
            ret += (label,)

        return ret


def collate_fn_train(batch):
    """ for dataset `no_label=False`
    """
    xs, seq_lens, labels = zip(*batch)

    x_batch = pad_sequence(xs, batch_first=True)
    seq_lens = torch.tensor(seq_lens).to(DEVICE)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1).to(DEVICE)

    return {"x_batch": x_batch, "seq_lens": seq_lens, "labels": labels}


def collate_fn_eval(batch):
    """ for dataset `no_label=True`
    """
    xs, seq_lens = zip(*batch)

    x_batch = pad_sequence(xs, batch_first=True)
    seq_lens = torch.tensor(seq_lens).to(DEVICE)

    return {"x_batch": x_batch, "seq_lens": seq_lens}
