import configparser
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import load_htk
from frontend import frame_stacking, spec_augment


class SpeechDataset(Dataset):
    def __init__(self, config_path, no_label=False):
        config = configparser.ConfigParser()
        config.read(config_path)

        if no_label:
            script_path = config["data"]["eval_script"]
        else:
            script_path = config["data"]["train_script"]

        lmfb_dim = int(config["frontend"]["lmfb_dim"])
        specaug = bool(int(config["frontend"]["specaug"]))
        num_framestack = int(config["frontend"]["num_framestack"])

        self.lmfb_dim = lmfb_dim
        self.specaug = specaug
        self.num_framestack = num_framestack
        self.no_label = no_label

        if specaug:
            max_mask_freq = int(config["frontend"]["max_mask_freq"])
            max_mask_time = int(config["frontend"]["max_mask_time"])
            self.max_mask_freq = max_mask_freq
            self.max_mask_time = max_mask_time

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

        if self.specaug:
            x = spec_augment(x, self.lmfb_dim, self.max_mask_freq, self.max_mask_time)

        if self.num_framestack > 1:
            x = frame_stacking(x, self.num_framestack)

        x_tensor = torch.tensor(x)

        seq_len = x.shape[0]

        ret = (x_tensor, seq_len,)

        if not self.no_label:
            label = torch.tensor(list(map(int, label.split(" "))))
            lab_len = label.shape[0]
            ret += (label, lab_len)

        return ret


def collate_fn_train(batch):
    """ for dataset `no_label=False`
    """
    xs, seq_lens, labels, lab_lens = zip(*batch)

    x_batch = pad_sequence(xs, batch_first=True)
    seq_lens = torch.tensor(seq_lens)
    labels = pad_sequence(labels, batch_first=True, padding_value=1)
    lab_lens = torch.tensor(lab_lens)

    return {"x_batch": x_batch, "seq_lens": seq_lens, "labels": labels, "lab_lens": lab_lens}


def collate_fn_eval(batch):
    """ for dataset `no_label=True`
    """
    xs, seq_lens = zip(*batch)

    x_batch = pad_sequence(xs, batch_first=True)
    seq_lens = torch.tensor(seq_lens)

    return {"x_batch": x_batch, "seq_lens": seq_lens}
