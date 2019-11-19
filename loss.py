import torch
from torch.nn.functional import log_softmax
from utils import to_onehot_ls


def label_smoothing_loss(preds, labels, lab_lens, vocab_size):
    onehot_ls = to_onehot_ls(labels, vocab_size)

    batch_size = preds.shape[0]

    loss = 0

    for b in range(batch_size):
        lab_len = lab_lens[b]

        # sum for seq_len, vocab_size
        loss -= torch.sum((log_softmax(preds[b][:lab_len], dim=1) * onehot_ls[b][:lab_len]) / lab_len)

    return loss
