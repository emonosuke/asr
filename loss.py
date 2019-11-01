from torch.nn.functional import log_softmax
from utils import to_onehot


def label_smoothing_loss(preds, labels, lab_lens):
    batch_size = preds.shape[0]
    onehot_ls = to_onehot(labels)

    loss = 0

    for b in enumerate(batch_size):
        lab_len = lab_lens[b]
        loss = - (log_softmax(preds[b][:lab_len], onehot_ls[b][:lab_len]) / lab_len)

    return loss
