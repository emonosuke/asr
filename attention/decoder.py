import configparser
import torch
import torch.nn as nn
from attention.attention import ContentBasedAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(self, config_path):
        super(Decoder, self).__init__()

        config = configparser.ConfigParser()
        config.read(config_path)

        self.hidden_size = int(config["model"]["hidden_size"])
        self.vocab_size = int(config["vocab"]["vocab_size"])

        self.attn = ContentBasedAttention(config_path)

        # generate
        self.L_sy = nn.Linear(self.hidden_size, self.hidden_size)
        self.L_gy = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.L_yy = nn.Linear(self.hidden_size, self.vocab_size)

        # recurrency
        # self.L_yr = nn.Linear(self.vocab_size, self.hidden_size * 4)
        self.L_yr = nn.Embedding(self.vocab_size, self.hidden_size * 4)
        self.L_sr = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.L_gr = nn.Linear(self.hidden_size * 2, self.hidden_size * 4)

    def forward(self, h_batch, seq_lens, labels):
        batch_size = h_batch.shape[0]
        frames_len = h_batch.shape[1]
        labels_len = labels.shape[1]

        """
        attn_mask = torch.tensor(
            [[[1.0] if i < seq_len else [0.0] for i in range(frames_len)] for seq_len in seq_lens],
            device=DEVICE, requires_grad=False
        )  # (batch, frames, 1)
        """
        attn_mask = torch.ones((batch_size, frames_len, 1), device=DEVICE, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0

        # for the first time (before <SOS>), generate from this 0-filled hidden_state and cell_state
        s = torch.zeros((batch_size, self.hidden_size), device=DEVICE, requires_grad=False)
        c = torch.zeros((batch_size, self.hidden_size), device=DEVICE, requires_grad=False)
        alpha = torch.zeros((batch_size, 1, frames_len), device=DEVICE, requires_grad=False)

        preds = torch.zeros((batch_size, labels_len, self.vocab_size), device=DEVICE, requires_grad=False)

        for step in range(labels_len):
            g, alpha = self.attn(s, h_batch, alpha, attn_mask)

            # generate
            y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))

            # recurrency
            rec_in = self.L_yr(labels[:, step]) + self.L_sr(s) + self.L_gr(g)

            s, c = self._func_lstm(rec_in, c)

            preds[:, step] = y

        return preds

    @staticmethod
    def _func_lstm(x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        half = 0.5
        ingate = torch.tanh(ingate * half) * half + half
        forgetgate = torch.tanh(forgetgate * half) * half + half
        cellgate = torch.tanh(cellgate)
        outgate = torch.tanh(outgate * half) * half + half
        c_next = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c_next)
        return h, c_next
