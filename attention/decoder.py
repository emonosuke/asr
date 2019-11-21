import configparser
from operator import itemgetter
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from attention.attention import ContentBasedAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INF_MIN = -1e10


class Decoder(nn.Module):
    def __init__(self, config_path):
        super(Decoder, self).__init__()

        config = configparser.ConfigParser()
        config.read(config_path)

        self.hidden_size = int(config["model"]["hidden_size"])
        self.vocab_size = int(config["vocab"]["vocab_size"])
        self.beam_width = int(config["test"]["beam_width"])
        self.max_seq_len = int(config["test"]["max_seq_len"])

        self.eos_id = int(config["vocab"]["eos_id"])

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
    
    def decode(self, h_batch, seq_lens):
        batch_size = h_batch.shape[0]
        assert batch_size == 1

        frames_len = h_batch.shape[1]

        # sequence, score, (cell state, hidden state, attention weight)
        beam_paths = [([], 0.0,
                       (torch.zeros((batch_size, self.hidden_size), device=DEVICE, requires_grad=False),
                        torch.zeros((batch_size, self.hidden_size), device=DEVICE, requires_grad=False),
                        torch.zeros((batch_size, 1, frames_len), device=DEVICE, requires_grad=False)))]

        attn_mask = torch.ones((batch_size, frames_len, 1), device=DEVICE, requires_grad=False)
        for b, seq_len in enumerate(seq_lens):
            if b < seq_len:
                attn_mask.data[b, seq_len:] = 0.0
        
        for _ in range(self.max_seq_len):
            current_beam_paths = []

            for beam_path in beam_paths:
                cand_seq, cand_score, (c, s, alpha) = beam_path
                g, alpha = self.attn(s, h_batch, alpha, attn_mask)

                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
                y = log_softmax(y, dim=1)

                y_c = y.clone()
                for _ in range(self.beam_width):
                    best_idx = y_c.data.argmax(1).item()

                    new_seq = cand_seq + [best_idx]
                    new_score = cand_score + y_c.data[0][best_idx]

                    y_c.data[0][best_idx] = INF_MIN  # this enable to pick up 2nd, 3rd ... best words

                    best_idx_tensor = torch.tensor([best_idx])
                    rec_in = self.L_yr(best_idx_tensor) + self.L_sr(s) + self.L_gr(g)
                    new_s, new_c = self._func_lstm(rec_in, c)

                    current_beam_paths.append((new_seq, new_score, (new_c, new_s, alpha)))

            # sort by its score
            current_beam_paths_sorted = sorted(current_beam_paths, key=itemgetter(1), reverse=True)

            beam_paths = current_beam_paths_sorted[:self.beam_width]

            res = []
            # if top candidate end with <eos>, finish decoding
            if beam_paths[0][0][-1] == self.eos_id:
                for char in beam_paths[0][0]:
                    res.append(char)
                break

        return res

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
