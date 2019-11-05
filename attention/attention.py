import configparser
import torch
import torch.nn as nn


class ContentBasedAttention(nn.Module):
    def __init__(self, config_path):
        super(ContentBasedAttention, self).__init__()

        config = configparser.ConfigParser()
        config.read(config_path)

        self.hidden_size = int(config["model"]["hidden_size"])

        self.L_se = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.L_he = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.L_fe = nn.Linear(10, self.hidden_size * 2)
        self.L_ee = nn.Linear(self.hidden_size * 2, 1)

        self.F_conv1d = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=100, stride=1, padding=50, bias=False)

    def forward(self, s, h_batch, alpha, attn_mask):
        frames_len = h_batch.shape[1]  # maximum frame length in batch

        # alpha: (batch, 1, frames)
        # conved: (batch, 10, L) L is a length of signal sequence
        conved = self.F_conv1d(alpha)

        # L = frames_len + 2 * padding - (kernel_size - 1) (> frames_len)
        conved = conved.transpose(1, 2)[:, :frames_len, :]  # (batch, frames, 10)

        e = self.L_ee(torch.tanh(self.L_se(s).unsqueeze(1) + self.L_he(h_batch) + self.L_fe(conved)))  # (batch, frames, 1)

        e_max, _ = torch.max(e, dim=1, keepdim=True)

        # avoid exp(too big value) becoming `inf`, then backprop `nan`
        e_cared = torch.exp(e - e_max)

        # mask e whose corresponding frame is zero-padded
        e_cared = e_cared * attn_mask

        alpha = e_cared / torch.sum(e_cared, dim=1, keepdim=True)  # (batch, frames, 1)

        g = torch.sum(alpha * h_batch, dim=1)  # (batch, hidden*2)

        alpha = alpha.transpose(1, 2)  # (batch, 1, frames)

        return g, alpha
