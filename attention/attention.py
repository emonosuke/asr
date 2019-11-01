import configparser
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
        frames_len = h_batch.shape[1]
        conved = self.F_conv1d(alpha)  # (batch, 10, channel)




        return g, alpha

