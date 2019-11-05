import torch
import torch.nn as nn
from attention.encoder import Encoder
from attention.decoder import Decoder


class AttnModel(nn.Module):
    def __init__(self, config_path):
        super(AttnModel, self).__init__()
        self.encoder = Encoder(config_path)
        self.decoder = Decoder(config_path)

    def forward(self, x_batch, seq_lens, onehot):
        h_batch = self.encoder(x_batch, seq_lens)
        preds = self.decoder(h_batch, seq_lens, onehot)

        return preds

    def decode(self, x, seq_lens):
        with torch.no_grad():
            h_batch = self.encoder(x, seq_lens)
            preds = self.decoder.decode(h_batch, seq_lens)

        return preds
