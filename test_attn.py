import argparse
import configparser
import random
import torch
from attention.attn_model import AttnModel
from dataset import SpeechDataset
from vocab import Vocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="params.conf")
    parser.add_argument("--data_id", type=int, default=-1)
    # parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    data_id = args.data_id
    # model_path = args.model_path
    # state_dict = torch.load(model_path, map_location=DEVICE)

    config = configparser.ConfigParser()
    config.read(config_path)

    vocab_path = config["vocab"]["vocab_path"]
    vocab = Vocab(vocab_path=vocab_path)

    model = AttnModel(config_path)
    # model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    dataset = SpeechDataset(config_path)

    if data_id == -1:
        data_id = random.randint(0, len(dataset))
    print("data_id = {}".format(data_id))
    x_tensor, seq_len, label, _ = dataset[data_id]

    seq_lens = torch.tensor([seq_len])

    print("ground:  {}".format(" ".join(vocab.ids2word(label.numpy()))))
    res = model.decode(x_tensor.unsqueeze(0), seq_lens)
    print(res)
    print("predict: {}".format(" ".join(vocab.ids2word(res))))


if __name__ == "__main__":
    test()
