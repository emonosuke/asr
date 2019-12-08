import argparse
import configparser
import torch
from attention.attn_model import AttnModel
from dataset import SpeechDataset
from vocab import Vocab
from utils import load_htk, subword_to_word
from frontend import frame_stacking

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="params.sw.conf")
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    model_path = args.model_path
    state_dict = torch.load(model_path, map_location=device)

    config = configparser.ConfigParser()
    config.read(config_path)

    model = AttnModel(config_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    script_path = config["data"]["eval_script"]
    lmfb_dim = int(config["frontend"]["lmfb_dim"])
    num_framestack = int(config["frontend"]["num_framestack"])
    vocab_path = config["vocab"]["vocab_path"]

    vocab = Vocab(vocab_path=vocab_path)

    with open(script_path) as f:
        xpaths = [line.strip() for line in f]

    for xpath in xpaths:
        x = load_htk(xpath)[:, :lmfb_dim]

        if num_framestack > 1:
            x = frame_stacking(x, num_framestack)

        x_tensor = torch.tensor(x).to(device)
        seq_len = x.shape[0]
        seq_lens = torch.tensor([seq_len]).to(device)

        res = model.decode(x_tensor.unsqueeze(0), seq_lens)
        res_subword = vocab.ids2word(res)

        res_word = subword_to_word(res_subword)

        print(xpath, " ".join(res_word), flush=True)


if __name__ == "__main__":
    eval()
