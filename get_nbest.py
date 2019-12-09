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
    parser.add_argument("model_path", type=str)

    # train: `/n/rd35/futami/data/aps/script.word.path`
    # eval: `/n/rd35/futami/data/eval/script.official.eval1`
    parser.add_argument("script_path", type=str)
    parser.add_argument("--config_path", type=str, default="params.sw.conf")
    parser.add_argument("--num_best", type=int, default=3)
    args = parser.parse_args()

    config_path = args.config_path
    model_path = args.model_path
    script_path = args.script_path
    num_best = args.num_best
    state_dict = torch.load(model_path, map_location=device)

    config = configparser.ConfigParser()
    config.read(config_path)

    model = AttnModel(config_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

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

        hyps = model.decode_nbest(x_tensor.unsqueeze(0), seq_lens, num_best=num_best)

        print(xpath)
        for hyp in hyps:
            seq, score = hyp

            # seq_str = " ".join(list(map(str, seq)))
            seq_str = " ".join(vocab.ids2word(seq))

            print("{:.4f} {}".format(score, seq_str))
        
        break


if __name__ == "__main__":
    eval()
