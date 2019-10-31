import configparser
from torch.utils.data import DataLoader

import sys
sys.path.append("../")

from dataset import SpeechDataset, collate_fn_train, collate_fn_eval

config = configparser.ConfigParser()
config.read("../params.conf")

script_path = config["data"]["train_script"]
dataset = SpeechDataset(script_path)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_train)

for data in dataloader:
    print(data)
    print(data["x_batch"].shape, data["seq_lens"].shape, data["labels"].shape)

    break

eval_path = config["data"]["eval_script"]
dataset_eval = SpeechDataset(eval_path, no_label=True)

dataloader = DataLoader(dataset=dataset_eval, batch_size=4, shuffle=False, collate_fn=collate_fn_eval)

for data in dataloader:
    print(data)
    print(data["x_batch"].shape, data["seq_lens"].shape)

    break
