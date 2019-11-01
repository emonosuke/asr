import argparse
import configparser
from datetime import datetime
import logging
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from attention.attn_model import AttnModel
from dataset import SpeechDataset, collate_fn_train
from loss import label_smoothing_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model, optimizer, data):
    x_batch = data["x_batch"]
    seq_lens = data["seq_lens"]
    labels = data["labels"]
    lab_lens = data["lab_lens"]

    optimizer.zero_grad()

    preds = model(x_batch, seq_lens, labels)

    loss = label_smoothing_loss(preds, labels, lab_lens)

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    loss.backward()
    optimizer.step()

    # torch.cuda.empty_cache()

    return loss.item()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="params.conf")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_path = args.config_path
    debug = args.debug

    # load configs
    config = configparser.ConfigParser()
    config.read(config_path)
    log_dir = config["log"]["log_path"]
    log_step = int(config["log"]["log_step"])
    save_dir = config["save"]["save_path"]
    save_step = int(config["save"]["save_step"])
    num_epochs = int(config["train"]["num_epochs"])
    batch_size = int(config["train"]["batch_size"])

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    dt_now = datetime.now()
    dt_str = dt_now.strftime("%m%d%H%M%S")

    if debug:
        logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)  # to stdout
    else:
        log_path = log_dir + "train_attn_{}.log".format(dt_str)

        logging.basicConfig(filename=log_path,
                            format="%(asctime)s %(message)s",
                            level=logging.DEBUG)

    logging.info("process id: {:d} is allocated".format(os.getpid()))

    model = AttnModel(config_path)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

    dataset = SpeechDataset(config_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)

    num_steps = len(dataset)

    for epoch in range(num_epochs):
        loss_sum = 0

        for step, data in enumerate(dataloader):
            loss_sum += train_step(model, optimizer, data)

            if (step + 1) % log_step == 0:
                logging.info("epoch = {} step = {} / {} loss = {:.3f}".format(epoch + 1,
                                                                              step + 1,
                                                                              num_steps,
                                                                              loss_sum / log_step))
        if epoch == 0 or (epoch + 1) % save_step == 0:
            save_path = save_dir + "attention{}.epoch{}".format(dt_str, epoch + 1)
            torch.save(model.state_dict(), save_path)
            torch.save(optimizer.state_dict(), save_path)


if __name__ == "__main__":
    train()
