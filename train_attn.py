import argparse
import configparser
from datetime import datetime
import logging
import os
import torch

LOG_NAME = "train"


def train_epoch():


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
    save_dir = config["save"]["save_path"]

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if debug:
        if args.debug:
            logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)  # to stdout
        else:
            dt_now = datetime.now()
            logging.basicConfig(filename=log_dir + LOG_NAME + ,
                                format="%(asctime)s %(message)s",
                                level=logging.DEBUG)
    else:


    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)



if __name__ == "__main__":
    train()
