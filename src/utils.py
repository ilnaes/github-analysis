import os
import random

import torch
import numpy as np
from logging import Formatter, StreamHandler, getLogger

from datetime import datetime
import pytz


def get_name():
    tz = pytz.timezone("US/Eastern")
    return datetime.now(tz).strftime("%m%d%-y-%H%M%S")


def load_env():
    with open("/content/drive/MyDrive/Google Drive sync/.env") as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            os.environ[key] = val


def flag_done(message):
    time = get_name()
    with open("/content/drive/MyDrive/Google Drive sync/watch/" + time, "w") as f:
        f.write(message)


def get_logger():
    log_fmt = Formatter("%(asctime)s [%(levelname)s][%(funcName)s] %(message)s ")
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.setLevel("INFO")
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
