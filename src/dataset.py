import re
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from . import cfg

classes = ["r", "java", "javascript", "c", "python", "js"]


def process_str(s):
    if type(s) is str:
        return re.sub(r"[^\x00-\x7F]+", " ", s)
    else:
        return "MISSING"


class MyDataset(Dataset):
    def __init__(self, config, data, target=None):
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        # strip non-ASCII
        data = data.map(process_str)

        # let's making this more interesting
        # excerpts = [" ".join(x.split()) for x in data]
        excerpts = [" ".join([x for x in s.split() if x.lower() not in classes]) for s in data]
        self.lengths = np.array([len(x.split()) for x in data])
        self.max_len = max(self.lengths)

        self.data = tokenizer(
            excerpts,
            return_tensors="pt",
            padding="max_length",
            max_length=config["max_len"],
            truncation=True,
        )

        if target is not None:
            self.target = torch.tensor(target.values, dtype=torch.float)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--max_len", type=int, default=cfg.MAX_LEN)
        return parser

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        res = {
            "input_ids": self.data["input_ids"][idx],
            "masks": self.data["attention_mask"][idx],
            "lengths": self.lengths[idx] / self.max_len,
        }
        if hasattr(self, "target"):
            res["target"] = self.target[idx]

        return res
