import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.optim import Adam

from transformers import get_cosine_schedule_with_warmup

import cfg
from utils import (
    seed_everything,
    get_logger,
    AverageMeter,
    get_name,
    load_env,
    flag_done,
)
from dataset import MyDataset
from model import MyModel

from tqdm import tqdm

import wandb

logger = get_logger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fcn = cross_entropy


def get_optimizer_params(config, model):
    # differential learning rate and weight decay
    # param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    group1 = ["layer.0.", "layer.1."]
    group2 = ["layer.2.", "layer.3."]
    group3 = ["layer.4.", "layer.5."]
    group_all = [
        "layer.0.",
        "layer.1.",
        "layer.2.",
        "layer.3.",
        "layer.4.",
        "layer.5.",
    ]

    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(nd in n for nd in group_all)
            ],
            "weight_decay_rate": config["wd"],
        },
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)
            ],
            "weight_decay_rate": config["wd"],
            "lr": config["lr"] / 2.6,
        },
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)
            ],
            "weight_decay_rate": config["wd"],
            "lr": config["lr"],
        },
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)
            ],
            "weight_decay_rate": config["wd"],
            "lr": config["lr"] * 2.6,
        },
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(nd in n for nd in group_all)
            ],
            "weight_decay_rate": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)
            ],
            "weight_decay_rate": 0.0,
            "lr": config["lr"] / 2.6,
        },
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)
            ],
            "weight_decay_rate": 0.0,
            "lr": config["lr"],
        },
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)
            ],
            "weight_decay_rate": 0.0,
            "lr": config["lr"] * 2.6,
        },
        {
            "params": [p for n, p in model.named_parameters() if "roberta" not in n],
            "lr": config["head_lr"],
            "momentum": 0.99,
        },
    ]
    return optimizer_parameters


def get_optimizer(config, model):
    return Adam(get_optimizer_params(config, model), lr=config["lr"])


def run(args, is_debug=False):
    """Trains models via cross-validation"""
    seed_everything(cfg.SEED)
    df = pd.read_csv(cfg.TRAIN_FILE)
    le = LabelEncoder().fit(df["language"])

    df["language"] = le.transform(df["language"])

    print(le.classes_)

    config = vars(args)

    # only do one batch on debug
    if is_debug:
        df = df.iloc[:16]
        folds = [(1, (list(range(16)), []))]
    else:
        cvx = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
        folds = enumerate(cvx.split(df), start=1)

    group_name = get_name()

    for fold, (train_idx, val_idx) in folds:
        if not is_debug and cfg.WANDB:
            wandb.init(
                entity="ilnaes",
                group=group_name,
                project="readability",
                config=config,
            )

        logger.info(f"[TRAIN] Starting fold {fold}")

        train_loader = DataLoader(
            MyDataset(
                config,
                df["description"].iloc[train_idx],
                df["language"].iloc[train_idx],
            ),
            batch_size=config["batch_size"],
            drop_last=not is_debug,
            shuffle=True,
        )
        valid_loader = (
            DataLoader(
                MyDataset(
                    config,
                    df["description"].iloc[val_idx],
                    df["language"].iloc[val_idx],
                ),
                batch_size=config["batch_size"],
                drop_last=False,
            )
            if not is_debug
            else None
        )

        model = MyModel(config)
        model = model.to(device)
        opt = get_optimizer(config, model)
        scheduler = None
        scheduler = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=0, num_training_steps=8
        )

        train(config, fold, model, opt, train_loader, valid_loader, scheduler)

        if not is_debug and cfg.WANDB:
            wandb.join()


def train(config, fold, model, optimizer, train, val, scheduler):
    """Trains model on data"""
    model.train()
    max_loss = np.inf
    patience = 0

    epochs = config["epochs"] if val is not None else 1000

    for e in range(epochs):
        logger.info(f"Epoch {e+1}")
        avg = AverageMeter()

        tk = tqdm(train)

        for _, data in enumerate(tk):
            optimizer.zero_grad()

            ids = data["input_ids"]
            masks = data["masks"]
            target = data["target"]

            ids = ids.to(device, dtype=torch.long)
            masks = masks.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.long)

            preds = model(input_ids=ids, attention_mask=masks)
            loss = loss_fcn(preds.logits, target)

            acc = torch.mean((torch.argmax(preds.logits, dim=1) == target).float())
            avg.update(acc.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            tk.set_postfix({"train_loss": np.sqrt(avg.avg)})

        # only validate when not debugging
        if val is not None:
            val_loss, _ = eval(model, val)
        else:
            val_loss = "DEBUG"

        logger.info(
            f"Epoch {e+1}/{epochs} -- Validation loss: {val_loss}\t Train loss: {np.sqrt(avg.avg)}"
        )

        if val is not None and cfg.WANDB:
            wandb.log(
                {
                    "train_loss": np.sqrt(avg.avg),
                    "val_loss": val_loss,
                }
            )

        if scheduler is not None:
            scheduler.step()

        if val is not None:
            if val_loss < max_loss:
                patience, max_loss = 0, val_loss
                torch.save(model.state_dict(), f"{cfg.MODEL_SAVE_DIR}model_{fold}.pt")
            else:
                patience += 1
                if cfg.PATIENCE != 0 and patience > cfg.PATIENCE:
                    break


def eval(model, val):
    """Evaluates model on validation data and computes loss"""
    model.eval()
    total = 0

    with torch.no_grad():
        tk = tqdm(val)
        avg = AverageMeter()

        for i, data in enumerate(tqdm(val)):
            ids = data["input_ids"]
            masks = data["masks"]
            target = data["target"]

            ids = ids.to(device, dtype=torch.long)
            masks = masks.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.long)

            preds = model(input_ids=ids, attention_mask=masks)
            loss = loss_fcn(preds.logits, target)

            acc = torch.mean((torch.argmax(preds.logits, dim=1) == target).float())
            avg.update(acc.detach().cpu().numpy())
            # avg.update(loss.detach().cpu().numpy())

            total += loss.detach().cpu()

            tk.set_postfix({"val_loss": np.sqrt(avg.avg)})

    return torch.sqrt(total / len(val)), None


def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Basic arguments
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--head_lr", type=float, default=cfg.HEAD_LR)
    parser.add_argument("--wd", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)
    # parser.add_argument("--patience", type=int, default=cfg.PATIENCE)

    MyModel.add_to_argparse(parser)
    MyDataset.add_to_argparse(parser)

    return parser


def main():
    if device == "cuda":
        torch.cuda.empty_cache()

    if cfg.WANDB:
        load_env()

    parser = setup_parser()
    args = parser.parse_args()

    run(args, cfg.DEBUG)

    flag_done()


if __name__ == "__main__":
    main()
