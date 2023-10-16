import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import SGD

from examples.regression.pipeline import (
    construct_regression_mlp,
    get_hyperparameters,
    get_loaders,
)
from examples.utils import set_seed


def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    lr: float,
    weight_decay: float,
    model_id: int = 0,
    save_name: Optional[str] = None,
) -> nn.Module:
    save = save_name is not None
    if save:
        os.makedirs(f"files/checkpoints/{model_id}", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"files/checkpoints/{model_id}/{save_name}_epoch_0.pt",
        )
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = MSELoss(reduction="mean")
    epochs = 20

    model.train()
    for epoch in range(1, epochs + 1):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        if save:
            torch.save(
                model.state_dict(),
                f"files/checkpoints/{model_id}/{save_name}_epoch_{epoch}.pt",
            )
    return model


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    with torch.no_grad():
        total_loss, total_num = 0.0, 0
        for inputs, targets in loader:
            out = model(inputs)
            total_loss += F.mse_loss(out, targets, reduction="sum").item()
            total_num += inputs.shape[0]
    avg_loss = total_loss / total_num
    return avg_loss


def main(data_name: str, num_train: int = 50) -> None:
    os.makedirs("../files", exist_ok=True)
    os.makedirs("../files/checkpoints", exist_ok=True)

    train_loader, _, valid_loader = get_loaders(
        data_name=data_name,
        eval_batch_size=512,
    )
    hyper_dict = get_hyperparameters(data_name)
    lr = hyper_dict["lr"]
    wd = hyper_dict["wd"]

    save_name = data_name
    for i in range(num_train):
        print(f"Training {i}th model ...")
        start_time = time.time()

        set_seed(i)
        model = construct_regression_mlp(data_name=data_name)

        train(
            model=model,
            loader=train_loader,
            lr=lr,
            weight_decay=wd,
            model_id=i,
            save_name=save_name,
        )
        del model

        print(f"Took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    data_names = ["concrete"]
    for dn in data_names:
        main(data_name=dn, num_train=1)
