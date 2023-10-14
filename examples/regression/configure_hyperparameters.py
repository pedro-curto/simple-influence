from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import SGD

from examples.regression.pipeline import construct_regression_mlp, get_loaders
from examples.regression.train import evaluate


def train_with_eval(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    lr: float,
    epochs: int,
    weight_decay: float,
) -> Tuple[List[float], List[float]]:
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = MSELoss(reduction="mean")

    train_loss_lst = []
    valid_loss_lst = []
    train_loss_lst.append(evaluate(model, eval_train_loader))
    valid_loss_lst.append(evaluate(model, valid_loader))
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss_lst.append(evaluate(model, eval_train_loader))
        valid_loss_lst.append(evaluate(model, valid_loader))

    return train_loss_lst, valid_loss_lst


def main(data_name: str, target_epochs: int = 20) -> None:
    lr_lst = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    wd_lst = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.0]

    train_loader, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name, eval_batch_size=512, data_path="data"
    )

    best_loss = None
    best_lr = None
    best_wd = None
    for lr in lr_lst:
        for wd in wd_lst:
            total_losses = []
            for seed in range(5):
                model = construct_regression_mlp(data_name=data_name)
                _, valid_losses = train_with_eval(
                    model=model,
                    train_loader=train_loader,
                    eval_train_loader=eval_train_loader,
                    valid_loader=valid_loader,
                    lr=lr,
                    weight_decay=wd,
                    epochs=target_epochs,
                )
                total_losses.append(valid_losses[-1])

            final_loss = np.mean(np.array(total_losses))
            if np.isnan(final_loss):
                final_loss = np.Inf

            if best_loss is None or final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr
                best_wd = wd

    print(f"Dataset: {data_name}")
    print(f"Best LR, WD: {best_lr}, {best_wd}")

    model = construct_regression_mlp(data_name=data_name)
    train_losses, valid_losses = train_with_eval(
        model=model,
        train_loader=train_loader,
        eval_train_loader=eval_train_loader,
        valid_loader=valid_loader,
        lr=best_lr,
        weight_decay=best_wd,
        epochs=100,
    )
    plt.plot(range(101), train_losses, label="Train")
    plt.plot(range(101), valid_losses, label="Valid")
    plt.title(f"{data_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Best LR, WD: 0.03, 1e-05.
    main(data_name="concrete", target_epochs=20)
    # Best LR, WD: 0.003, 1e-05
    main(data_name="parkinsons", target_epochs=20)
