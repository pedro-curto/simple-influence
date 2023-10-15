import math
import os
import time
from typing import Optional, Tuple

import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss

from examples.glue.pipeline import construct_model, get_loaders
from examples.utils import set_seed


def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    model_id: int = 0,
    lr: float = 2e-5,
    weight_decay: float = 0.0,
    save_name: Optional[str] = None,
) -> nn.Module:
    save = save_name is not None
    if save:
        os.makedirs(f"checkpoints/{model_id}", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"checkpoints/{model_id}/{save_name}_iter_0.pt",
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss()
    # For details, see `configure_hp.py`.
    epochs = 3

    num_update_steps_per_epoch = math.ceil(len(loader))
    accelerator = Accelerator()
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    assert math.ceil(len(loader)) == num_update_steps_per_epoch

    model.train()
    num_iter = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]
            )
            loss = loss_fn(outputs, batch["labels"])
            loss.backward()
            optimizer.step()
            num_iter += 1

            if save and num_iter % 375 == 0:
                # This should yield 25 checkpoints.
                torch.save(
                    model.state_dict(),
                    f"checkpoints/{model_id}/{save_name}_iter_{num_iter}.pt",
                )
    return model


def model_evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[float, float]:
    model.eval()
    accelerator = Accelerator()
    loader = accelerator.prepare(loader)
    # Task name does not really matter here.
    metric = evaluate.load("glue", "qnli")
    total_loss, total_num = 0.0, 0.0
    for step, batch in enumerate(loader):
        with torch.no_grad():
            outputs = model(
                batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]
            )
        total_loss += (
            F.cross_entropy(outputs, batch["labels"], reduction="sum").cpu().item()
        )
        total_num += batch["input_ids"].shape[0]
        predictions = outputs.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )
    eval_metric = metric.compute()
    return total_loss / total_num, eval_metric["accuracy"]


def get_hyperparameters(data_name: str) -> Tuple[float, float]:
    # For details, see `configure_hp.py`.
    del data_name
    lr = 2e-5
    wd = 0.01
    return lr, wd


def main(
    data_name: str = "qnli", num_train: int = 50, do_corrupt: bool = False
) -> None:
    init_pytorch()
    os.makedirs("../checkpoints", exist_ok=True)

    train_loader, _, valid_loader = get_loaders(
        data_name=data_name,
        do_corrupt=do_corrupt,
    )
    lr, wd = get_hyperparameters(data_name)

    save_name = data_name
    if do_corrupt:
        save_name += "_corrupted"

    for i in range(num_train):
        print(f"Training {i}th model ...")
        start_time = time.time()

        set_seed(i)
        model = construct_model(data_name=data_name)

        train(
            model=model,
            loader=train_loader,
            lr=lr,
            weight_decay=wd,
            model_id=i,
            save_name=save_name,
        )

        _, valid_acc = model_evaluate(model=model, loader=valid_loader)
        print(f"Validation Accuracy: {valid_acc}")
        del model

        print(f"Took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    import sys

    try:
        option = int(sys.argv[1])
        if option == 1:
            main(data_name="qnli", do_corrupt=False, num_train=50)
            main(data_name="qnli", do_corrupt=True, num_train=50)
        elif option == 2:
            main(data_name="sst2", do_corrupt=False, num_train=50)
            main(data_name="sst2", do_corrupt=True, num_train=50)
        else:
            raise NotImplementedError(f"Not available option {option}.")

    except IndexError:
        print("Debug mode...")
        # main(data_name="sst2", do_corrupt=False, num_train=1)
        main(data_name="qnli", do_corrupt=False, num_train=1)
