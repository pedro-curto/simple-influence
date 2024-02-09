import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

from examples.utils import clear_gpu_cache, set_seed
from examples.wiki.pipeline import construct_model, get_loaders


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
        os.makedirs(f"files/checkpoints/{model_id}", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"files/checkpoints/{model_id}/{save_name}_epoch_0.pt",
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss()
    epochs = 3

    num_update_steps_per_epoch = math.ceil(len(loader))
    accelerator = Accelerator()
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    assert math.ceil(len(loader)) == num_update_steps_per_epoch

    model.train()
    for epoch in trange(1, epochs + 1):
        for step, batch in tqdm(enumerate(loader), total=num_update_steps_per_epoch):
            optimizer.zero_grad()
            lm_logits = model(batch["input_ids"], batch["attention_mask"])
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss.backward()
            optimizer.step()

        if save:
            torch.save(
                model.state_dict(),
                f"files/checkpoints/{model_id}/{save_name}_epoch_{epoch}.pt",
            )
    return model


def model_evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    accelerator = Accelerator()
    loss_fn = CrossEntropyLoss(reduction="sum")
    loader = accelerator.prepare(loader)
    total_loss, total_num = 0.0, 0
    for step, batch in enumerate(loader):
        with torch.no_grad():
            lm_logits = model(batch["input_ids"], batch["attention_mask"])
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            loss = (
                loss_fn(reshaped_shift_logits, shift_labels.view(-1)).detach().float()
            )
            total_loss += loss
        total_num += reshaped_shift_logits.shape[0]
    return total_loss.item() / total_num


def main(
    num_train: int = 1,
) -> None:
    os.makedirs("files/", exist_ok=True)
    os.makedirs("files/checkpoints", exist_ok=True)

    train_loader, _, valid_loader = get_loaders()

    save_name = "wiki"
    for i in range(num_train):
        print(f"Training {i}th model ...")
        start_time = time.time()

        set_seed(i)
        model = construct_model()

        train(
            model=model,
            loader=train_loader,
            model_id=i,
            save_name=save_name,
        )

        loss = model_evaluate(model=model, loader=valid_loader)
        print(f"Validation Loss: {loss}")
        print(f"Validation Perplexity: {math.exp(loss)}")
        del model
        clear_gpu_cache()
        print(f"Took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    main(num_train=1)
