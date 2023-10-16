import os
from typing import List

import torch

from examples.glue.task import TextClassificationTask
from examples.glue.pipeline import (
    construct_model,
    get_loaders,
)
from src.influence_function import InfluenceFunctionComputer

BASE_PATH = "files/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_everything(data_name: str = "qnli", model_id: int = 0):
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{model_id}", exist_ok=True)

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name,
        eval_batch_size=32,
        train_indices=None,
        # Compute influences on the first 128 validation points.
        valid_indices=list(range(128)),
    )

    model = construct_model(data_name=data_name)
    model.load_state_dict(
        torch.load(
            f"files/checkpoints/{model_id}/{data_name}_iter_9375.pt",
            map_location="cpu",
        )
    )
    model.eval()

    task = TextClassificationTask(device=DEVICE)
    return model.to(DEVICE), eval_train_loader, valid_loader, task


def compute_if(data_name: str, model_ids: List[int]) -> None:
    for mid in model_ids:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name,
            model_id=mid,
        )

        ekfac = InfluenceFunctionComputer(
            model=model,
            task=task,
            n_epoch=1,
        )
        ekfac.build_curvature_blocks(eval_train_loader)
        scores = ekfac.compute_scores_with_loader(
            test_loader=valid_loader, train_loader=eval_train_loader
        )
        expt_name = f"{data_name}_if"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")


if __name__ == "__main__":
    model_ids = [0]
    compute_if(data_name="qnli", model_ids=model_ids)
    compute_if(data_name="sst2", model_ids=model_ids)
