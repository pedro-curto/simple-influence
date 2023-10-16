import os
from typing import List

import torch

from examples.wiki.task import LanguageModelTask
from examples.wiki.pipeline import (
    construct_model,
    get_loaders,
)
from src.influence_function import InfluenceFunctionComputer

BASE_PATH = "files/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_everything(model_id: int = 0):
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{model_id}", exist_ok=True)

    _, eval_train_loader, valid_loader = get_loaders(
        eval_batch_size=4,
        train_indices=None,
        # Compute influences on the first 32 validation points.
        valid_indices=list(range(32)),
    )

    model = construct_model()
    model.load_state_dict(
        torch.load(
            f"files/checkpoints/{model_id}/wiki_epoch_3.pt",
            map_location="cpu",
        )
    )
    model.eval()

    task = LanguageModelTask(device=DEVICE)
    return model.to(DEVICE), eval_train_loader, valid_loader, task


def compute_if(model_ids: List[int]) -> None:
    for mid in model_ids:
        model, eval_train_loader, valid_loader, task = prepare_everything(
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
        expt_name = f"wiki_if"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")


if __name__ == "__main__":
    model_ids = [0]
    compute_if(model_ids=model_ids)
