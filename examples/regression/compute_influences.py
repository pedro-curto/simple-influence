import os
from typing import List

import torch

from examples.regression.pipeline import (
    construct_regression_mlp,
    get_hyperparameters,
    get_loaders,
)
from examples.regression.task import RegressionTask
from src.gradient_similarity import GradientSimilarityComputer
from src.influence_function import InfluenceFunctionComputer
from src.representation_similarity import RepresentationSimilarityComputer
from src.tracin import TracinComputer

BASE_PATH = "files/results"


def prepare_everything(data_name: str, model_id: int = 0):
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{model_id}", exist_ok=True)

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name,
        train_indices=None,
        data_path="data",
    )

    model = construct_regression_mlp(data_name=data_name)
    model.load_state_dict(
        torch.load(
            f"files/checkpoints/{model_id}/{data_name}_epoch_20.pt",
            map_location="cpu",
        )
    )
    model.eval()

    task = RegressionTask()
    return model, eval_train_loader, valid_loader, task


def get_single_trajectory_checkpoints(
    data_name: str,
    model_id: int = 0,
):
    epoch_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert len(epoch_list) == 10

    checkpoints = []
    for epoch in epoch_list:
        checkpoints.append(f"files/checkpoints/{model_id}/{data_name}_epoch_{epoch}.pt")
    return checkpoints


def compute_reps_similarity(data_name: str, model_ids: List[int]) -> None:
    for mid in model_ids:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name,
            model_id=mid,
        )

        computer = RepresentationSimilarityComputer(
            model=model,
            task=task,
            metric="l2",
        )
        scores = computer.compute_scores_with_loader(
            test_loader=valid_loader, train_loader=eval_train_loader
        )
        expt_name = f"{data_name}_representations_similarity_l2"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")

        computer = RepresentationSimilarityComputer(
            model=model,
            task=task,
            metric="cos",
        )
        scores = computer.compute_scores_with_loader(
            test_loader=valid_loader, train_loader=eval_train_loader
        )
        expt_name = f"{data_name}_representations_similarity_cos"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")


def compute_grads_similarity(
    data_name: str,
    model_ids: List[int],
) -> None:
    for mid in model_ids:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name, model_id=mid
        )

        computer = GradientSimilarityComputer(
            model=model,
            task=task,
            metric="dot",
        )
        scores = computer.compute_scores_with_loader(
            test_loader=valid_loader, train_loader=eval_train_loader
        )
        expt_name = f"{data_name}_gradients_similarity_dot"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")

        computer = GradientSimilarityComputer(
            model=model,
            task=task,
            metric="cos",
        )
        scores = computer.compute_scores_with_loader(
            test_loader=valid_loader, train_loader=eval_train_loader
        )
        expt_name = f"{data_name}_gradients_similarity_cos"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")


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


def compute_tracin(data_name: str, model_ids: List[int]) -> None:
    hyper_dict = get_hyperparameters(data_name)
    lr = hyper_dict["lr"]

    for mid in model_ids:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name,
            model_id=mid,
        )
        checkpoints = get_single_trajectory_checkpoints(
            data_name=data_name,
            model_id=mid,
        )

        computer = TracinComputer(model, task=task, metric="dot")
        scores = computer.compute_scores_with_loader(
            checkpoints=checkpoints,
            lrs=lr,
            test_loader=valid_loader,
            train_loader=eval_train_loader,
        )
        expt_name = f"{data_name}_tracin"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")

        computer = TracinComputer(model, task=task, metric="cos")
        scores = computer.compute_scores_with_loader(
            checkpoints=checkpoints,
            lrs=lr,
            test_loader=valid_loader,
            train_loader=eval_train_loader,
        )
        expt_name = f"{data_name}_gas"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")


if __name__ == "__main__":
    data_names = ["concrete"]
    model_ids = [0]

    for dn in data_names:
        compute_reps_similarity(data_name=dn, model_ids=model_ids)
        compute_grads_similarity(data_name=dn, model_ids=model_ids)
        compute_tracin(data_name=dn, model_ids=model_ids)
        compute_if(data_name=dn, model_ids=model_ids)
