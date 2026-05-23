"""Compute training-data-attribution scores for the MNIST example.

After training with `train.py` (which writes a checkpoint per epoch to
`files/checkpoints/{model_id}/mnist_epoch_*.pt`), this script computes
attribution scores using one or more methods and saves them under
`files/results/{model_id}/`.

Run:
    python compute_influences.py
"""

import math
import os
from typing import List, Optional, Tuple

import torch

from examples.mnist.pipeline import construct_mlp, get_hyperparameters, get_loaders
from examples.mnist.task import ClassificationTask
from src.gradient_similarity import GradientSimilarityComputer
from src.influence_function import InfluenceFunctionComputer
from src.representation_similarity import RepresentationSimilarityComputer
from src.source import SourceComputer

BASE_PATH = "files/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training settings replicated from `train.py`. Used to derive the SOURCE
# per-segment hyperparameters (iterations and effective learning rate).
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 512
TRAIN_SET_SIZE = 60_000
SGD_MOMENTUM = 0.9


def prepare_everything(data_name: str, model_id: int = 0):
    """Load the trained model + evaluation/training loaders + task.

    Args:
        data_name (str):
            ``"mnist"`` or ``"fmnist"``.
        model_id (int, optional):
            Index of the trained model directory under ``files/checkpoints/``.
            Defaults to ``0``.

    Returns:
        Tuple of ``(model, eval_train_loader, valid_loader, task)``, with the
        model already moved to ``DEVICE``.
    """
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{model_id}", exist_ok=True)

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name, train_indices=None, valid_indices=list(range(16))
    )

    model = construct_mlp()
    model.load_state_dict(
        torch.load(
            f"files/checkpoints/{model_id}/{data_name}_epoch_{TRAIN_EPOCHS}.pt",
            map_location="cpu",
        )
    )
    model.eval()

    task = ClassificationTask(device=DEVICE)
    return model.to(DEVICE), eval_train_loader, valid_loader, task


def compute_reps_similarity(data_name: str, model_ids: List[int]) -> None:
    """Compute representation-similarity scores for both `dot` and `cos` metrics."""
    for mid in model_ids:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name,
            model_id=mid,
        )

        for metric in ("l2", "cos"):
            computer = RepresentationSimilarityComputer(
                model=model,
                task=task,
                metric=metric,
            )
            scores = computer.compute_scores_with_loader(
                test_loader=valid_loader, train_loader=eval_train_loader
            )
            expt_name = f"{data_name}_representations_similarity_{metric}"
            torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")


def compute_grads_similarity(data_name: str, model_ids: List[int]) -> None:
    """Compute gradient-similarity scores for both `dot` and `cos` metrics."""
    for mid in model_ids:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name, model_id=mid
        )

        for metric in ("dot", "cos"):
            computer = GradientSimilarityComputer(
                model=model,
                task=task,
                metric=metric,
            )
            scores = computer.compute_scores_with_loader(
                test_loader=valid_loader, train_loader=eval_train_loader
            )
            expt_name = f"{data_name}_gradients_similarity_{metric}"
            torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")


def compute_if(data_name: str, model_ids: List[int]) -> None:
    """Compute influence-function scores via EK-FAC."""
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


def _build_source_segments(
    data_name: str,
    model_id: int,
    num_segments: int,
    epochs_per_segment: List[List[int]],
) -> Tuple[List[List[str]], List[int], List[float]]:
    """Build per-segment checkpoint / iteration / learning-rate triples.

    Splits the training trajectory equally into ``num_segments`` chronological
    chunks. For each segment we pick the checkpoints at the END of each
    listed epoch within the segment (so the checkpoint "represents" the
    parameters at that point in training).

    Args:
        data_name (str):
            ``"mnist"`` or ``"fmnist"``.
        model_id (int):
            Index of the trained model directory.
        num_segments (int):
            Number of segments ``L`` to partition into.
        epochs_per_segment (List[int]):
            For each segment, the list of epoch numbers (1-indexed, matching
            the names produced by ``train.py``) whose checkpoints will be
            averaged for that segment.

    Returns:
        ``(checkpoints_per_segment, iters_per_segment, lrs_per_segment)`` as
        accepted by :class:`SourceComputer`.
    """
    hyper = get_hyperparameters(data_name)
    base_lr = hyper["lr"]
    # SGD with heavy-ball momentum: paper Appendix D, paragraph below Eq. 54
    # ("we scaled the learning rate used in Source as eta * (1 - beta)^-1
    #  to account for the effective learning rate (terminal velocity)").
    effective_lr = base_lr / (1.0 - SGD_MOMENTUM)

    # Iterations per epoch (mirroring `train.py`: shuffle=True, drop_last=True).
    iters_per_epoch = TRAIN_SET_SIZE // TRAIN_BATCH_SIZE

    assert (
        len(epochs_per_segment) == num_segments
    ), "epochs_per_segment must have one inner list per segment."

    checkpoints_per_segment: List[List[str]] = []
    iters_per_segment: List[int] = []
    lrs_per_segment: List[float] = []
    boundaries = _equal_segment_boundaries(TRAIN_EPOCHS, num_segments)

    for seg_idx, ckpt_epochs in enumerate(epochs_per_segment):
        ckpt_paths = [
            f"files/checkpoints/{model_id}/{data_name}_epoch_{e}.pt"
            for e in ckpt_epochs
        ]
        # K_l = number of gradient updates in this segment.
        k_l = (boundaries[seg_idx + 1] - boundaries[seg_idx]) * iters_per_epoch
        checkpoints_per_segment.append(ckpt_paths)
        iters_per_segment.append(k_l)
        lrs_per_segment.append(effective_lr)
    return checkpoints_per_segment, iters_per_segment, lrs_per_segment


def _equal_segment_boundaries(total_epochs: int, num_segments: int) -> List[int]:
    """Return ``L+1`` integer epoch boundaries that split training into L chunks.

    The first boundary is always 0 and the last is ``total_epochs``; the
    interior boundaries are spread as evenly as possible.

    Args:
        total_epochs (int): total number of training epochs.
        num_segments (int): ``L``.

    Returns:
        List[int]: sorted boundaries of length ``num_segments + 1``.
    """
    return [
        int(math.floor(total_epochs * i / num_segments))
        for i in range(num_segments + 1)
    ]


def compute_source(
    data_name: str,
    model_ids: List[int],
    num_segments: int = 3,
    epochs_per_segment: Optional[List[List[int]]] = None,
) -> None:
    """Compute SOURCE attribution scores.

    Args:
        data_name (str):
            ``"mnist"`` or ``"fmnist"``.
        model_ids (List[int]):
            Trained model directories to compute scores for.
        num_segments (int, optional):
            Number of segments ``L``. Defaults to ``3`` (paper default).
        epochs_per_segment (List[List[int]], optional):
            Per-segment list of checkpoint epoch numbers. If ``None``, picks
            two evenly spaced checkpoints in each segment (paper default of
            6 total checkpoints when ``num_segments=3``).
    """
    boundaries = _equal_segment_boundaries(TRAIN_EPOCHS, num_segments)
    if epochs_per_segment is None:
        # Two evenly spaced checkpoints per segment (paper default with
        # `num_segments=3` -> 6 checkpoints total).
        epochs_per_segment = []
        for i in range(num_segments):
            start, end = boundaries[i], boundaries[i + 1]
            # Pick the 1/3 and 2/3 points within the segment, then round to
            # an integer epoch in [start + 1, end].
            picks = [
                start + max(1, (end - start) // 3),
                start + max(1, 2 * (end - start) // 3),
            ]
            picks = sorted(set(min(end, max(start + 1, p)) for p in picks))
            epochs_per_segment.append(picks)

    for mid in model_ids:
        model, eval_train_loader, valid_loader, task = prepare_everything(
            data_name=data_name,
            model_id=mid,
        )

        (
            checkpoints_per_segment,
            iters_per_segment,
            lrs_per_segment,
        ) = _build_source_segments(
            data_name=data_name,
            model_id=mid,
            num_segments=num_segments,
            epochs_per_segment=epochs_per_segment,
        )

        source = SourceComputer(
            model=model,
            task=task,
            checkpoints_per_segment=checkpoints_per_segment,
            iters_per_segment=iters_per_segment,
            lrs_per_segment=lrs_per_segment,
            n_epoch=1,
        )
        source.build_curvature_blocks(loader=eval_train_loader)
        scores = source.compute_scores_with_loader(
            test_loader=valid_loader, train_loader=eval_train_loader
        )
        expt_name = f"{data_name}_source"
        torch.save(scores, f"{BASE_PATH}/{mid}/{expt_name}.pt")


if __name__ == "__main__":
    data_names = ["mnist", "fmnist"]
    model_ids = [0]

    for dn in data_names:
        compute_if(data_name=dn, model_ids=model_ids)
        compute_source(data_name=dn, model_ids=model_ids)
