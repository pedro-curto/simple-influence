"""Visualize the top influential MNIST training images for a few queries.

Loads a pre-computed score table from `files/results/{model_id}/{name}.pt`
(produced by `compute_influences.py`) and, for each of the first few query
images, plots the query alongside the top-5 most positively influential
training images. The query and influential images are annotated with their
true labels so you can quickly tell whether the result looks reasonable.

Run from the project root with `examples/mnist` on the working directory
(so the relative paths resolve):

    cd examples/mnist
    PYTHONPATH=../.. python -m examples.mnist.evaluate.visualize_influences
"""

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.mnist.pipeline import get_loaders


def _imshow_mnist(ax: plt.Axes, image: torch.Tensor, title: str) -> None:
    """Plot a single 28x28 MNIST image into the given axes."""
    ax.imshow(np.asarray(image).reshape(28, 28), cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main(
    data_name: str = "mnist",
    model_id: int = 0,
    expt_name: str = "mnist_if",
    num_queries: int = 6,
    top_k: int = 5,
) -> None:
    """Plot the top-`top_k` influential training images for the first
    `num_queries` validation images.

    Args:
        data_name: ``"mnist"`` or ``"fmnist"``.
        model_id: index of the trained model directory under
            ``files/checkpoints/``.
        expt_name: name of the score file under
            ``files/results/{model_id}/`` (without the ``.pt`` suffix).
            Examples: ``"mnist_if"``, ``"mnist_source"``.
        num_queries: number of validation images to visualize.
        top_k: number of influential training images per query.
    """
    score_path = f"files/results/{model_id}/{expt_name}.pt"
    if not os.path.exists(score_path):
        raise FileNotFoundError(
            f"Score file not found at {score_path}. Run `compute_influences.py` first."
        )
    scores = torch.load(score_path, map_location="cpu")

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name,
        train_indices=None,
        valid_indices=list(range(num_queries)),
        eval_batch_size=8,
    )
    train_ds = eval_train_loader.dataset
    valid_ds = valid_loader.dataset

    # The score table has shape (num_valid, num_train). The compute script
    # already restricted num_valid; we only need to assert it covers our
    # `num_queries`.
    if scores.shape[0] < num_queries:
        raise ValueError(
            f"Score file only has {scores.shape[0]} query rows, but num_queries="
            f"{num_queries} was requested. Re-run `compute_influences.py` with a "
            f"larger valid set or lower `num_queries`."
        )

    for q in range(num_queries):
        fig, axs = plt.subplots(ncols=top_k + 1, figsize=(2.0 * (top_k + 1), 2.5))
        fig.suptitle(f"Top-{top_k} influential train images ({expt_name})", fontsize=10)

        query_img, query_label = valid_ds[q]
        _imshow_mnist(axs[0], query_img, f"query #{q}\nlabel={query_label}")

        top_idxs: List[int] = scores[q].argsort(descending=True)[:top_k].tolist()
        for ii, train_idx in enumerate(top_idxs):
            train_img, train_label = train_ds[train_idx]
            _imshow_mnist(
                axs[ii + 1],
                train_img,
                f"rank {ii + 1}\nlabel={train_label}\nscore={scores[q, train_idx]:.2f}",
            )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-name", default="mnist", choices=["mnist", "fmnist"])
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument(
        "--expt-name",
        default="mnist_if",
        help="score-file basename under files/results/{model_id}/ (no .pt suffix). "
        "e.g. mnist_if, mnist_source.",
    )
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    main(
        data_name=args.data_name,
        model_id=args.model_id,
        expt_name=args.expt_name,
        num_queries=args.num_queries,
        top_k=args.top_k,
    )
