"""Plot the influence-score distribution for a few validation queries.

Loads a pre-computed score table from `files/results/{model_id}/{name}.pt`
(produced by `compute_influences.py`) and plots the sorted influence scores
for the first few validation queries - useful for eyeballing how heavy-tailed
the attribution is for a given dataset / method.

Run from `examples/regression/`:

    cd examples/regression
    PYTHONPATH=../.. python -m examples.regression.evaluate.visualize_distribution
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch


def main(
    data_name: str = "concrete",
    model_id: int = 0,
    expt_name: str = "concrete_if",
    eval_idxs: tuple = (0, 1, 2),
) -> None:
    """Plot the sorted influence scores for each query in `eval_idxs`.

    Args:
        data_name: UCI dataset name (used only for the plot title).
        model_id: index of the trained model directory under
            ``files/checkpoints/``.
        expt_name: name of the score file under
            ``files/results/{model_id}/`` (without the ``.pt`` suffix).
        eval_idxs: which query rows to plot.
    """
    score_path = f"files/results/{model_id}/{expt_name}.pt"
    if not os.path.exists(score_path):
        raise FileNotFoundError(
            f"Score file not found at {score_path}. Run `compute_influences.py` first."
        )
    scores = torch.load(score_path, map_location="cpu")

    for idx in eval_idxs:
        plt.figure()
        plt.plot(torch.sort(scores[idx]).values)
        plt.title(f"{data_name}: influence distribution for query index {idx}")
        plt.ylabel("Influence score")
        plt.xlabel("Training data example (sorted)")
        plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-name", default="concrete")
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument("--expt-name", default="concrete_if")
    parser.add_argument(
        "--eval-idxs",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Validation indices whose distributions to plot.",
    )
    args = parser.parse_args()
    main(
        data_name=args.data_name,
        model_id=args.model_id,
        expt_name=args.expt_name,
        eval_idxs=tuple(args.eval_idxs),
    )
