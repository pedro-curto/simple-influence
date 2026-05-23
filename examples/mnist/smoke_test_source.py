"""Tiny end-to-end smoke test: train MNIST briefly, then run SOURCE on it.

Trains a small MLP on MNIST for a handful of epochs (saving one checkpoint
per epoch) and immediately runs `SourceComputer` to confirm the pipeline
works end-to-end. Intentionally smaller than `train.py` + `compute_influences.py`
so it finishes in a minute or two on CPU.

Run from `examples/mnist/`:
    python smoke_test_source.py
"""

import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from examples.mnist.pipeline import construct_mlp, get_mnist_dataloader
from examples.mnist.task import ClassificationTask
from examples.utils import set_seed
from src.influence_function import InfluenceFunctionComputer
from src.source import SourceComputer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Smoke-test config. The training loader iterates over the first
# `NUM_TRAIN_SUBSET` MNIST examples (large enough that batches actually run
# under `drop_last=True`); the EK-FAC / attribution loader iterates over the
# first `NUM_EVAL_TRAIN_SUBSET` examples so we don't end up with a
# `(NUM_TRAIN_SUBSET x NUM_TRAIN_SUBSET)` score table.
EPOCHS = 4
TRAIN_BATCH_SIZE = 128
NUM_TRAIN_SUBSET = 4096      # enough for 32 batches/epoch -> 128 training iterations
NUM_EVAL_TRAIN_SUBSET = 256  # subset used as both "training data" for EK-FAC and the
                             # train side of the score table
NUM_VALID = 8
LR = 0.03
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CKPT_DIR = "files/checkpoints/0"


def train_and_save(loader: torch.utils.data.DataLoader) -> None:
    """Train an MNIST MLP for `EPOCHS` epochs and dump one checkpoint per epoch.

    Args:
        loader (DataLoader):
            Training loader.
    """
    set_seed(0)
    model = construct_mlp().to(DEVICE)
    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_fn = CrossEntropyLoss()

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{CKPT_DIR}/mnist_epoch_0.pt")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f"{CKPT_DIR}/mnist_epoch_{epoch}.pt")
        print(f"  epoch {epoch} done")
    return model


def main() -> None:
    """Train briefly and run both IF and SOURCE; print summary statistics."""
    print(f"Device: {DEVICE}")

    # Three independent loaders so the training subset (~4k examples) is large
    # enough for `drop_last=True` to actually yield batches, while the
    # EK-FAC / attribution loader stays small (~256 examples) so the score
    # table is cheap to compute.
    train_loader = get_mnist_dataloader(
        batch_size=TRAIN_BATCH_SIZE,
        split="train",
        indices=list(range(NUM_TRAIN_SUBSET)),
    )
    eval_train_loader = get_mnist_dataloader(
        batch_size=128,
        split="eval_train",
        indices=list(range(NUM_EVAL_TRAIN_SUBSET)),
    )
    valid_loader = get_mnist_dataloader(
        batch_size=128,
        split="valid",
        indices=list(range(NUM_VALID)),
    )
    assert len(train_loader) > 0, (
        "Training loader yields no batches - NUM_TRAIN_SUBSET is too small for the "
        "current TRAIN_BATCH_SIZE with drop_last=True."
    )

    print(
        f"Training MLP for {EPOCHS} epochs on {NUM_TRAIN_SUBSET} MNIST examples "
        f"({len(train_loader)} batches/epoch)..."
    )
    model = train_and_save(loader=train_loader)

    # --- Influence functions, for comparison ---
    print("Computing influence-function scores...")
    task = ClassificationTask(device=DEVICE)
    # Reload the final checkpoint (`train_and_save` returns the in-place
    # trained model, which is already at theta_s, but be explicit).
    model.load_state_dict(torch.load(f"{CKPT_DIR}/mnist_epoch_{EPOCHS}.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    ifc = InfluenceFunctionComputer(model=model, task=task, n_epoch=1)
    ifc.build_curvature_blocks(loader=eval_train_loader)
    if_scores = ifc.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=eval_train_loader
    )
    print(f"  IF scores: shape={tuple(if_scores.shape)}, "
          f"mean={if_scores.mean():.4f}, std={if_scores.std():.4f}")

    # --- SOURCE ---
    print("Computing SOURCE scores...")
    # Two segments to keep wall time low. Each segment averages 2 checkpoints.
    num_segments = 2
    epochs_per_segment = [[1, 2], [3, 4]]
    boundaries = [0, EPOCHS // 2, EPOCHS]

    # `K_l` reflects the actual training schedule the checkpoints sit on top
    # of (number of gradient updates per epoch on the smoke-test training
    # subset), NOT the size of the EK-FAC evaluation subset. Getting this
    # wrong puts the damping `lambda = 1 / (K_l * eta_l)` far off the IF
    # heuristic and the two methods diverge for the wrong reason.
    iters_per_epoch = len(train_loader)
    iters_per_segment = [
        (boundaries[i + 1] - boundaries[i]) * iters_per_epoch
        for i in range(num_segments)
    ]
    effective_lr = LR / (1.0 - MOMENTUM)
    lrs_per_segment = [effective_lr] * num_segments
    checkpoints_per_segment = [
        [f"{CKPT_DIR}/mnist_epoch_{e}.pt" for e in seg] for seg in epochs_per_segment
    ]
    print(f"  segments: {checkpoints_per_segment}")
    print(f"  K_l: {iters_per_segment}, eta_l: {lrs_per_segment}")

    source = SourceComputer(
        model=model,
        task=task,
        checkpoints_per_segment=checkpoints_per_segment,
        iters_per_segment=iters_per_segment,
        lrs_per_segment=lrs_per_segment,
        n_epoch=1,
    )
    source.build_curvature_blocks(loader=eval_train_loader)
    source_scores = source.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=eval_train_loader
    )
    print(f"  SOURCE scores: shape={tuple(source_scores.shape)}, "
          f"mean={source_scores.mean():.4f}, std={source_scores.std():.4f}")
    assert source_scores.shape == if_scores.shape
    assert torch.isfinite(source_scores).all()

    # Rank-overlap with IF. SOURCE is *designed* to differ from IF on
    # non-converged models (paper Fig 7) - for 4-epoch MNIST a meaningful
    # divergence is expected, not a bug.
    from scipy.stats import spearmanr  # local to keep it optional

    rhos = []
    for q in range(source_scores.shape[0]):
        rho, _ = spearmanr(source_scores[q].cpu().numpy(), if_scores[q].cpu().numpy())
        rhos.append(rho)
    print(f"  per-query Spearman(SOURCE, IF): mean={sum(rhos) / len(rhos):.3f} "
          f"min={min(rhos):.3f} max={max(rhos):.3f}")

    # Stronger sanity check: top-k positively-influential training points for
    # a correctly-classified query should mostly share the query's label.
    train_labels = torch.tensor(
        [eval_train_loader.dataset[i][1] for i in range(len(eval_train_loader.dataset))]
    )
    valid_labels = torch.tensor(
        [valid_loader.dataset[i][1] for i in range(len(valid_loader.dataset))]
    )
    top_k = 5
    for name, scores in (("SOURCE", source_scores), ("IF", if_scores)):
        same_label = 0
        for q in range(scores.shape[0]):
            top_idx = scores[q].topk(top_k).indices.cpu()
            same_label += int((train_labels[top_idx] == valid_labels[q]).sum().item())
        rate = same_label / (scores.shape[0] * top_k)
        print(f"  {name}: top-{top_k} same-label rate = {rate:.2%}")

    print("Done.")


if __name__ == "__main__":
    main()
