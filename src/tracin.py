"""TracIn / GAS gradient-trajectory TDA.

Implements the practical (checkpoint-based) variant of TracIn from
https://arxiv.org/pdf/2002.08484.pdf. The attribution score for each pair
``(z_q, z_m)`` is the average over training checkpoints of

    learning_rate(t) * <grad f(z_q, theta_t), grad L(z_m, theta_t)>

When the similarity metric is cosine (``metric="cos"``) the method is also
known as Gradient-Aggregated Similarity (GAS).
"""

import copy
from typing import Any, List, Union

import torch
import torch.nn as nn

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask
from src.gradient_similarity import GradientSimilarityComputer


class TracinComputer(AbstractComputer):
    """Compute TracIn / GAS attribution scores from a list of checkpoints.

    For each checkpoint we instantiate a :class:`GradientSimilarityComputer`
    on the loaded parameters, compute a similarity table, and add it
    (weighted by the learning rate) to the running sum. After all
    checkpoints have been visited the running sum is divided by the number
    of checkpoints. The model's original parameters are restored at the
    end.

    Attributes:
        original_state_dict (dict): the model's parameters at construction
            time (kept on CPU); restored after every score computation.
        metric (str): ``"dot"`` (TracIn) or ``"cos"`` (GAS).
    """

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        metric: str = "cos",
    ) -> None:
        """Initialize the TracIn computer.

        Args:
            model (nn.Module):
                Model whose final parameters are the starting point; the
                computer will overwrite these with each checkpoint while
                scoring and restore them at the end.
            task (AbstractTask):
                Task adapter.
            metric (str, optional):
                Similarity metric: ``"dot"`` (TracIn) or ``"cos"`` (GAS).
                Defaults to ``"cos"``.
        """
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        # Snapshot the original parameters on CPU so that we can restore them
        # after each score computation (the inner loop overwrites them with
        # each checkpoint).
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        for name, tensor in self.original_state_dict.items():
            self.original_state_dict[name] = tensor.cpu()

        self.metric = metric

    def _load_checkpoint(self, checkpoint: str) -> None:
        """Load a checkpoint into ``self.model`` and move it to the task device.

        Args:
            checkpoint (str):
                Path to a state-dict ``.pt`` file.
        """
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(self.task.device)
        self.model.eval()

    def _reload_original_params(self) -> None:
        """Restore the parameters captured at construction time."""
        self.model.load_state_dict(self.original_state_dict)
        self.model = self.model.to(self.task.device)

    @staticmethod
    def _normalize_lrs(
        lrs: Union[int, float, List[int], List[float]], num_checkpoints: int
    ) -> List[float]:
        """Broadcast a scalar learning rate to one entry per checkpoint.

        Args:
            lrs (int, float, or list):
                Either a single learning rate (applied to every checkpoint)
                or a list whose length matches ``num_checkpoints``.
            num_checkpoints (int):
                Number of checkpoints to score against.

        Returns:
            List[float]: a list of learning rates with length
            ``num_checkpoints``.
        """
        if isinstance(lrs, (int, float)):
            return [float(lrs) for _ in range(num_checkpoints)]
        return [float(x) for x in lrs]

    def compute_scores_with_batch(
        self,
        batch1: Any,
        batch2: Any,
        checkpoints: List[str],
        lrs: Union[List[float], float],
    ) -> torch.Tensor:
        """Compute TracIn scores for a pair of batches.

        Args:
            batch1 (Any):
                Query (test) batch.
            batch2 (Any):
                Training batch.
            checkpoints (List[str]):
                Paths to checkpoints to aggregate over. Must be non-empty.
            lrs (float or List[float]):
                Per-checkpoint learning rate(s). If a scalar, the same rate
                is used for every checkpoint.

        Returns:
            torch.Tensor: ``(|batch1|, |batch2|)`` score table.

        Raises:
            ValueError: if ``checkpoints`` is empty.
        """
        if len(checkpoints) == 0:
            raise ValueError("`checkpoints` must contain at least one path.")
        self.model.eval()

        score_table = 0.0
        lrs = self._normalize_lrs(lrs, len(checkpoints))

        for i, ckpt in enumerate(checkpoints):
            self._load_checkpoint(ckpt)
            gsc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
            )
            score_table += lrs[i] * gsc.compute_scores_with_batch(
                batch1=batch1, batch2=batch2
            )
            del gsc

        score_table.div_(len(checkpoints))
        self._reload_original_params()
        return score_table

    def compute_scores_with_loader(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        checkpoints: List[str],
        lrs: Union[int, float, List[int], List[float]] = 1.0,
    ) -> torch.Tensor:
        """Compute TracIn scores for a pair of loaders.

        Args:
            test_loader (DataLoader):
                Loader yielding query (test) data.
            train_loader (DataLoader):
                Loader yielding training data.
            checkpoints (List[str]):
                Paths to checkpoints to aggregate over. Must be non-empty.
            lrs (float or List[float], optional):
                Per-checkpoint learning rate(s); defaults to ``1.0`` for all.

        Returns:
            torch.Tensor: ``(num_test, num_train)`` score table.

        Raises:
            ValueError: if ``checkpoints`` is empty.
        """
        if len(checkpoints) == 0:
            raise ValueError("`checkpoints` must contain at least one path.")
        self.model.eval()

        score_table = torch.zeros(
            (len(test_loader.dataset), len(train_loader.dataset)),
            dtype=self.score_dtype,
            device=self.task.device,
            requires_grad=False,
        )

        lrs = self._normalize_lrs(lrs, len(checkpoints))

        for i, ckpt in enumerate(checkpoints):
            self._load_checkpoint(ckpt)
            gsc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
            )
            score_table += lrs[i] * gsc.compute_scores_with_loader(
                test_loader=test_loader, train_loader=train_loader
            )
            del gsc

        score_table.div_(len(checkpoints))
        self._reload_original_params()
        return score_table

    def compute_self_scores_with_loader(
        self,
        loader: torch.utils.data.DataLoader,
        checkpoints: List[str],
        lrs: Union[int, float, List[int], List[float]] = 1.0,
    ) -> torch.Tensor:
        """Compute TracIn self-similarity scores for one loader.

        Args:
            loader (DataLoader):
                Loader yielding the data points whose self-influence is
                computed.
            checkpoints (List[str]):
                Paths to checkpoints. Must be non-empty.
            lrs (float or List[float], optional):
                Per-checkpoint learning rate(s); defaults to ``1.0`` for all.

        Returns:
            torch.Tensor: 1-D tensor of self-similarity scores.

        Raises:
            ValueError: if ``checkpoints`` is empty.
        """
        if len(checkpoints) == 0:
            raise ValueError("`checkpoints` must contain at least one path.")
        self.model.eval()

        score_table = 0.0
        lrs = self._normalize_lrs(lrs, len(checkpoints))

        for i, ckpt in enumerate(checkpoints):
            self._load_checkpoint(ckpt)
            gsc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
            )
            score_table += lrs[i] * gsc.compute_self_scores_with_loader(loader=loader)
            del gsc

        score_table.div_(len(checkpoints))
        self._reload_original_params()
        return score_table
