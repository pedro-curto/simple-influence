import copy
from typing import Any, List, Union

import torch
import torch.nn as nn

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask
from src.gradient_similarity import GradientSimilarityComputer


class TracinComputer(AbstractComputer):
    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        metric: str = "cos",
    ):
        """Initializes the `TracIn` class.

        This class performs TDA using similarity between gradients (over trajectories) based on a specified metric
        (for more details, see https://arxiv.org/pdf/2002.08484.pdf). Note that when `metric = "cos"`, the method
        is called Gradients Aggregated Similarity (GAS).

        Args:
            model (nn.Module):
                The PyTorch model for which representations are computed.
            task (AbstractTask):
                The task for the pipeline.
            metric (str, optional):
                The metric used to measure similarity. Supported metrics include "dot"
                and "cos". Defaults to "dot".
        """
        super().__init__(model, task)
        # Save original parameters to CPU.
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        for name, tensor in self.original_state_dict.items():
            self.original_state_dict[name] = tensor.cpu()

        self.metric = metric

    def _load_checkpoint(self, checkpoint: str) -> None:
        """Given the path to the checkpoint, load the parameters and buffers."""
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(self.task.device)
        self.model.eval()

    def compute_pairwise_influence(
        self, checkpoints: List[str], batch1: Any, batch2: Any, lrs
    ) -> torch.Tensor:
        """Compute pairwise influence scores between data points in `batch1` and `batch2`.

        Args:
            batch1 (object):
                The first set of data points from the data loader.
            batch2 (object):
                The second set of data points from the data loader.
        """
        self.model.eval()

        # Perform lazy initializations.
        score_table = 0.0

        if isinstance(lrs, float) or isinstance(lrs, int):
            lrs = [lrs for _ in range(len(checkpoints))]

        for i, ckpt in enumerate(checkpoints):
            self._load_checkpoint(ckpt)
            gsc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
            )
            score_table += lrs[i] * gsc.compute_pairwise_influence(
                batch1=batch1, batch2=batch2
            )
            del gsc

        score_table.div_(len(checkpoints))
        self.model.load_state_dict(self.original_state_dict)
        self.model = self.model.to(self.task.device)
        return score_table

    def compute_total_influence(
        self,
        checkpoints: List[str],
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        lrs: Union[int, float, List[int], List[float]] = 1.0,
    ) -> torch.Tensor:
        self.model.eval()
        score_table = torch.zeros(
            (len(test_loader.dataset), len(train_loader.dataset)),
            dtype=self.score_dtype,
            device=self.task.device,
            requires_grad=False,
        )

        if isinstance(lrs, float) or isinstance(lrs, int):
            lrs = [lrs for _ in range(len(checkpoints))]

        for i, ckpt in enumerate(checkpoints):
            self._load_checkpoint(ckpt)
            gsc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
            )
            score_table += lrs[i] * gsc.compute_total_influence(
                test_loader=test_loader, train_loader=train_loader
            )
            del gsc

        score_table.div_(len(checkpoints))
        self.model.load_state_dict(self.original_state_dict)
        self.model = self.model.to(self.task.device)
        return score_table
