import copy
from typing import List, Union

import torch
import torch.nn as nn

from src.abstract_computer import AbstractComputer
from src.gradient_similarity import GradientSimilarityComputer


class TracinComputer(AbstractComputer):
    def __init__(
        self,
        model: nn.Module,
        task,
        metric: str = "cos",
    ):
        super().__init__(model, task)
        self.model = model
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        for name, tensor in self.original_state_dict.items():
            self.original_state_dict[name] = tensor.cpu()

        self.task = task
        self.metric = metric
        assert self.metric in ["cos", "dot"]

    def load_checkpoint(self, checkpoint: str) -> None:
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(self.task.device)
        self.model.eval()

    def compute_total_influence(
        self,
        checkpoints: List[str],
        valid_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        lrs: Union[int, float, List[int], List[float]] = 1.0,
    ) -> torch.Tensor:
        self.model.eval()

        with torch.no_grad():
            score_table = torch.zeros(
                (len(valid_loader.dataset), len(train_loader.dataset)),
                dtype=self.score_dtype,
                device=self.task.device,
                requires_grad=False,
            )

        if isinstance(lrs, float) or isinstance(lrs, int):
            lrs = [lrs for _ in range(len(checkpoints))]

        for i, ckpt in enumerate(checkpoints):
            self.load_checkpoint(ckpt)
            gc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
            )
            score_table += lrs[i] * gc.compute_total_influence(
                valid_loader=valid_loader, train_loader=train_loader
            )
            del gc

        score_table.div_(len(checkpoints))
        self.model.load_state_dict(self.original_state_dict)
        self.model = self.model.to(self.task.device)
        return score_table
