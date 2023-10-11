import copy
from typing import List, Optional, Union

import torch
import torch.nn as nn

from src.baselines.gradients_similarity import GradientsSimilarityComputer


class TracinComputer:
    score_dtype: torch.dtype = torch.float32

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        task: str = "classification",
        metric: str = "cos",
        last_only: bool = False,
    ):
        self.model = model
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        if device is None:
            self.device = next(iter(self.model.parameters())).device
        else:
            self.device = device

        self.task = task
        self.metric = metric
        assert self.metric in ["cos", "dot"]
        self.last_only = last_only

    def load_checkpoint(self, checkpoint: str) -> None:
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(self.device)
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
                device=self.device,
                requires_grad=False,
            )

        if isinstance(lrs, float) or isinstance(lrs, int):
            lrs = [lrs for _ in range(len(checkpoints))]

        for i, ckpt in enumerate(checkpoints):
            self.load_checkpoint(ckpt)
            gc = GradientsSimilarityComputer(
                model=self.model,
                device=self.device,
                task=self.task,
                metric=self.metric,
                last_only=self.last_only,
            )
            score_table += lrs[i] * gc.compute_total_influence(
                valid_loader=valid_loader, train_loader=train_loader
            )
            del gc

        score_table.div_(len(checkpoints))
        self.model.load_state_dict(self.original_state_dict)
        return score_table
