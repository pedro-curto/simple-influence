from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.abstract_task import AbstractTask

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]


class RegressionTask(AbstractTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def get_train_loss(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        inputs, targets = batch

        if parameter_and_buffer_dicts is None:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
        else:
            inputs = inputs.unsqueeze(0).to(self.device)
            targets = targets.unsqueeze(0).to(self.device)
            params, buffers = parameter_and_buffer_dicts
            outputs = torch.func.functional_call(model, (params, buffers), (inputs,))

        if not sample:
            return F.mse_loss(outputs, targets.to(self.device), reduction=reduction)
        else:
            with torch.no_grad():
                sampled_targets = torch.normal(outputs, std=0.5)
            return F.mse_loss(outputs, sampled_targets.detach(), reduction=reduction)

    def get_measurement(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        return self.get_train_loss(
            model, batch, parameter_and_buffer_dicts, sample, reduction
        )

    def get_batch_size(self, batch: BATCH_DTYPE) -> int:
        inputs, _ = batch
        return inputs.shape[0]
