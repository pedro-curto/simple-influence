"""Task adapter for UCI regression.

Implements the regression-specific loss / measurement / batch-size /
influence-modules contract that every TDA computer in this repo consumes.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.abstract_task import AbstractTask

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]


class RegressionModelOutput:
    """TRAK adapter for the regression task (`get_output` + `get_out_to_loss_grad`)."""

    @staticmethod
    def get_output(
        model: torch.nn.Module,
        weights: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        outputs = torch.func.functional_call(
            model, (weights, buffers), inputs.unsqueeze(0)
        )
        return ((outputs - targets) ** 2.0).sum()

    @staticmethod
    def get_out_to_loss_grad(
        model, weights, buffers, batch: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        del model, weights, buffers
        inputs, targets = batch
        # The gradient of the loss with respect to loss is just 1.
        return torch.ones(1, device=inputs.device)


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

    def influence_modules(self) -> List[str]:
        return ["0", "2", "4", "6"]

    def representation_module(self) -> str:
        return "4"

    def get_model_output(self) -> Optional[Any]:
        return RegressionModelOutput()
