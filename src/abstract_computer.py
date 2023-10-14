import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.abstract_task import AbstractTask


class AbstractComputer(ABC):
    """Abstract class for all influence computing baseline methods."""

    score_dtype: torch.dtype = torch.float32

    @abstractmethod
    def __init__(
        self, model: nn.Module, task: AbstractTask, logging_level: int = logging.INFO
    ) -> None:
        """Initializes the class AbstractComputer.

        Args:
            model (nn.Module):
                The module for computing influence scores.
            task (AbstractTask):
                The Task for the problem. For details, see `src/abstract_computer`.
            logging_level (int):
                The logging level. Defaults to `logging.INFO`.
        """
        self.model = model
        self.task = task

        # Setup logging configurations.
        logging.basicConfig()
        self.logger = logging.getLogger("influence-prototype")
        self.logger.setLevel(logging_level)
        self.logger.warning(
            "The repository is still under development. Please report any issues at GitHub."
        )

    def _compute_train_loss(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Any,
    ) -> torch.Tensor:
        """Given the parameters, buffers, and a batch, compute the sum of all individual training losses."""
        return self.task.get_train_loss(
            model=self.model,
            batch=batch,
            parameter_and_buffer_dicts=(params, buffers),
            sample=False,
            reduction="sum",
        )

    def _compute_train_loss_grad(self):
        return torch.func.grad(self._compute_train_loss, argnums=0, has_aux=False)

    def _compute_measurement(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Any,
    ) -> torch.Tensor:
        """Given the parameters, buffers, and a batch, compute the sum of all individual measurements."""
        return self.task.get_measurement(
            model=self.model,
            batch=batch,
            parameter_and_buffer_dicts=(params, buffers),
            sample=False,
            reduction="sum",
        )

    def _compute_measurement_grad(self):
        return torch.func.grad(self._compute_measurement, argnums=0, has_aux=False)
