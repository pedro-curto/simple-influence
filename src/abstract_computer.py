"""Abstract base class shared by every attribution computer.

Provides the small bit of plumbing that all computers in this package need:
holding references to a model and its task, a configured logger, and helpers
for computing per-sample loss / measurement gradients via `torch.func`.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from src.abstract_task import AbstractTask, validate_task


class AbstractComputer(ABC):
    """Abstract base class for attribution computers.

    Class attributes control numerical precision throughout the pipeline. They
    are conservative defaults; individual subclasses are free to override.

    Attributes:
        score_dtype (torch.dtype): dtype used to store final attribution scores.
        grads_dtype (torch.dtype): dtype used to store per-sample gradients.
        stats_dtype (torch.dtype): dtype used to store EK-FAC covariance
            statistics (applies only to influence-function-style computers).
        eig_dtype (torch.dtype): dtype used for eigendecomposition (applies
            only to influence-function-style computers).
    """

    score_dtype: torch.dtype = torch.float32
    grads_dtype: torch.dtype = torch.float32
    stats_dtype: torch.dtype = torch.float32
    eig_dtype: torch.dtype = torch.float64

    @abstractmethod
    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        logger_name: str,
        logging_level: int = logging.INFO,
    ) -> None:
        """Initialize the computer.

        Args:
            model (nn.Module):
                Model for which attribution scores will be computed.
            task (AbstractTask):
                Task adapter for the model. See `src/abstract_task.py`.
            logger_name (str):
                Name passed to `logging.getLogger`; typically the subclass
                name.
            logging_level (int, optional):
                Logger level. Defaults to `logging.INFO`.
        """
        self.model = model
        self.task = task

        # Use a named logger and let the application configure
        # handlers/formatters. Avoid calling `logging.basicConfig()` here,
        # since it mutates global logging state and can collide with the
        # user's own configuration.
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging_level)

        validate_task(model=self.model, task=self.task, logger=self.logger)

    def _compute_train_loss(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Any,
    ) -> torch.Tensor:
        """Compute the summed training loss for a batch under the given params.

        Used as the inner function of `torch.func.grad` to obtain per-sample
        training-loss gradients.

        Args:
            params (dict):
                Parameter dict, typically obtained from `model.named_parameters()`.
            buffers (dict):
                Buffer dict, typically obtained from `model.named_buffers()`.
            batch (Any):
                A single batch.

        Returns:
            torch.Tensor: scalar summed loss.
        """
        return self.task.get_train_loss(
            model=self.model,
            batch=batch,
            parameter_and_buffer_dicts=(params, buffers),
            sample=False,
            reduction="sum",
        )

    def _compute_train_loss_grad(self) -> Callable:
        """Return `torch.func.grad` of `_compute_train_loss` w.r.t. params.

        Returns:
            Callable: a function `(params, buffers, batch) -> param-grads-dict`.
        """
        return torch.func.grad(self._compute_train_loss, argnums=0, has_aux=False)

    def _compute_measurement(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Any,
    ) -> torch.Tensor:
        """Compute the summed measurement for a batch under the given params.

        Used as the inner function of `torch.func.grad` for query-side
        gradients.

        Args:
            params (dict):
                Parameter dict.
            buffers (dict):
                Buffer dict.
            batch (Any):
                A single batch (typically the query / test batch).

        Returns:
            torch.Tensor: scalar summed measurement.
        """
        return self.task.get_measurement(
            model=self.model,
            batch=batch,
            parameter_and_buffer_dicts=(params, buffers),
            reduction="sum",
        )

    def _compute_measurement_grad(self) -> Callable:
        """Return `torch.func.grad` of `_compute_measurement` w.r.t. params.

        Returns:
            Callable: a function `(params, buffers, batch) -> param-grads-dict`.
        """
        return torch.func.grad(self._compute_measurement, argnums=0, has_aux=False)
