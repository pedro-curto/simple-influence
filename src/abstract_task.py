from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class AbstractTask(ABC):
    """Implementations of the Task class should allow to compute influence functions (and any
    baseline methods) with desired configuration. See `examples/` for how the Task classes are
    implemented for regression, classification, and language modeling.
    """

    @abstractmethod
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        """Initializes the class AbstractTask.

        Args:
            device (torch.dtype):
                Device you want to use for influence computation. Defaults to CPU.
            generator (torch.Generator, optional):
                Generator to fix the behaviour of the experiment (e.g., sampling outputs for
                computing the true Fisher). Defaults to using non-deterministic sampling.
        """
        self.device = device
        self.generator = generator

    @abstractmethod
    def get_train_loss(
        self,
        model: nn.Module,
        batch: Any,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Given the model and batch, compute the training loss. Note that the operations
        in this function should be able to traced by autograd.

        Args:
            model (nn.Module):
                Model.
            batch (Any):
                Batch received from the DataLoader.
            parameter_and_buffer_dicts (Union[Dict[str, torch.Tensor]], optional):
                Optionally, instead of using the parameters given by `model`, one can directly
                specify the parameters to compute the loss.
            sample (bool):
                Whether to sample the labels from the outputs for computing true Fisher. If False,
                it uses the true label.
            reduction (str):
                By default, the function returns the sum of all losses. One can modify this behaviour
                by specifying `average` or `none`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_measurement(
        self,
        model: nn.Module,
        batch: Any,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Given the model and batch, compute the measurement (e.g., loss, margin, conditional log probability).
        In many cases, measurements are defined as the loss, but one can implement their custom measurement.

        Args:
            model (nn.Module):
                Model.
            batch (Any):
                Batch received from the DataLoader.
            parameter_and_buffer_dicts (Union[Dict[str, torch.Tensor]], optional):
                Optionally, instead of using the parameters given by `model`, one can directly
                specify the parameters to compute the loss.
            sample (bool):
                Whether to sample the labels from the outputs for computing true Fisher. If False,
                it uses the true label.
            reduction (str):
                By default, the function returns the sum of all losses. One can modify this behaviour
                by specifying `average` or `none`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_batch_size(self, batch: Any) -> int:
        """Given a batch, return the batch size.

        Args:
            batch (Any):
                Batch received from the DataLoader.
        """
        raise NotImplementedError()

    @abstractmethod
    def influence_modules(self) -> List[str]:
        """For a desired architecture, return the module names for layers wanting to compute influences on.
        Returning None will compute influences on all available modules (e.g., Embedding, Linear, Conv2d).
        """
        raise NotImplementedError()

    @abstractmethod
    def representation_modules(self) -> List[str]:
        """For a desired architecture, return the module names for layers wanting to compute influences on.
        Returning None will compute influences on all available modules (e.g., Embedding, Linear, Conv2d).
        """
        raise NotImplementedError()

    def get_activation_masks(self, batch: Any) -> Optional[torch.Tensor]:
        """For some architectures (e.g., Transformers), the data point is padded to have equal
        length. Return the masks that is applied on these padded features. Returning None should
        be the standard behaviour of architectures with fixed input size.

        Args:
            batch (Any):
                Batch received from the DataLoader.
        """
        del batch
        return None
