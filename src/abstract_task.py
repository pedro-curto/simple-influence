"""Abstract task interface for training-data-attribution pipelines.

A `Task` bundles together the model-specific pieces that every TDA computer in
this package needs: the loss function used for training, the measurement
function used at query time, batch-size introspection, and the list of modules
whose parameters participate in attribution. Subclass `AbstractTask` for your
own model and dataset; see `examples/` for four end-to-end implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# `trak` is only required for the TRAK attribution method. Keep the import
# behind `TYPE_CHECKING` so the rest of the package can be used without
# installing it.
if TYPE_CHECKING:
    from trak.modelout_functions import AbstractModelOutput


class InvalidTaskError(Exception):
    """Raised by `validate_task` when a task is misconfigured for a given model."""


class AbstractTask(ABC):
    """Abstract base class for tasks consumed by every `*Computer`.

    Subclasses adapt a TDA pipeline (loss, measurement, batch handling, list of
    influence-bearing modules) to a particular model architecture and dataset.
    See `examples/regression/task.py` and similar files for concrete reference
    implementations covering regression, image classification, sequence
    classification, and language modeling.
    """

    @abstractmethod
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize the task.

        Args:
            device (torch.device, optional):
                Device on which task-side computations (e.g. moving the batch,
                sampling targets) take place. Defaults to CPU.
            generator (torch.Generator, optional):
                Optional pre-seeded generator used when `sample=True` paths draw
                targets from the model output (true-Fisher computation). If
                `None`, sampling uses PyTorch's default global RNG (i.e.
                non-deterministic across calls).
        """
        self.device = device
        self.generator = generator

    @abstractmethod
    def get_train_loss(
        self,
        model: nn.Module,
        batch: Any,
        parameter_and_buffer_dicts: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the training loss for a given model and batch.

        All operations inside must be traceable by autograd; the result is
        differentiated by every gradient-based computer in this package.

        Args:
            model (nn.Module):
                Model to evaluate.
            batch (Any):
                A single batch yielded by the data loader. The exact type
                depends on the task - tuples for vision tasks, dicts for
                transformer tasks, etc.
            parameter_and_buffer_dicts (tuple, optional):
                If provided, a `(params, buffers)` pair used in place of
                `model.parameters()` / `model.buffers()`. This is what makes the
                loss compatible with `torch.func.functional_call` (and hence
                with per-sample gradients via `vmap`).
            sample (bool, optional):
                If `True`, sample targets from the current model output (used
                to estimate the true Fisher); otherwise use the supplied
                labels. Defaults to `False`.
            reduction (str, optional):
                Reduction strategy: `"sum"` (default), `"mean"`, or `"none"`.

        Returns:
            torch.Tensor: the (possibly reduced) loss.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_measurement(
        self,
        model: nn.Module,
        batch: Any,
        parameter_and_buffer_dicts: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the measurement function `f(theta)` for query data points.

        The "measurement" is the quantity whose change under leave-one-out
        retraining we are trying to attribute (loss, margin, conditional
        log-likelihood, etc.); see Section 2.1 and Eq. 6 of
        https://arxiv.org/pdf/2308.03296.pdf. For many tasks `get_measurement`
        is the same as `get_train_loss`, but they can differ (e.g. margin for
        classification).

        Args:
            model (nn.Module):
                Model to evaluate.
            batch (Any):
                A single batch from the data loader (query points).
            parameter_and_buffer_dicts (tuple, optional):
                `(params, buffers)` pair used in place of the model's own
                parameters; see `get_train_loss` for details.
            sample (bool, optional):
                If `True`, sample targets from the model output; otherwise use
                the supplied labels. Defaults to `False`.
            reduction (str, optional):
                Reduction strategy: `"sum"` (default), `"mean"`, or `"none"`.

        Returns:
            torch.Tensor: the (possibly reduced) measurement value.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_batch_size(self, batch: Any) -> int:
        """Return the batch size for a batch yielded by the data loader.

        Args:
            batch (Any):
                A single batch from the data loader.

        Returns:
            int: the number of examples in the batch.
        """
        raise NotImplementedError()

    def influence_modules(self) -> Optional[List[str]]:
        """Return the list of module names whose parameters carry attribution.

        TDA scores are computed over the parameters of these modules only.
        Returning `None` means "use every supported module in the model"
        (Linear, Conv2d, Embedding, LayerNorm, BatchNorm2d).

        Returns:
            Optional[List[str]]: list of qualified module names, or `None`.
        """
        return None

    @abstractmethod
    def representation_module(self) -> str:
        """Return the module name whose output is used as the representation.

        Required by `RepresentationSimilarityComputer`; should typically be the
        module immediately before the final classification / regression layer.

        Returns:
            str: qualified name of the representation-producing module.
        """
        raise NotImplementedError()

    def get_activation_masks(self, batch: Any) -> Optional[torch.Tensor]:
        """Return a mask for padded positions in the batch's activations.

        Used by EK-FAC to ignore padded tokens when accumulating activation
        covariances (e.g. in Transformer pipelines where input sequences are
        padded to a common length). Architectures with fixed input size should
        return `None`.

        Args:
            batch (Any):
                A single batch from the data loader.

        Returns:
            Optional[torch.Tensor]: a mask broadcastable over the activations,
            or `None` if no masking is needed.
        """
        del batch
        return None

    def get_model_output(self) -> Optional["AbstractModelOutput"]:
        """Return the TRAK `AbstractModelOutput` adapter for this task.

        TRAK requires a small adapter exposing the model output function and
        the gradient of the training loss w.r.t. that output; see
        https://trak.readthedocs.io/en/latest/modeloutput.html. Only required
        when using `TrakComputer`.

        Returns:
            Optional[AbstractModelOutput]: TRAK adapter, or `None` if this task
            does not support TRAK.
        """
        return None


def validate_task(
    model: nn.Module, task: AbstractTask, logger: Optional[logging.Logger] = None
) -> None:
    """Verify that a task is consistent with a given model.

    Specifically:
      - every module named in `task.influence_modules()` exists on the model
        and is a supported layer type;
      - the module named in `task.representation_module()` exists on the model
        (or both it and `logger` are `None`, in which case a warning is
        emitted).

    Args:
        model (nn.Module):
            The model that the task will be applied to.
        task (AbstractTask):
            The task to validate.
        logger (logging.Logger, optional):
            If provided, soft validation failures (e.g. missing optional
            method) are logged here. Hard failures are always raised as
            `InvalidTaskError`.

    Raises:
        InvalidTaskError: if a required module is missing or has the wrong type.
    """
    # `influence_modules()` may return `None` to indicate "use every supported
    # module"; in that case there's nothing to validate against an explicit list.
    influence_modules_list = task.influence_modules()
    influence_module_exists_dict: Dict[str, bool] = (
        {} if influence_modules_list is None else {name: False for name in influence_modules_list}
    )

    # `representation_module()` is only required for representation-similarity
    # methods. If it's `None`, we skip the existence check (and warn via the
    # logger when one is available).
    representation_module = task.representation_module()
    if representation_module is None:
        if logger is not None:
            logger.warning("`representation_module` is not defined.")
        representation_module_exists = True
    else:
        representation_module_exists = False

    model_output = task.get_model_output()
    if model_output is None and logger is not None:
        logger.warning(
            "`model_output` is not defined (this must be implemented for TRAK)."
        )

    for name, module in model.named_modules():
        if name in influence_module_exists_dict:
            if not isinstance(
                module,
                (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, nn.BatchNorm2d),
            ):
                error_msg = (
                    f"The provided influence module with {name} is not supported."
                )
                if logger is not None:
                    logger.error(error_msg)
                raise InvalidTaskError(error_msg)
            influence_module_exists_dict[name] = True
        if representation_module is not None and representation_module == name:
            representation_module_exists = True

    missing_modules = [name for name, found in influence_module_exists_dict.items() if not found]
    if missing_modules:
        error_msg = (
            f"Some provided influence modules were not found in the model: {missing_modules}."
        )
        if logger is not None:
            logger.error(error_msg)
        raise InvalidTaskError(error_msg)

    if not representation_module_exists:
        error_msg = f"Provided representation module with name {representation_module} was not found."
        if logger is not None:
            logger.error(error_msg)
        raise InvalidTaskError(error_msg)
