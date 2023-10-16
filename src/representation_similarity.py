from typing import Any

import torch
import torch.nn as nn

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask


class RepresentationSimilarityComputer(AbstractComputer):
    # Specifies the dtype for computing similarity scores.
    _similarity_dtype: torch.dtype = torch.float64

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        metric: str = "dot",
    ) -> None:
        """Initializes the `RepresentationsSimilarityComputer` class.

        This class performs TDA using similarity between representations (activations) based on a specified metric
        (for more details, see https://arxiv.org/pdf/2006.04528.pdf).

        Args:
            model (nn.Module):
                The PyTorch model for which representation similarities are computed.
            task (AbstractTask):
                The task for the pipeline. It is essential that `AbstractTask.representation_modules()`
                is defined for this task.
            metric (str, optional):
                The metric used to measure similarity. Supported metrics include "l2", "dot",
                and "cos". Defaults to "dot".
        """
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        self.target_module_name = self.task.representation_modules()
        if self.target_module_name is None:
            error_msg = (
                f"For `{self.__class__.__name__}`, the target module name must be specified "
                "by defining `AbstractTask.representation_modules()`."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        self.metric = metric
        if self.metric not in ["dot", "cos", "l2"]:
            error_msg = (
                f"Not supported metric {self.metric} for `{self.__class__.__name__}`."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        self._handle = None
        self._target_module = None
        self._temp_acts = None
        self.initialize()

    def initialize(self) -> None:
        """Initializes the forward hook to cache intermediate activation."""
        self.logger.info("Initializing forward hook...")
        for name, module in self.model.named_modules():
            if name == self.target_module_name:
                self.logger.info(f"Found module with name {name}")
                self._handle = module.register_forward_hook(self._forward_hook)
                return

        error_msg = f"Unable to find the specified module {self.target_module_name}."
        self.logger.error(error_msg)
        assert AttributeError(error_msg)

    def remove_hook(self) -> None:
        """Removes the initialized hook."""
        self.logger.info("Removing hook...")
        self._handle.remove()
        self._handle = None
        self._temp_acts = None

    def _forward_hook(
        self, module: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> None:
        """Cache the intermediate activations."""
        del inputs
        self._temp_acts = outputs.data

    def _compute_similarity(
        self, batch_vector1: torch.Tensor, batch_vector2: torch.Tensor
    ) -> torch.Tensor:
        """Computes the pairwise similarities between vectors in `batch_vector1` and `batch_vector2`.

        Given vectors of size `batch_size x dim`, the output is a matrix of size `batch_size1 x batch_size2`.
        Each element in the output matrix represents the similarity distance between corresponding vectors
        from `batch_vector1` and `batch_vector2`.
        """
        if self.metric == "dot":
            score = torch.matmul(batch_vector1, batch_vector2.t())
        elif self.metric == "cos":
            score = torch.matmul(batch_vector1, batch_vector2.t())
            query_norm = torch.linalg.norm(batch_vector1, dim=-1)
            train_norm = torch.linalg.norm(batch_vector2, dim=-1)
            score /= query_norm.unsqueeze(-1)
            score /= train_norm.unsqueeze(0)
        elif self.metric == "l2":
            # For consistency (higher score means similar), we multiply the score by -1.
            score = -torch.cdist(batch_vector1, batch_vector2, p=2)
        else:
            raise RuntimeError()
        return score.to(self.score_dtype)

    def _perform_forward_pass_and_extract_acts(self, batch: Any) -> torch.Tensor:
        """Perform the forward pass with the given `batch`."""
        _ = self.task.get_train_loss(
            model=self.model,
            batch=batch,
            sample=False,
            reduction="none",
        )
        acts = self._temp_acts.reshape(self._temp_acts.shape[0], -1).to(
            self._similarity_dtype
        )
        self._temp_acts = None
        return acts

    def compute_scores_with_batch(
        self, batch1: Any, batch2: Any, remove_hook: bool = False
    ) -> torch.Tensor:
        """Compute pairwise similarity scores between data points in `batch1` and `batch2`.

        Args:
            batch1 (object):
                The first set of data points from the data loader.
            batch2 (object):
                The second set of data points from the data loader.
            remove_hook (bool, optional):
                If set to True, removes hook after computing the pairwise similarity scores. Defaults to False.
        """
        if self._handle is None:
            error_msg = (
                "The cache has been reset. Call `initialize` to reinitialize the hook."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.model.eval()
        acts1 = self._perform_forward_pass_and_extract_acts(batch1)
        acts2 = self._perform_forward_pass_and_extract_acts(batch2)

        if remove_hook:
            self.remove_hook()

        return self._compute_similarity(acts1, acts2)

    def compute_scores_with_loader(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        remove_hook: bool = True,
    ) -> torch.Tensor:
        """Compute pairwise similarity scores between `test_loader` and `train_loader`.

        Args:
            test_loader (DataLoader):
                The loader with test dataset.
            train_loader (DataLoader):
                The loader with training dataset.
            remove_hook (bool, optional):
                If set to True, removes hook after computing the pairwise similarity scores. Defaults to True.
        """
        if self._handle is None:
            error_msg = (
                "The hook has been removed. Call `initialize` to reinitialize the hook."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.model.eval()
        with torch.no_grad():
            score_table = torch.zeros(
                (len(test_loader.dataset), len(train_loader.dataset)),
                dtype=self.score_dtype,
                device=self.task.device,
                requires_grad=False,
            )

            num_processed_test = 0
            for test_batch in test_loader:
                test_batch_size = self.task.get_batch_size(test_batch)
                test_acts = self._perform_forward_pass_and_extract_acts(test_batch)

                num_processed_train = 0
                for train_batch in train_loader:
                    train_batch_size = self.task.get_batch_size(train_batch)
                    train_acts = self._perform_forward_pass_and_extract_acts(
                        train_batch
                    )

                    current_score = self._compute_similarity(test_acts, train_acts)
                    score_table[
                        num_processed_test : num_processed_test + test_batch_size,
                        num_processed_train : num_processed_train + train_batch_size,
                    ].add_(current_score)
                    num_processed_train += train_batch_size
                num_processed_test += test_batch_size

        if remove_hook:
            self.remove_hook()

        return score_table

    def compute_self_scores_with_loader(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:
        # The notion of self-influence does not exist for RepresentationSimilarityComputer.
        del loader
        error_msg = (
            f"self-score computation is not supported for {self.__class__.__name__}."
        )
        self.logger.error(error_msg)
        raise NotImplementedError(error_msg)
