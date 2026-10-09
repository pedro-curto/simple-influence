"""Gradient-similarity TDA baselines.

For each pair of query / training points, this computer measures the
similarity (dot product or cosine) between their per-sample loss gradients.
With ``metric="dot"`` it is equivalent to an influence function with an
identity Hessian; see https://arxiv.org/pdf/2006.04528.pdf for a discussion
of representation- and gradient-similarity baselines for TDA.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask


class GradientSimilarityComputer(AbstractComputer):
    """Compute pairwise gradient-similarity attribution scores.

    Attributes:
        metric (str): ``"dot"`` or ``"cos"``.
        supported_param_names (List[str]): qualified parameter names
            (``module_name.weight`` / ``module_name.bias``) whose gradients
            contribute to the score, derived from ``task.influence_modules()``.
    """

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        metric: str = "dot",
    ) -> None:
        """Initialize the gradient-similarity computer.

        Args:
            model (nn.Module):
                Model whose per-sample gradients are compared.
            task (AbstractTask):
                Task adapter describing the loss / measurement and the modules
                to attribute over.
            metric (str, optional):
                Similarity metric, ``"dot"`` (default) or ``"cos"``.

        Raises:
            NotImplementedError: if ``metric`` is neither ``"dot"`` nor ``"cos"``.
            AttributeError: if no parameter matches any module in
                ``task.influence_modules()``.
        """
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        self.func_params = dict(self.model.named_parameters())
        self.func_buffers = dict(self.model.named_buffers())

        self.metric = metric
        if self.metric not in ["dot", "cos"]:
            error_msg = (
                f"Not supported metric {self.metric} for `{self.__class__.__name__}`."
            )
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        self.supported_param_names = []
        for name, param in self.model.named_parameters():
            if any(
                (
                    name.startswith(module_name)
                    for module_name in self.task.influence_modules()
                )
            ):
                self.logger.info(f"Found parameter with name {name}.")
                self.supported_param_names.append(name)
        if len(self.supported_param_names) == 0:
            error_msg = f"Cannot find any parameters for modules {self.task.influence_modules()}."
            self.logger.error(error_msg)
            raise AttributeError(error_msg)

    def _compute_similarity(
        self, grads_dict1: Dict[str, torch.Tensor], grads_dict2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the pairwise similarity matrix between two gradient dicts.

        Args:
            grads_dict1 (dict):
                Per-parameter per-sample gradients for the first set, each of
                shape ``(B1, num_params)``.
            grads_dict2 (dict):
                Same for the second set, with shape ``(B2, num_params)``.

        Returns:
            torch.Tensor: ``(B1, B2)`` similarity matrix.
        """
        total_score = 0.0
        sq_norm1 = 0.0
        sq_norm2 = 0.0
        with torch.no_grad():
            for name in self.supported_param_names:
                if isinstance(total_score, float):
                    total_score = torch.matmul(grads_dict1[name], grads_dict2[name].t())
                else:
                    total_score.addmm_(grads_dict1[name], grads_dict2[name].t())

                if self.metric == "cos":
                    sq_norm1 += torch.sum(grads_dict1[name] ** 2.0, -1)
                    sq_norm2 += torch.sum(grads_dict2[name] ** 2.0, -1)

            if self.metric == "cos":
                norm1 = torch.sqrt(sq_norm1)
                norm2 = torch.sqrt(sq_norm2)
                total_score /= norm1.unsqueeze(-1)
                total_score /= norm2.unsqueeze(0)
        return total_score.to(dtype=self.score_dtype)

    def _get_reshaped_grads_dict(
        self, batch: Any, use_measurement: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample gradients via vmap and reshape to ``(B, -1)`` per param.

        Args:
            batch (Any):
                A single batch.
            use_measurement (bool, optional):
                If ``True``, differentiate the measurement; otherwise the
                training loss.

        Returns:
            Dict[str, torch.Tensor]: gradients keyed by parameter name.
        """
        batch_size = self.task.get_batch_size(batch)
        grads_dict = torch.func.vmap(
            self._compute_measurement_grad()
            if use_measurement
            else self._compute_train_loss_grad(),
            in_dims=(None, None, 0),
            randomness="different",
        )(self.func_params, self.func_buffers, batch)
        with torch.no_grad():
            reshaped_grads_dict = {}
            key_list = list(grads_dict.keys())
            for key in key_list:
                if key in self.supported_param_names:
                    reshaped_grads_dict[key] = grads_dict[key].reshape(batch_size, -1)
                # Drop references to unneeded gradients to free memory.
                del grads_dict[key]
            del grads_dict
        return reshaped_grads_dict

    def compute_scores_with_batch(self, batch1: Any, batch2: Any) -> torch.Tensor:
        """Compute pairwise similarity scores between two batches.

        Args:
            batch1 (Any):
                First batch.
            batch2 (Any):
                Second batch.

        Returns:
            torch.Tensor: ``(|batch1|, |batch2|)`` similarity matrix.
        """
        self.model.eval()
        reshaped_grads_dict1 = self._get_reshaped_grads_dict(
            batch1, use_measurement=False
        )
        reshaped_grads_dict2 = self._get_reshaped_grads_dict(
            batch2, use_measurement=False
        )
        with torch.no_grad():
            current_score = self._compute_similarity(
                reshaped_grads_dict1, reshaped_grads_dict2
            )
        return current_score

    def compute_scores_with_loader(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Compute pairwise similarity scores between two loaders.

        Args:
            test_loader (DataLoader):
                Loader yielding query (test) data.
            train_loader (DataLoader):
                Loader yielding training data.

        Returns:
            torch.Tensor: ``(num_test, num_train)`` similarity matrix.
        """
        self.model.eval()
        score_table = torch.zeros(
            (len(test_loader.dataset), len(train_loader.dataset)),
            dtype=self.score_dtype,
            device=self.task.device,
            requires_grad=False,
        )

        num_processed_test = 0
        for test_batch in test_loader:
            test_batch_size = self.task.get_batch_size(test_batch)
            reshaped_test_grads_dict = self._get_reshaped_grads_dict(
                test_batch, use_measurement=True
            )

            num_processed_train = 0
            for train_batch in train_loader:
                train_batch_size = self.task.get_batch_size(train_batch)
                reshaped_train_grads_dict = self._get_reshaped_grads_dict(
                    train_batch, use_measurement=False
                )

                current_score = self._compute_similarity(
                    reshaped_test_grads_dict, reshaped_train_grads_dict
                )
                score_table[
                    num_processed_test : num_processed_test + test_batch_size,
                    num_processed_train : num_processed_train + train_batch_size,
                ].add_(current_score)
                num_processed_train += train_batch_size
            num_processed_test += test_batch_size
        return score_table

    def compute_self_scores_with_loader(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Compute self-similarity scores for every data point in a loader.

        Only supported for ``metric="dot"``; for cosine the diagonal is
        trivially 1 and offers no information.

        Args:
            loader (DataLoader):
                Loader yielding the data points whose self-similarity is
                computed.

        Returns:
            torch.Tensor: 1-D tensor of self-similarity scores, same length
            as the dataset.

        Raises:
            RuntimeError: if ``metric != "dot"``.
        """
        if self.metric != "dot":
            error_msg = "Self-scores are only supported for dot similarity."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        scores = []
        for batch in loader:
            batch_size = self.task.get_batch_size(batch)
            current_score = torch.zeros(
                (batch_size,),
                dtype=self.score_dtype,
                device=self.task.device,
                requires_grad=False,
            )
            grads_dict = self._get_reshaped_grads_dict(batch, use_measurement=False)

            with torch.no_grad():
                for name in self.supported_param_names:
                    current_score.add_(torch.square(grads_dict[name]).sum(dim=-1))
                scores.append(current_score)
        return torch.cat(scores)
