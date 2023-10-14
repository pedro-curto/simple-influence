from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask


class GradientSimilarityComputer(AbstractComputer):
    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        metric: str = "dot",
    ) -> None:
        """Initializes the `GradientSimilarityComputer` class.

        This class performs TDA using similarity between gradients based on a specified metric
        (for more details, see https://arxiv.org/pdf/2006.04528.pdf). When `metric = "dot"`, it can
        be seen as influence function computation with identity Hessian approximation.

        Args:
            model (nn.Module):
                The PyTorch model for which representations are computed.
            task (AbstractTask):
                The task for the pipeline.
            metric (str, optional):
                The metric used to measure similarity. Supported metrics include "dot"
                and "cos". Defaults to "dot".
        """
        super().__init__(model, task)
        self.func_params = dict(self.model.named_parameters())
        self.func_buffers = dict(self.model.named_buffers())

        self.metric = metric
        if self.metric not in ["dot", "cos"]:
            error_msg = (
                f"Not supported metric {self.metric} for `GradientSimilarityComputer`."
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
        """Given `grads_dict1` and `grads_dict2` which contains the dictionary with key name and value the gradients
        (`batch_size x dim`), return the pairwise gradient similarities."""
        # Perform lazy initializations.
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
                # Remove references to unncessary gradients.
                del grads_dict[key]
            del grads_dict
        return reshaped_grads_dict

    def compute_pairwise_influence(self, batch1: Any, batch2: Any) -> torch.Tensor:
        """Compute pairwise influence scores between data points in `batch1` and `batch2`.

        Args:
            batch1 (object):
                The first set of data points from the data loader.
            batch2 (object):
                The second set of data points from the data loader.
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

    def compute_total_influence(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Compute pairwise influence scores between `test_loader` and `train_loader`.

        Args:
            test_loader (DataLoader):
                The loader with test dataset.
            train_loader (DataLoader):
                The loader with training dataset.
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

    def compute_self_influence(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:
        if self.metric == "cos":
            self.logger.warning("Self-influence scores with cos metric is simply 1.")

        scores = []
        for batch in loader:
            batch_size = self.task.get_batch_size(batch)
            grads_dict = self._get_reshaped_grads_dict(batch, use_measurement=False)

            pass
