from typing import Optional

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
        super().__init__(model, task)
        self.func_params = dict(self.model.named_parameters())
        self.func_buffers = dict(self.model.named_buffers())
        self.metric = metric
        assert self.metric in ["cos", "dot"]

        self.supported_param_names = []
        for name, param in self.model.named_parameters():
            if any(
                (
                    name.startswith(module_name)
                    for module_name in self.task.influence_modules()
                )
            ):
                self.supported_param_names.append(name)



    def _validate_inputs(self):
        assert self.metric in ["cos", "dot"]
        assert len(self.supported_param_names) > 0

    def _compute_score(self, valid_grads_dict, train_grads_dict):
        current_score = 0.
        query_sq_norm = 0.
        train_sq_norm = 0.
        with torch.no_grad():
            for name in self.supported_param_names:
                current_score += (
                        torch.matmul(valid_grads_dict[name], train_grads_dict[name].t())
                )
                if self.metric == "cos":
                    query_sq_norm += torch.sum(
                        valid_grads_dict[name] ** 2.0, -1
                    )
                    train_sq_norm += torch.sum(
                        train_grads_dict[name] ** 2.0, -1
                    )

            if self.metric == "cos":
                query_norm = torch.sqrt(query_sq_norm)
                train_norm = torch.sqrt(train_sq_norm)
                current_score /= query_norm.unsqueeze(-1)
                current_score /= train_norm.unsqueeze(0)
        return current_score.to(dtype=self.score_dtype)

    def compute_total_influence(
        self,
        valid_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        self.model.eval()

        with torch.no_grad():
            score_table = torch.zeros(
                (len(valid_loader.dataset), len(train_loader.dataset)),
                dtype=self.score_dtype,
                device=self.task.device,
                requires_grad=False,
            )

        ft_compute_train_grad = torch.func.grad(self._compute_train_loss, argnums=0, has_aux=False)
        ft_compute_measurement_grad = torch.func.grad(
            self._compute_measurement, argnums=0, has_aux=False
        )

        num_processed_valid = 0
        for valid_batch in valid_loader:
            valid_batch_size = self.task.get_batch_size(valid_batch)

            valid_grads_dict = torch.func.vmap(
                ft_compute_measurement_grad,
                in_dims=(None, None, 0),
                randomness="different",
            )(self.func_params, self.func_buffers, valid_batch)

            with torch.no_grad():
                reshaped_valid_grads_dict = {}
                key_list = list(valid_grads_dict.keys())
                for key in key_list:
                    if key in self.supported_param_names:
                        reshaped_valid_grads_dict[key] = valid_grads_dict[key].reshape(
                            valid_batch_size, -1
                        )
                    del valid_grads_dict[key]
                del valid_grads_dict

            num_processed_train = 0
            for train_batch in train_loader:
                train_batch_size = self.task.get_batch_size(train_batch)

                train_grads_dict = torch.func.vmap(
                    ft_compute_train_grad,
                    in_dims=(None, None, 0),
                    randomness="different",
                )(self.func_params, self.func_buffers, train_batch)

                with torch.no_grad():
                    reshaped_train_grads_dict = {}
                    key_list = list(train_grads_dict.keys())
                    for key in key_list:
                        if key in self.supported_param_names:
                            reshaped_train_grads_dict[key] = train_grads_dict[key].reshape(
                                train_batch_size, -1
                            )
                        del train_grads_dict[key]
                    del train_grads_dict

                current_score = self._compute_score(reshaped_valid_grads_dict, reshaped_train_grads_dict)

                v_start = num_processed_valid
                t_start = num_processed_train
                score_table[
                    v_start : v_start + valid_batch_size,
                    t_start : t_start + train_batch_size,
                ].add_(current_score)
                num_processed_train += train_batch_size
            num_processed_valid += valid_batch_size
        return score_table
