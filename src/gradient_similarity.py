from typing import Optional

import torch
import torch.nn as nn

from src.abstract_task import AbstractTask


class GradientSimilarityComputer:
    supported_modules = {"Linear", "Conv2d", "BatchNorm2d", "LayerNorm", "Embedding"}
    score_dtype: torch.dtype = torch.float32

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        device: Optional[torch.device] = None,
        metric: str = "dot",
    ) -> None:
        self.model = model
        self.func_params = dict(self.model.named_parameters())
        self.func_buffers = dict(self.model.named_buffers())
        if device is None:
            self.device = next(iter(self.model.parameters())).device
        else:
            self.device = device

        self.task = task
        self.metric = metric
        assert self.metric in ["cos", "dot"]

        supported_param_names = []
        for name, param in self.model.named_parameters():
            if any(
                (
                    name.startswith(module_name)
                    for module_name in self.task.influence_modules()
                )
            ):
                supported_param_names.append(name)
        self.supported_param_names = supported_param_names

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
                device=self.device,
                requires_grad=False,
            )

        def compute_train_loss(_params, _buffers, _batch):
            return self.task.get_train_loss(
                model=self.model,
                batch=_batch,
                parameter_and_buffer_dicts=(_params, _buffers),
                sample=False,
                reduction="sum",
            )

        def compute_measurement(_params, _buffers, _batch):
            return self.task.get_measurement(
                model=self.model,
                batch=_batch,
                parameter_and_buffer_dicts=(_params, _buffers),
                sample=False,
                reduction="sum",
            )

        ft_compute_train_grad = torch.func.grad(compute_train_loss, has_aux=False)
        ft_compute_measurement_grad = torch.func.grad(
            compute_measurement, has_aux=False
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
                for key in valid_grads_dict:
                    if key in self.supported_param_names:
                        valid_grads_dict[key] = valid_grads_dict[key].reshape(
                            valid_grads_dict[key].shape[0], -1
                        )

            num_processed_train = 0
            for train_batch in train_loader:
                train_batch_size = self.task.get_batch_size(train_batch)

                train_grads_dict = torch.func.vmap(
                    ft_compute_train_grad,
                    in_dims=(None, None, 0),
                    randomness="different",
                )(self.func_params, self.func_buffers, train_batch)

                with torch.no_grad():
                    for key in train_grads_dict:
                        if key in self.supported_param_names:
                            train_grads_dict[key] = train_grads_dict[key].reshape(
                                train_grads_dict[key].shape[0], -1
                            )

                current_score = 0.0
                query_sq_norm = 0.0
                train_sq_norm = 0.0
                with torch.no_grad():
                    for name in self.supported_param_names:
                        current_score += (
                            valid_grads_dict[name] @ train_grads_dict[name].t()
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

                v_start = num_processed_valid
                t_start = num_processed_train
                score_table[
                    v_start : v_start + valid_batch_size,
                    t_start : t_start + train_batch_size,
                ].add_(current_score)
                num_processed_train += train_batch_size
            num_processed_valid += valid_batch_size
        return score_table.numpy()
