from typing import Optional

import torch
import torch.nn as nn

from src.abstract_task import AbstractTask


class RepresentationSimilarityComputer:
    supported_modules = {"Linear", "Conv2d", "BatchNorm2d", "LayerNorm", "Embedding"}
    score_dtype: torch.dtype = torch.float32

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        metric: str = "l2",
        similarity_dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initializes the class RepresentationsSimilarityComputer."""
        self.model = model

        self.task = task
        self.target_module_name = self.task.representation_modules()
        if self.target_module_name is None:
            raise NotImplementedError(
                "For `RepresentationSimilarityComputer`, you need to specify the module "
                "name by definiting `AbstractTask.representation_modules()`"
            )

        self.metric = metric
        assert self.metric in ["dot", "cos", "l2"]
        self.similarity_dtype = similarity_dtype

        self._handle = None
        self._target_module = None
        self._temp_acts = None
        self.initialize_hooks()

    def initialize_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_module_name:
                self._handle = module.register_forward_hook(self._forward_hook)
                return
        assert NotImplementedError()

    def _forward_hook(
        self, module: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> None:
        del inputs
        # Save the intermediate activations.
        self._temp_acts = outputs.data

    def remove_cache(self):
        self._handle.remove()
        self._handle = None
        self._temp_acts = None

    def compute_total_influence(
        self,
        valid_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        remove_cache: bool = True,
    ) -> torch.Tensor:
        if self._handle is None:
            raise ValueError("Initialize the model")

        self.model.eval()
        with torch.no_grad():
            score_table = torch.zeros(
                (len(valid_loader.dataset), len(train_loader.dataset)),
                dtype=self.score_dtype,
                device=self.task.device,
                requires_grad=False,
            )

            num_processed_valid = 0
            for valid_batch in valid_loader:
                valid_batch_size = self.task.get_batch_size(valid_batch)

                # Perform a forward pass with a valid_batch.
                _ = self.task.get_train_loss(
                    model=self.model,
                    batch=valid_batch,
                    sample=False,
                    reduction="none",
                )
                valid_acts = self._temp_acts.reshape(self._temp_acts.shape[0], -1).to(
                    torch.float64
                )

                num_processed_train = 0
                for train_batch in train_loader:
                    train_batch_size = self.task.get_batch_size(train_batch)

                    # Perform a forward pass with a train_batch.
                    _ = self.task.get_train_loss(
                        model=self.model,
                        batch=train_batch,
                        sample=False,
                        reduction="none",
                    )
                    train_acts = self._temp_acts.reshape(
                        self._temp_acts.shape[0], -1
                    ).to(self.similarity_dtype)

                    if self.metric == "dot":
                        current_score = valid_acts @ train_acts.t()
                    elif self.metric == "cos":
                        current_score = valid_acts @ train_acts.t()
                        query_norm = torch.linalg.norm(valid_acts, dim=-1)
                        train_norm = torch.linalg.norm(train_acts, dim=-1)
                        current_score /= query_norm.unsqueeze(-1)
                        current_score /= train_norm.unsqueeze(0)
                    elif self.metric == "l2":
                        current_score = torch.cdist(valid_acts, train_acts, p=2)
                    else:
                        raise NotImplementedError()
                    current_score = current_score.to(self.score_dtype)

                    v_start = num_processed_valid
                    t_start = num_processed_train
                    score_table[
                        v_start : v_start + valid_batch_size,
                        t_start : t_start + train_batch_size,
                    ].add_(current_score)
                    num_processed_train += train_batch_size
                num_processed_valid += valid_batch_size

        if self.metric == "l2":
            score_table *= -1

        if remove_cache:
            self.remove_cache()

        return score_table
