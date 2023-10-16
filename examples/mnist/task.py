from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.abstract_task import AbstractTask

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]


class ClassificationTask(AbstractTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def get_train_loss(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        images, labels = batch

        if parameter_and_buffer_dicts is None:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
        else:
            images = images.unsqueeze(0).to(self.device)
            labels = labels.unsqueeze(0).to(self.device)
            params, buffers = parameter_and_buffer_dicts
            outputs = torch.func.functional_call(model, (params, buffers), (images,))

        if not sample:
            return F.cross_entropy(outputs, labels.to(self.device), reduction=reduction)
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                sampled_labels = torch.multinomial(
                    probs, num_samples=1, generator=self.generator
                ).flatten()
            return F.cross_entropy(
                outputs, sampled_labels.detach(), reduction=reduction
            )

    def get_measurement(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        return self.get_train_loss(
            model, batch, parameter_and_buffer_dicts, sample, reduction
        )

    def get_batch_size(self, batch: BATCH_DTYPE) -> int:
        images, _ = batch
        return images.shape[0]

    def influence_modules(self) -> List[str]:
        return ["1", "3", "5", "7"]

    def representation_module(self) -> str:
        return "5"
