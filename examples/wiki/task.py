from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.abstract_task import AbstractTask

BATCH_DTYPE = Dict[str, torch.Tensor]


class LanguageModelTask(AbstractTask):
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
        if parameter_and_buffer_dicts is None:
            inputs = (
                batch["input_ids"].to(self.device),
                batch["token_type_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            outputs = model(*inputs)
        else:
            params, buffers = parameter_and_buffer_dicts
            outputs = torch.func.functional_call(
                model,
                (params, buffers),
                args=(
                    batch["input_ids"].unsqueeze(0).to(self.device),
                    batch["token_type_ids"].unsqueeze(0).to(self.device),
                    batch["attention_mask"].unsqueeze(0).to(self.device),
                ),
            )
            batch["labels"] = batch["labels"].unsqueeze(0).to(self.device)

        if not sample:
            return F.cross_entropy(
                outputs, batch["labels"].to(self.device), reduction=reduction
            )
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
        return batch["labels"].shape[0]

    def get_activation_masks(self, batch: Any) -> Optional[torch.Tensor]:
        return batch["attention_mask"].unsqueeze(-1).to(self.device)
