from typing import Any, Dict, List, Optional, Union

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
                batch["attention_mask"].to(self.device),
            )
            lm_logits = model(*inputs)
        else:
            params, buffers = parameter_and_buffer_dicts
            lm_logits = torch.func.functional_call(
                model,
                (params, buffers),
                args=(
                    batch["input_ids"].unsqueeze(0).to(self.device),
                    batch["attention_mask"].unsqueeze(0).to(self.device),
                ),
            )
            batch["labels"] = batch["labels"].unsqueeze(0).to(self.device)

        batch_size = lm_logits.shape[0]
        shift_logits = lm_logits[..., :-1, :].contiguous()

        if not sample:
            labels = batch["labels"].to(self.device)
            shift_labels = labels[..., 1:].contiguous()
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            summed_loss = F.cross_entropy(
                reshaped_shift_logits, shift_labels.view(-1), reduction="sum"
            )
        else:
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            with torch.no_grad():
                probs = torch.nn.functional.softmax(reshaped_shift_logits, dim=-1)
                sampled_labels = torch.multinomial(
                    probs, num_samples=1, generator=self.generator
                ).flatten()
            summed_loss = F.cross_entropy(
                reshaped_shift_logits, sampled_labels.detach(), reduction="sum"
            )

        if reduction == "sum":
            return summed_loss
        elif reduction == "mean":
            return summed_loss / batch_size
        else:
            raise NotImplementedError("Not supported reduction provided.")

    def get_measurement(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        # Alternatively, we can provide the conditional log-likelihood (given the prompt and completion).
        return self.get_train_loss(
            model, batch, parameter_and_buffer_dicts, sample, reduction
        )

    def get_batch_size(self, batch: BATCH_DTYPE) -> int:
        return batch["labels"].shape[0]

    def influence_modules(self) -> List[str]:
        total_modules = []

        # Add attention layers:
        for i in range(12):
            total_modules.append(f"model.transformer.h.{i}.attn.c_attn")
            total_modules.append(f"model.transformer.h.{i}.attn.c_proj")

        # Add MLP layers:
        for i in range(12):
            total_modules.append(f"model.transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"model.transformer.h.{i}.mlp.c_proj")

        return total_modules

    def representation_module(self) -> str:
        return "model.transformer.ln_f"

    def get_activation_masks(self, batch: Any) -> Optional[torch.Tensor]:
        return batch["attention_mask"].unsqueeze(-1).to(self.device)
