from typing import Optional, Tuple

import torch
import torch.nn as nn


class InvalidModuleError(Exception):
    pass


def extract_patches(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> torch.Tensor:
    # Patch extraction for KFC approximations.
    if padding[0] + padding[1] > 0:
        x = torch.nn.functional.pad(
            x,
            (padding[1], padding[1], padding[0], padding[0]),
        ).data
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0),
        x.size(1),
        x.size(2),
        x.size(3) * x.size(4) * x.size(5),
    )
    return x


def make_grad_to_matrix(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        p_grad_mat = module.weight.grad.view(module.weight.grad.size(0), -1)
        if module.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, module.bias.grad.view(-1, 1)], 1)
    else:
        p_grad_mat = module.weight.grad
        if module.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, module.bias.grad.view(-1, 1)], 1)
    return p_grad_mat


def make_grad_dict_to_matrix(module, module_name, grad_dict):
    if isinstance(module, nn.Conv2d):
        p_grad_mat = grad_dict[module_name + ".weight"]
        p_grad_mat = p_grad_mat.view(p_grad_mat.size(0), p_grad_mat.size(1), -1)
        if module_name + ".bias" in grad_dict:
            p_grad_mat = torch.cat(
                [p_grad_mat, grad_dict[module_name + ".bias"].unsqueeze(-1)], 1
            )
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
        p_grad_mat = torch.cat(
            (grad_dict[module_name + ".weight"], grad_dict[module_name + ".bias"]), -1
        )
    elif isinstance(module, nn.Embedding):
        p_grad_mat = grad_dict[module_name + ".weight"]
    elif isinstance(module, nn.Linear):
        p_grad_mat = grad_dict[module_name + ".weight"]
        if module_name + ".bias" in grad_dict:
            p_grad_mat = torch.cat(
                (p_grad_mat, grad_dict[module_name + ".bias"].unsqueeze(-1)), -1
            )
    else:
        raise InvalidModuleError()
    return p_grad_mat


class ActivationHandler:
    @classmethod
    def extract_activations(
        cls,
        activations: torch.Tensor,
        module: nn.Module,
        activations_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return cls.__call__(activations, module, activations_mask)

    @classmethod
    def __call__(
        cls,
        activations: torch.Tensor,
        module: nn.Module,
        activations_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(module, nn.Linear):
            reshaped_activations = cls.linear(activations, module, activations_mask)
        elif isinstance(module, nn.Conv2d):
            reshaped_activations = cls.conv2d(activations, module, activations_mask)
        else:
            raise InvalidModuleError()
        return reshaped_activations

    @staticmethod
    def conv2d(
        activations: torch.Tensor,
        module: nn.Module,
        activations_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del activations_mask

        reshaped_acts = extract_patches(
            activations, module.kernel_size, module.stride, module.padding
        )
        # spatial_size = acts.size(1) * acts.size(2)
        reshaped_acts = reshaped_acts.view(-1, reshaped_acts.size(-1))
        if module.bias is not None:
            shape = list(reshaped_acts.shape[:-1]) + [1]
            reshaped_acts = torch.cat(
                [reshaped_acts, reshaped_acts.new_ones(shape)], dim=-1
            )
        # acts = acts / spatial_size
        # v2_acts = F.unfold(
        #     acts, module.kernel_size, padding=module.padding, stride=module.stride
        # )
        # v2_acts = (
        #     v2_acts.data.permute(1, 0, 2).contiguous().view(v2_acts.shape[1], -1).t()
        # )
        #
        # assert torch.allclose(v2_acts, reshaped_acts / spatial_size)
        return reshaped_acts

    @staticmethod
    def linear(
        activations: torch.Tensor,
        module: nn.Module,
        activations_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            activations_mask is not None
            and activations_mask.shape[:-1] == activations.shape[:-1]
        ):
            activations *= activations_mask
        reshaped_activations = activations.reshape(-1, activations.shape[-1])
        if module.bias is not None:
            shape = list(reshaped_activations.shape[:-1]) + [1]

            append_term = reshaped_activations.new_ones(shape)
            if (
                activations_mask is not None
                and activations_mask.shape[:-1] == activations.shape[:-1]
            ):
                append_term *= activations_mask.view(-1, 1)
            reshaped_activations = torch.cat(
                [reshaped_activations, append_term], dim=-1
            )
        return reshaped_activations


class GradientHandler:
    @classmethod
    def extract_grads(cls, grads: torch.Tensor, module: nn.Module) -> torch.Tensor:
        return cls.__call__(grads, module)

    @classmethod
    def __call__(cls, grads: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        if isinstance(layer, nn.Conv2d):
            reshaped_grads = cls.conv2d(grads, layer)
        elif isinstance(layer, nn.Linear):
            reshaped_grads = cls.linear(grads, layer)
        else:
            raise NotImplementedError()
        return reshaped_grads

    @staticmethod
    def conv2d(grads: torch.Tensor, module: nn.Module) -> torch.Tensor:
        del module
        reshaped_grads = grads.permute(0, 2, 3, 1)
        reshaped_grads = reshaped_grads.reshape(-1, reshaped_grads.size(-1))
        return reshaped_grads

    @staticmethod
    def linear(grads: torch.Tensor, module: nn.Module) -> torch.Tensor:
        del module
        reshaped_grads = grads.reshape(-1, grads.shape[-1])
        return reshaped_grads
