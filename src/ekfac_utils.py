"""Helpers shared by the EK-FAC machinery.

These functions handle the layout / reshaping work that the influence-function
and SOURCE computers need but that doesn't belong on any specific computer
class: extracting activation patches for convolution, flattening per-sample
gradients to module-major layout, and packaging the bias gradient into the
weight gradient via the standard "augmented input" trick.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class InvalidModuleError(Exception):
    """Raised when an unsupported module type is passed to an EK-FAC helper."""


def extract_patches(
    inputs: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> torch.Tensor:
    """Extract sliding patches from a Conv2d input for the K-FAC approximation.

    This is the standard "unfolding" step used by K-FAC for convolutional
    layers (see Grosse and Martens, 2016, https://arxiv.org/pdf/1602.01407.pdf):
    each output spatial location's input patch becomes a row, so the resulting
    tensor can be treated like a Linear layer's activations.

    Args:
        inputs (torch.Tensor):
            `(N, C_in, H, W)` activations entering the convolution.
        kernel_size (tuple):
            `(kH, kW)` kernel size of the conv layer.
        stride (tuple):
            `(sH, sW)` stride of the conv layer.
        padding (tuple):
            `(pH, pW)` padding of the conv layer.

    Returns:
        torch.Tensor: `(N, H_out, W_out, C_in * kH * kW)` patches.
    """
    if padding[0] + padding[1] > 0:
        inputs = torch.nn.functional.pad(
            inputs,
            (padding[1], padding[1], padding[0], padding[0]),
        ).detach()
    inputs = inputs.unfold(2, kernel_size[0], stride[0])
    inputs = inputs.unfold(3, kernel_size[1], stride[1])
    # Move spatial dims so the patch elements end up as the trailing flattened
    # axis (each row is one output location's input patch).
    inputs = inputs.transpose_(1, 2).transpose_(2, 3).contiguous()
    inputs = inputs.view(
        inputs.size(0),
        inputs.size(1),
        inputs.size(2),
        inputs.size(3) * inputs.size(4) * inputs.size(5),
    )
    return inputs


def make_grads_dict_to_matrix(
    module: nn.Module,
    module_name: str,
    grads_dict: Dict[str, torch.Tensor],
    remove_grads: bool = True,
) -> torch.Tensor:
    """Reshape per-parameter gradients into a single (B, ...) matrix per module.

    The output layout depends on the module:
      - `Linear` / `Embedding`: `(B, out, in [+1 if bias])` - bias is
        concatenated as an extra "in" column, matching the augmented-input
        trick used by `extract_activations`.
      - `Conv2d`: `(B, out, in * kH * kW [+1 if bias])` - weight is flattened
        across kernel positions, bias concatenated as an extra column.
      - `LayerNorm` / `BatchNorm2d`: `(B, 2 * num_features)` - weight and bias
        concatenated along the trailing axis.

    Args:
        module (nn.Module):
            Module the gradients belong to. Must be one of `Linear`, `Conv2d`,
            `Embedding`, `LayerNorm`, or `BatchNorm2d`.
        module_name (str):
            Qualified module name (matching keys in `grads_dict`).
        grads_dict (dict):
            Dict mapping parameter names (e.g. `"layer1.weight"`) to per-sample
            gradient tensors.
        remove_grads (bool, optional):
            If `True` (default), delete the consumed keys from `grads_dict` to
            free memory. **Mutates the input dict.**

    Returns:
        torch.Tensor: the reshaped per-sample gradient matrix.

    Raises:
        InvalidModuleError: if `module` is not a supported type.
    """
    if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
        grads_mat = grads_dict[module_name + ".weight"]
        if remove_grads:
            del grads_dict[module_name + ".weight"]
        if module_name + ".bias" in grads_dict:
            grads_mat = torch.cat(
                (grads_mat, grads_dict[module_name + ".bias"].unsqueeze(-1)), -1
            )
            if remove_grads:
                del grads_dict[module_name + ".bias"]
    elif isinstance(module, nn.Conv2d):
        grads_mat = grads_dict[module_name + ".weight"]
        grads_mat = grads_mat.view(grads_mat.size(0), grads_mat.size(1), -1)
        if remove_grads:
            del grads_dict[module_name + ".weight"]
        if module_name + ".bias" in grads_dict:
            grads_mat = torch.cat(
                [grads_mat, grads_dict[module_name + ".bias"].unsqueeze(-1)], -1
            )
            if remove_grads:
                del grads_dict[module_name + ".bias"]
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
        # For LayerNorm / BatchNorm we keep weight and bias as one flat block,
        # since the "full Fisher" branch downstream operates on the
        # concatenated vector directly.
        grads_mat = torch.cat(
            (grads_dict[module_name + ".weight"], grads_dict[module_name + ".bias"]), -1
        )
        if remove_grads:
            del grads_dict[module_name + ".weight"], grads_dict[module_name + ".bias"]
    else:
        raise InvalidModuleError()
    return grads_mat


def extract_activations(
    activations: torch.Tensor,
    module: nn.Module,
    activations_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reshape activations entering a module for covariance accumulation.

    For modules with a bias, an extra column of ones is appended to the
    activations so the same outer product `acts.T @ acts` covers both the
    weight and the bias rows of the K-FAC factor (the standard "augmented
    input" trick).

    Args:
        activations (torch.Tensor):
            Raw pre-activations as captured by a forward hook.
        module (nn.Module):
            The module the activations feed into (`Linear` or `Conv2d`).
        activations_mask (torch.Tensor, optional):
            Multiplicative mask applied to the activations (and the appended
            ones column for the bias) before reshaping. Use this to zero out
            padded positions in Transformer batches so they don't contribute
            to the covariance.

    Returns:
        torch.Tensor: a 2-D `(num_locations, in_dim [+1])` activation matrix
        ready for `acts.T @ acts`.

    Raises:
        InvalidModuleError: if `module` is not Linear or Conv2d.
    """
    if isinstance(module, nn.Linear):
        if (
            activations_mask is not None
            and activations_mask.shape[:-1] == activations.shape[:-1]
        ):
            activations = activations * activations_mask
        reshaped_activations = activations.reshape(-1, activations.shape[-1])
        if module.bias is not None:
            shape = list(reshaped_activations.shape[:-1]) + [1]
            append_term = reshaped_activations.new_ones(shape)
            if (
                activations_mask is not None
                and activations_mask.shape[:-1] == activations.shape[:-1]
            ):
                append_term = append_term * activations_mask.view(-1, 1)
            reshaped_activations = torch.cat(
                [reshaped_activations, append_term], dim=-1
            )
    elif isinstance(module, nn.Conv2d):
        del activations_mask
        reshaped_activations = extract_patches(
            activations, module.kernel_size, module.stride, module.padding
        )
        reshaped_activations = reshaped_activations.view(
            -1, reshaped_activations.size(-1)
        )
        if module.bias is not None:
            shape = list(reshaped_activations.shape[:-1]) + [1]
            reshaped_activations = torch.cat(
                [reshaped_activations, reshaped_activations.new_ones(shape)], dim=-1
            )
    else:
        raise InvalidModuleError()
    return reshaped_activations


def extract_gradients(gradients: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Reshape pseudo-gradients on a module's output for covariance accumulation.

    The output is laid out so that `grads.T @ grads` yields the pseudo-gradient
    covariance `S` from K-FAC (Martens & Grosse 2015).

    Args:
        gradients (torch.Tensor):
            Pseudo-gradients on the module's pre-activations, as captured by a
            backward hook.
        module (nn.Module):
            The module the gradients belong to (`Linear` or `Conv2d`).

    Returns:
        torch.Tensor: a 2-D `(num_locations, out_dim)` gradient matrix.

    Raises:
        InvalidModuleError: if `module` is not Linear or Conv2d.
    """
    if isinstance(module, nn.Linear):
        del module
        reshaped_grads = gradients.reshape(-1, gradients.shape[-1])
        return reshaped_grads
    elif isinstance(module, nn.Conv2d):
        del module
        reshaped_grads = gradients.permute(0, 2, 3, 1)
        reshaped_grads = reshaped_grads.reshape(-1, reshaped_grads.size(-1))
    else:
        raise InvalidModuleError()
    return reshaped_grads
