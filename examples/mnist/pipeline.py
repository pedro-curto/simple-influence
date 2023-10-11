import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision


def construct_mlp(num_inputs: int = 784, num_classes: int = 10) -> nn.Module:
    model = torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, num_classes, bias=True),
    )
    return model


def get_loaders(
    data_name: str,
    eval_batch_size: int = 2048,
    train_indices: Optional[List[int]] = None,
    do_corrupt: bool = False,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    assert data_name in ["mnist", "fmnist"]
    train_batch_size = 128

    if data_name == "mnist":
        train_loader = get_mnist_dataloader(
            batch_size=train_batch_size,
            split="train",
            indices=train_indices,
            do_corrupt=do_corrupt,
        )
        eval_train_loader = get_mnist_dataloader(
            batch_size=eval_batch_size,
            split="eval_train",
            indices=train_indices,
            do_corrupt=do_corrupt,
        )
        valid_loader = get_mnist_dataloader(
            batch_size=eval_batch_size,
            split="valid",
            indices=None,
            do_corrupt=False,
        )
    else:
        train_loader = get_fmnist_dataloader(
            batch_size=train_batch_size,
            split="train",
            indices=train_indices,
            do_corrupt=do_corrupt,
        )
        eval_train_loader = get_fmnist_dataloader(
            batch_size=eval_batch_size,
            split="eval_train",
            indices=train_indices,
            do_corrupt=do_corrupt,
        )
        valid_loader = get_fmnist_dataloader(
            batch_size=eval_batch_size,
            split="valid",
            indices=None,
            do_corrupt=False,
        )
    return train_loader, eval_train_loader, valid_loader


def get_mnist_dataloader(
    batch_size: int = 128,
    split: str = "train",
    indices: List[int] = None,
    do_corrupt: bool = False,
) -> torch.utils.data.DataLoader:
    assert split in ["train", "eval_train", "valid"]

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root="/tmp/mnist/",
        download=True,
        train=split in ["train", "eval_train"],
        transform=transforms,
    )

    if do_corrupt:
        if split == "valid":
            raise NotImplementedError("Corruption on validation dataset not supported.")
        num_corrupt = math.ceil(len(dataset) * 0.1)
        original_targets = copy.deepcopy(dataset.targets[:num_corrupt])
        new_targets = torch.randint(
            0,
            10,
            size=original_targets[:num_corrupt].shape,
            generator=torch.Generator().manual_seed(0),
        )
        offsets = torch.randint(
            1,
            9,
            size=new_targets[new_targets == original_targets].shape,
            generator=torch.Generator().manual_seed(0),
        )
        new_targets[new_targets == original_targets] = (
            new_targets[new_targets == original_targets] + offsets
        ) % 10
        assert (new_targets == original_targets).sum() == 0
        dataset.targets[:num_corrupt] = new_targets

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=0,
        drop_last=split == "train",
        pin_memory=True,
    )


def get_fmnist_dataloader(
    batch_size: int = 128,
    split: str = "train",
    indices: List[int] = None,
    do_corrupt: bool = False,
) -> torch.utils.data.DataLoader:
    assert split in ["train", "eval_train", "valid"]

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    dataset = torchvision.datasets.FashionMNIST(
        root="/tmp/fmnist/",
        download=True,
        train=split in ["train", "eval_train"],
        transform=transforms,
    )

    if do_corrupt:
        if split == "valid":
            raise NotImplementedError("Corruption on validation dataset not supported.")
        num_corrupt = math.ceil(len(dataset) * 0.1)
        original_targets = copy.deepcopy(dataset.targets[:num_corrupt])
        new_targets = torch.randint(
            0,
            10,
            size=original_targets[:num_corrupt].shape,
            generator=torch.Generator().manual_seed(0),
        )
        offsets = torch.randint(
            1,
            9,
            size=new_targets[new_targets == original_targets].shape,
            generator=torch.Generator().manual_seed(0),
        )
        new_targets[new_targets == original_targets] = (
            new_targets[new_targets == original_targets] + offsets
        ) % 10
        assert (new_targets == original_targets).sum() == 0
        dataset.targets[:num_corrupt] = new_targets

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=0,
        drop_last=split == "train",
        pin_memory=True,
    )
