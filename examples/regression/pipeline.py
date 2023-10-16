import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def construct_regression_mlp(data_name: str) -> nn.Module:
    assert data_name in ["concrete", "naval", "parkinsons"]
    if data_name == "parkinsons":
        num_inputs = 21
    else:
        num_inputs = 8

    model = torch.nn.Sequential(
        nn.Linear(num_inputs, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 1, bias=True),
    )
    return model


def get_hyperparameters(data_name: str) -> dict:
    if data_name == "concrete":
        lr = 0.03
        wd = 1e-05
    elif data_name == "parkinsons":
        lr = 0.003
        wd = 1e-05
    else:
        raise NotImplementedError()
    return {"lr": lr, "wd": wd}


def get_loaders(
    data_name: str,
    eval_batch_size: int = 4096,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
    data_path: str = "data/",
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    assert data_name in ["concrete", "parkinsons"]
    train_batch_size = 32

    train_loader = get_uci_dataloader(
        data_name=data_name,
        batch_size=train_batch_size,
        split="train",
        indices=train_indices,
        data_path=data_path,
    )
    eval_train_loader = get_uci_dataloader(
        data_name=data_name,
        batch_size=eval_batch_size,
        split="eval_train",
        indices=train_indices,
        data_path=data_path,
    )
    valid_loader = get_uci_dataloader(
        data_name=data_name,
        batch_size=eval_batch_size,
        split="valid",
        indices=valid_indices,
        data_path=data_path,
    )
    return train_loader, eval_train_loader, valid_loader


class UCIDataset(torch.utils.data.Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray) -> None:
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.data_x[index], self.data_y[index]

    def __len__(self) -> int:
        return self.data_x.shape[0]


def get_uci_dataloader(
    data_name: str,
    batch_size: int,
    split: str,
    indices: List[int] = None,
    data_path: str = "data/",
) -> torch.utils.data.DataLoader:
    assert split in ["train", "eval_train", "valid"]

    # Load the dataset from the `.data` file.
    data = np.loadtxt(os.path.join(data_path, data_name + ".data"), delimiter=None)
    data = data.astype(np.float32)

    # Shuffle the dataset.
    seed = np.random.RandomState(0)
    permutation = seed.choice(np.arange(data.shape[0]), data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0:size_train]
    index_val = permutation[size_train:]

    x_train, y_train = data[index_train, :-1], data[index_train, -1]
    x_val, y_val = data[index_val, :-1], data[index_val, -1]

    # Standardize the inputs.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    # Standardize the targets.
    scaler = StandardScaler()
    y_train = np.expand_dims(y_train, -1)
    y_val = np.expand_dims(y_val, -1)
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)

    if split in ["train", "eval_train"]:
        dataset = UCIDataset(
            x_train_scaled.astype(np.float32),
            y_train_scaled.astype(np.float32),
        )
    else:
        dataset = UCIDataset(
            x_val_scaled.astype(np.float32), y_val_scaled.astype(np.float32)
        )

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=0,
        drop_last=split == "train",
    )
