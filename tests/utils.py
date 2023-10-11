from typing import List, Tuple

import torch
import torchvision
from torch import nn
from torch.utils import data

from examples.glue.pipeline import construct_model, get_dataloader
from examples.glue.task import TextClassificationTask
from src.abstract_task import AbstractTask
from tests.dummy_tasks import ConvBNTask, ConvTask, MLPTask


def prepare_test(
    test_name: str,
    device: torch.device = torch.device("cpu"),
    train_size: int = 32,
    valid_size: int = 16,
    train_batch_size: int = 4,
    valid_batch_size: int = 2,
    do_not_pad: bool = False,
    seed: int = 0,
) -> Tuple[
    nn.Module, torch.utils.data.DataLoader, torch.utils.data.DataLoader, AbstractTask
]:
    if test_name == "mlp":
        model = make_dummy_linear_module(seed=seed)
        train_loader = make_dummy_linear_loader(
            batch_size=train_batch_size, num_data=train_size, seed=seed
        )
        valid_loader = make_dummy_linear_loader(
            batch_size=valid_batch_size, num_data=valid_size, seed=seed
        )
        task = MLPTask(device=device)
    elif test_name == "conv":
        model = make_dummy_conv_module(seed=seed)
        train_loader = make_dummy_conv_loader(
            batch_size=train_batch_size, num_data=train_size, seed=seed
        )
        valid_loader = make_dummy_conv_loader(
            batch_size=valid_batch_size, num_data=valid_size, seed=seed
        )
        task = ConvTask(device=device)
    elif test_name == "conv_bn":
        model = make_dummy_conv_bn_module(seed=seed)
        train_loader = make_dummy_conv_loader(
            batch_size=train_batch_size, num_data=train_size, seed=seed
        )
        valid_loader = make_dummy_conv_loader(
            batch_size=valid_batch_size, num_data=valid_size, seed=seed
        )
        task = ConvBNTask(device=device)
    elif test_name == "transformer":
        model = make_glue_module(seed=seed)
        train_loader = make_dummy_glue_loader(
            batch_size=train_batch_size,
            num_data=train_size,
            do_not_pad=do_not_pad,
            seed=seed,
        )
        valid_loader = make_dummy_glue_loader(
            batch_size=valid_batch_size,
            num_data=valid_size,
            do_not_pad=do_not_pad,
            seed=seed,
        )
        task = TextClassificationTask(device=device)
    else:
        raise NotImplementedError
    return model.to(device=device), train_loader, valid_loader, task


def make_dummy_linear_module(bias: bool = False, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(10, 16, bias=bias),
        nn.ReLU(),
        nn.Linear(16, 16, bias=bias),
        nn.ReLU(),
        nn.Linear(16, 1, bias=bias),
    )


def make_dummy_linear_loader(
    batch_size: int = 1, num_data: int = 16, seed: int = 0
) -> torch.utils.data.DataLoader:
    torch.manual_seed(seed)
    dataset = data.TensorDataset(
        torch.randn((num_data, 10), dtype=torch.float32),
        torch.randint(low=-5, high=5, size=(num_data, 1), dtype=torch.float32),
    )
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def make_dummy_conv_module(bias: bool = False, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, 1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(4, 8, 3, 1, bias=bias),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1152, 5, bias=bias),
    )


def make_dummy_conv_bn_module(bias: bool = False, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, 1, bias=bias),
        nn.ReLU(),
        nn.BatchNorm2d(4),
        nn.Conv2d(4, 8, 3, 1, bias=bias),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Flatten(),
        nn.Linear(1152, 5, bias=bias),
    )


def make_dummy_conv_loader(
    batch_size: int = 1, num_data: int = 16, seed: int = 0
) -> torch.utils.data.DataLoader:
    torch.manual_seed(seed)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.FakeData(
        size=num_data, image_size=(3, 16, 16), num_classes=5, transform=transform
    )
    return data.DataLoader(dataset, batch_size=batch_size)


def make_glue_module(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return construct_model(data_name="qnli")


def make_dummy_glue_loader(
    batch_size: int = 1,
    num_data: int = 16,
    do_not_pad: bool = False,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    torch.manual_seed(seed)

    loader = get_dataloader(
        data_name="qnli",
        batch_size=batch_size,
        split="eval_train",
        indices=list(range(num_data)),
        do_not_pad=do_not_pad,
    )
    return loader


def parameters_to_vector(parameters: List[torch.Tensor]) -> torch.Tensor:
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)


def get_num_params(model: torch.nn.Module) -> int:
    return parameters_to_vector(list(model.parameters())).numel()


def check_model_equivalence(model1: nn.Module, model2: nn.Module) -> bool:
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())

    model1_buffers = list(model1.buffers())
    model2_buffers = list(model2.buffers())

    if len(model1_params) != len(model2_params):
        return False

    for param1, param2 in zip(model1_params, model2_params):
        if not torch.allclose(param1, param2):
            return False

    if len(model1_buffers) != len(model2_buffers):
        return False

    for buffer1, buffer2 in zip(model1_buffers, model2_buffers):
        if not torch.allclose(buffer1, buffer2):
            return False

    return True
