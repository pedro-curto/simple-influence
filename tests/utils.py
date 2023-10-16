from typing import List, Tuple, Union

import torch
import torchvision
from torch import nn
from torch.utils import data

from examples.glue.task import TextClassificationTask
from src.abstract_task import AbstractTask, validate_task
from tests.dummy_tasks import ConvBNTask, ConvTask, MLPTask


def prepare_test(
    test_name: str,
    device: Union[torch.device, str] = torch.device("cpu"),
    train_size: int = 32,
    valid_size: int = 16,
    train_batch_size: int = 4,
    valid_batch_size: int = 2,
    do_not_pad: bool = False,
    seed: int = 0,
) -> Tuple[
    nn.Module, torch.utils.data.DataLoader, torch.utils.data.DataLoader, AbstractTask
]:
    """Obtain model, data loaders, and task based on specified test configuration.

    This function supports four different test scenarios, each identified by its `test_name`:
    1. "mlp": Regression task using a 2 hidden-layer MLP.
    2. "conv: Classification task with a convolutional neural network.
    3. "conv_bn": Classification task with a convolutional neural network (the architecture uses batch norm).
    4. "glue": Text classification task utilizing the BERT model.
    5. "wiki": Language modeling task with GPT-2.

    Args:
        test_name (str):
            Identifier for the test scenario.
        device (torch.device, str, optional):
            Computational device for the test. Defaults to cpu.
        train_size (int, optional):
            Size of the synthetic training dataset. Defaults to 32.
        valid_size (int, optional):
            Size of the synthetic validation dataset. Defaults to 16.
        train_batch_size (int, optional):
            Batch size for the training data loader. Defaults to 4.
        valid_batch_size (int, optional):
            Batch size for the validation data loader. Defaults to 4.
        do_not_pad (bool, optional):
            If set, avoids padding batches for Transformer-based models. Defaults to False.
        seed (int, optional):
            Random seed for model initialization and other stochastic operations. Defaults to 0.
    """
    if test_name == "mlp":
        model = make_dummy_mlp_module(seed=seed)
        train_loader = make_dummy_regression_loader(
            batch_size=train_batch_size, num_data=train_size, seed=seed
        )
        valid_loader = make_dummy_regression_loader(
            batch_size=valid_batch_size, num_data=valid_size, seed=seed
        )
        task = MLPTask(device=device)
    elif test_name == "conv":
        model = make_dummy_conv_module(seed=seed)
        train_loader = make_dummy_classification_loader(
            batch_size=train_batch_size, num_data=train_size, seed=seed
        )
        valid_loader = make_dummy_classification_loader(
            batch_size=valid_batch_size, num_data=valid_size, seed=seed
        )
        task = ConvTask(device=device)
    elif test_name == "conv_bn":
        model = make_dummy_conv_bn_module(seed=seed)
        train_loader = make_dummy_classification_loader(
            batch_size=train_batch_size, num_data=train_size, seed=seed
        )
        valid_loader = make_dummy_classification_loader(
            batch_size=valid_batch_size, num_data=valid_size, seed=seed
        )
        task = ConvBNTask(device=device)
    elif test_name == "glue":
        model = make_glue_module(seed=seed)
        train_loader = make_qnli_loader(
            batch_size=train_batch_size,
            num_data=train_size,
            do_not_pad=do_not_pad,
            seed=seed,
        )
        valid_loader = make_qnli_loader(
            batch_size=valid_batch_size,
            num_data=valid_size,
            do_not_pad=do_not_pad,
            seed=seed,
        )
        task = TextClassificationTask(device=device)
    elif test_name == "wiki":
        model = make_wiki_module(seed=seed)
        train_loader = make_wiki_loader(
            batch_size=train_batch_size,
            num_data=train_size,
            seed=seed,
        )
        valid_loader = make_wiki_loader(
            batch_size=valid_batch_size,
            num_data=valid_size,
            seed=seed,
        )
        task = TextClassificationTask(device=device)
    else:
        raise NotImplementedError(f"{test_name} is not a valid test configuration.")
    return model.to(device=device), train_loader, valid_loader, task


def make_dummy_mlp_module(bias: bool = True, seed: int = 0) -> nn.Module:
    """Creates an MLP model.

    Args:
        bias (str, optional):
            If set, use bias term. Defaults to True.
        seed (int, optional):
            Random seed for model initialization. Defaults to 0.
    """
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(10, 16, bias=bias),
        nn.ReLU(),
        nn.Linear(16, 16, bias=bias),
        nn.ReLU(),
        nn.Linear(16, 1, bias=bias),
    )


def make_dummy_regression_loader(
    batch_size: int, num_data: int, seed: int = 0
) -> torch.utils.data.DataLoader:
    """Creates a synthetic regression dataset and return the data loader.

    Args:
        batch_size (int):
            Batch size for the data loader.
        num_data (int):
            Size of the synthetic dataset.
        seed (int, optional):
            Random seed for dataset generation. Defaults to 0.
    """
    torch.manual_seed(seed)
    dataset = data.TensorDataset(
        torch.randn((num_data, 10), dtype=torch.float32),
        torch.randint(low=-5, high=5, size=(num_data, 1), dtype=torch.float32),
    )
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def make_dummy_conv_module(bias: bool = True, seed: int = 0) -> nn.Module:
    """Creates an CNN model.

    Args:
        bias (str, optional):
            If set, use bias term. Defaults to True.
        seed (int, optional):
            Random seed for model initialization. Defaults to 0.
    """
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, 1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(4, 8, 3, 1, bias=bias),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1152, 5, bias=bias),
    )


def make_dummy_conv_bn_module(bias: bool = True, seed: int = 0) -> nn.Module:
    """Creates an CNN model (with batch normalization).

    Args:
        bias (str, optional):
            If set, use bias term. Defaults to True.
        seed (int, optional):
            Random seed for model initialization. Defaults to 0.
    """
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


def make_dummy_classification_loader(
    batch_size: int, num_data: int, seed: int = 0
) -> torch.utils.data.DataLoader:
    """Creates a synthetic classification image dataset and return the data loader.

    Args:
        batch_size (int):
            Batch size for the data loader.
        num_data (int):
            Size of the synthetic dataset.
        seed (int, optional):
            Random seed for dataset generation. Defaults to 0.
    """
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
    """Creates a GLUE model.

    Args:
        seed (int, optional):
            Random seed for model initialization. Defaults to 0.
    """
    from examples.glue.pipeline import construct_model

    torch.manual_seed(seed)
    return construct_model(data_name="qnli")


def make_qnli_loader(
    batch_size: int,
    num_data: int,
    do_not_pad: bool = False,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Load the QNLI dataset and return the data loader.

    Args:
        batch_size (int):
            Batch size for the data loader.
        num_data (int):
            Size of the synthetic dataset.
        do_not_pad (bool, optional):
            If set, avoids padding batches for Transformer-based models. Defaults to False.
        seed (int, optional):
            Random seed for dataset generation. Defaults to 0.
    """
    from examples.glue.pipeline import get_dataloader

    torch.manual_seed(seed)
    loader = get_dataloader(
        data_name="qnli",
        batch_size=batch_size,
        split="eval_train",
        indices=list(range(num_data)),
        do_not_pad=do_not_pad,
    )
    return loader


def make_wiki_module(seed: int = 0) -> nn.Module:
    """Creates a wiki model.

    Args:
        seed (int, optional):
            Random seed for model initialization. Defaults to 0.
    """
    from examples.wiki.pipeline import construct_model

    torch.manual_seed(seed)
    return construct_model()


def make_wiki_loader(
    batch_size: int,
    num_data: int,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Load the QNLI dataset and return the data loader.

    Args:
        batch_size (int):
            Batch size for the data loader.
        num_data (int):
            Size of the synthetic dataset.
        seed (int, optional):
            Random seed for dataset generation. Defaults to 0.
    """
    from examples.wiki.pipeline import get_wiki_dataloader

    torch.manual_seed(seed)
    loader = get_wiki_dataloader(
        batch_size=batch_size,
        split="eval_train",
        indices=list(range(num_data)),
    )
    return loader


def parameters_to_vector(parameters: List[torch.Tensor]) -> torch.Tensor:
    """Given a list of torch Tensors, return the vectorized version.

    Args:
        parameters (list):
            A list of PyTorch tesnors to vectorize.
    """
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)


def get_num_params(model: torch.nn.Module) -> int:
    """Return the number of parameters, given a PyTorch module.

    Args:
        model (nn.Module):
            PyTorch module.
    """
    return parameters_to_vector(list(model.parameters())).numel()


def check_model_equivalence(model1: nn.Module, model2: nn.Module) -> bool:
    """Return True if two PyTorch modules have equivalent parameters and buffers
    (e.g., batch norm statistics).

    Args:
        model1 (nn.Module):
            The first module for comparison.
        model2 (AbstractTask):
            The second module for comparison.
    """
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
