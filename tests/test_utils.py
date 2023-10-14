import pytest
import torch

from tests.utils import (
    make_dummy_classification_loader,
    make_dummy_conv_bn_module,
    make_dummy_conv_module,
    make_dummy_mlp_module,
    make_dummy_regression_loader,
    make_glue_module,
    make_qnli_loader,
)


def test_mlp() -> None:
    # Test "mlp" test scenario.
    loader = make_dummy_regression_loader(batch_size=1, num_data=8)
    model = make_dummy_mlp_module()

    inputs, targets = next(iter(loader))
    outputs = model(inputs)
    assert outputs is not None


def test_conv() -> None:
    # Test "conv" test scenario.
    loader = make_dummy_classification_loader(batch_size=1, num_data=8)
    model = make_dummy_conv_module()

    inputs, targets = next(iter(loader))
    outputs = model(inputs)
    assert outputs is not None


def test_conv_bn() -> None:
    # Test "conv_bn" test scenario.
    loader = make_dummy_classification_loader(batch_size=1, num_data=8)
    model = make_dummy_conv_bn_module()

    inputs, targets = next(iter(loader))
    outputs = model(inputs)
    assert outputs is not None


@pytest.mark.smoke
def test_glue() -> None:
    # Test "glue" test scenario.
    loader = make_qnli_loader(batch_size=1, num_data=8)
    model = make_glue_module()

    batch = next(iter(loader))
    outputs = model(
        batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]
    )
    assert outputs is not None

    loader = make_qnli_loader(batch_size=1, num_data=8, do_not_pad=True)
    batch = next(iter(loader))
    no_pad_outputs = model(
        batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]
    )
    assert no_pad_outputs is not None

    # When `batch_size = 1`, the outputs should be the same with and w/o padding.
    assert torch.allclose(outputs, no_pad_outputs)
