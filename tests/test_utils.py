import pytest

from tests.utils import (
    make_dummy_conv_bn_module,
    make_dummy_conv_loader,
    make_dummy_conv_module,
    make_dummy_glue_loader,
    make_dummy_linear_loader,
    make_dummy_linear_module,
    make_glue_module,
)


def test_linear() -> None:
    loader = make_dummy_linear_loader(batch_size=1)
    model = make_dummy_linear_module()

    inputs, targets = next(iter(loader))
    outputs = model(inputs)
    assert outputs is not None


def test_conv() -> None:
    loader = make_dummy_conv_loader(batch_size=1)
    model = make_dummy_conv_module()

    inputs, targets = next(iter(loader))
    outputs = model(inputs)
    assert outputs is not None


def test_conv_bn() -> None:
    loader = make_dummy_conv_loader(batch_size=1)
    model = make_dummy_conv_bn_module()

    inputs, targets = next(iter(loader))
    outputs = model(inputs)
    assert outputs is not None


def test_transformer() -> None:
    loader = make_dummy_glue_loader(batch_size=1)
    model = make_glue_module()

    batch = next(iter(loader))
    outputs = model(
        batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]
    )
    assert outputs is not None


@pytest.mark.smoke
def test_transformer_with_no_padding() -> None:
    loader = make_dummy_glue_loader(batch_size=1, do_not_pad=True)

    data_iter = iter(loader)
    batch1 = next(data_iter)
    batch2 = next(data_iter)

    assert batch1["input_ids"].shape[1] != batch2["input_ids"].shape[1]
    assert batch1["attention_mask"].sum() == batch1["attention_mask"].shape[1]
    assert batch2["attention_mask"].sum() == batch2["attention_mask"].shape[1]
