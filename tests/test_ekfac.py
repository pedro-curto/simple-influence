import copy

import pytest
import torch

from src.gradient_similarity import GradientSimilarityComputer
from src.influence_function import InfluenceFunctionComputer
from tests.utils import check_model_equivalence, prepare_test

RTOL = 1e-1
ATOL = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("test_name", ["mlp", "conv", "conv_bn"])
def test_model_unchanged(test_name: str) -> None:
    train_size, valid_size = 16, 4
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        train_batch_size=1,
        valid_batch_size=1,
        seed=0,
    )
    original_model = copy.deepcopy(model)

    computer = InfluenceFunctionComputer(
        model=model,
        task=task,
        use_true_fisher=False,
    )
    computer.build_curvature_blocks(train_loader)
    scores1 = computer.compute_total_influence(valid_loader, train_loader)
    assert len(scores1.shape) == 2
    assert scores1.shape[0] == valid_size
    assert scores1.shape[1] == train_size
    assert check_model_equivalence(original_model, model)

    computer = InfluenceFunctionComputer(
        model=model,
        task=task,
        use_true_fisher=False,
    )
    computer.build_curvature_blocks(train_loader)
    scores2 = computer.compute_total_influence(valid_loader, train_loader)
    assert check_model_equivalence(original_model, model)
    assert torch.allclose(scores1, scores2, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("test_name", ["mlp", "conv", "conv_bn"])
def test_batch_additivity(test_name: str) -> None:
    train_size, valid_size = 32, 16
    model, train_loader_bs1, valid_loader_bs1, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        train_batch_size=1,
        valid_batch_size=1,
        seed=0,
    )
    _, train_loader_bs8, valid_loader_bs4, _ = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        train_batch_size=8,
        valid_batch_size=4,
        seed=0,
    )
    original_model = copy.deepcopy(model)

    computer_bs1 = InfluenceFunctionComputer(
        model=model, task=task, use_true_fisher=False,
    )
    computer_bs1.build_curvature_blocks(train_loader_bs1, keep_cache=True)
    assert check_model_equivalence(original_model, model)

    computer_bs8 = InfluenceFunctionComputer(
        model=model, task=task, use_true_fisher=False,
    )
    computer_bs8.build_curvature_blocks(train_loader_bs8, keep_cache=True)
    assert check_model_equivalence(original_model, model)

    batch1_act_cov_list = list(computer_bs1.activation_cov.values())
    batch8_act_cov_list = list(computer_bs8.activation_cov.values())
    for x, y in zip(batch1_act_cov_list, batch8_act_cov_list):
        assert torch.allclose(x, y, rtol=RTOL, atol=ATOL)

    batch1_grad_cov_list = list(computer_bs1.pseudograd_cov.values())
    batch8_grad_cov_list = list(computer_bs8.pseudograd_cov.values())
    for x, y in zip(batch1_grad_cov_list, batch8_grad_cov_list):
        assert torch.allclose(x, y, rtol=RTOL, atol=ATOL)

    batch1_lamb_list = list(computer_bs1.kronecker_eigvals.values())
    batch8_lamb_list = list(computer_bs8.kronecker_eigvals.values())
    for x, y in zip(batch1_lamb_list, batch8_lamb_list):
        assert torch.allclose(x, y, rtol=RTOL, atol=ATOL)

    batch1_full_list = list(computer_bs1.full_factors.values())
    batch8_full_list = list(computer_bs8.full_factors.values())
    for x, y in zip(batch1_full_list, batch8_full_list):
        assert torch.allclose(x, y, rtol=RTOL, atol=ATOL)

    batch1_diag_list = list(computer_bs1.diag_factors.values())
    batch8_diag_list = list(computer_bs8.diag_factors.values())
    for x, y in zip(batch1_diag_list, batch8_diag_list):
        assert torch.allclose(x, y, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("test_name", ["mlp", "conv", "conv_bn"])
def test_disable_precondition(test_name: str) -> None:
    train_size, valid_size = 16, 4
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        train_batch_size=1,
        valid_batch_size=1,
        seed=0,
    )

    computer = InfluenceFunctionComputer(
        model=model, task=task, use_true_fisher=False,
    )
    computer.build_curvature_blocks(train_loader)
    scores = computer.compute_total_influence(
        valid_loader, train_loader, disable_precondition=True
    )

    grad_computer = GradientSimilarityComputer(
        model, task=task, metric="dot",
    )
    grad_scores = grad_computer.compute_total_influence(valid_loader, train_loader)
    assert torch.allclose(scores, grad_scores, rtol=RTOL, atol=ATOL)


@pytest.mark.smoke
@pytest.mark.parametrize("test_name", ["transformer"])
def test_transformer_masks(test_name: str) -> None:
    train_size, valid_size = 4, 1
    model, no_pad_train_loader, no_pad_valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        train_batch_size=1,
        valid_batch_size=1,
        do_not_pad=True,
        seed=0,
    )
    _, train_loader, valid_loader, _ = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        train_batch_size=2,
        valid_batch_size=1,
        do_not_pad=False,
        seed=0,
    )

    no_pad_computer = InfluenceFunctionComputer(
        model=model,
        task=task,
        use_true_fisher=False,
    )
    no_pad_computer.build_curvature_blocks(no_pad_train_loader, keep_cache=True)

    computer = InfluenceFunctionComputer(
        model=model,
        task=task,
        use_true_fisher=False,
    )
    computer.build_curvature_blocks(train_loader, keep_cache=True)

    no_pad_grad_cov_list = list(no_pad_computer.pseudograd_cov.values())
    act_grad_list = list(computer.pseudograd_cov.values())
    for x, y in zip(no_pad_grad_cov_list, act_grad_list):
        assert torch.allclose(x, y, rtol=RTOL, atol=ATOL)

    no_pad_act_cov_list = list(no_pad_computer.activation_cov.values())
    act_cov_list = list(computer.activation_cov.values())
    for x, y in zip(no_pad_act_cov_list, act_cov_list):
        assert torch.allclose(x, y, rtol=RTOL, atol=ATOL)
