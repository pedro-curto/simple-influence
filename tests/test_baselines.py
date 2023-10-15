import copy

import pytest
import torch

from src.gradient_similarity import GradientSimilarityComputer
from src.influence_function import InfluenceFunctionComputer
from src.representation_similarity import RepresentationSimilarityComputer
from src.tracin import TracinComputer
from tests.utils import check_model_equivalence, prepare_test

RTOL = 1e-3
ATOL = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("test_name", ["mlp", "conv", "conv_bn"])
def test_representations_similarity(test_name: str) -> None:
    train_size, valid_size = 32, 16
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=0,
    )
    original_model = copy.deepcopy(model)

    computer = RepresentationSimilarityComputer(
        model=model,
        task=task,
        metric="dot",
    )
    scores = computer.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )
    assert len(scores.shape) == 2
    assert scores.shape[0] == valid_size
    assert scores.shape[1] == train_size
    assert check_model_equivalence(original_model, model)

    metric_list = ["l2", "cos", "dot"]
    for metric in metric_list:
        computer = RepresentationSimilarityComputer(
            model=model,
            task=task,
            metric=metric,
        )
        scores = computer.compute_scores_with_loader(
            test_loader=train_loader, train_loader=train_loader
        )
        assert scores.shape[0] == train_size
        assert scores.shape[1] == train_size
        assert check_model_equivalence(original_model, model)

        if metric == "l2":
            assert torch.allclose(
                torch.diag(scores),
                torch.zeros((train_size,), device=DEVICE),
                rtol=RTOL,
                atol=ATOL,
            )
            assert torch.sum(scores > 0) == 0

        if metric == "cos":
            assert torch.allclose(
                torch.diag(scores),
                torch.ones((train_size,), device=DEVICE),
                rtol=RTOL,
                atol=ATOL,
            )

        if metric == "dot":
            assert not torch.allclose(
                torch.diag(scores),
                torch.ones((train_size,), device=DEVICE),
                rtol=RTOL,
                atol=ATOL,
            )


@pytest.mark.parametrize("test_name", ["mlp", "conv", "conv_bn"])
def test_gradients_similarity(test_name: str) -> None:
    train_size, valid_size = 32, 16
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=0,
    )
    original_model = copy.deepcopy(model)

    computer = GradientSimilarityComputer(
        model=model,
        task=task,
        metric="dot",
    )
    scores = computer.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )
    assert len(scores.shape) == 2
    assert scores.shape[0] == valid_size
    assert scores.shape[1] == train_size
    assert check_model_equivalence(original_model, model)

    metric_list = ["cos", "dot"]
    for metric in metric_list:
        computer = GradientSimilarityComputer(model=model, task=task, metric=metric)
        scores = computer.compute_scores_with_loader(
            test_loader=train_loader, train_loader=train_loader
        )

        if metric == "cos":
            assert torch.allclose(
                torch.diag(scores),
                torch.ones((train_size,), device=DEVICE),
                rtol=RTOL,
                atol=ATOL,
            )

        if metric == "dot":
            self_scores = computer.compute_self_score_with_loader(loader=train_loader)
            assert torch.allclose(self_scores, torch.diag(scores), rtol=RTOL, atol=ATOL)

            assert not torch.allclose(
                torch.diag(scores),
                torch.ones((train_size,), device=DEVICE),
                rtol=RTOL,
                atol=ATOL,
            )


@pytest.mark.parametrize("test_name", ["mlp", "conv", "conv_bn"])
def test_tracin(test_name: str) -> None:
    train_size, valid_size = 32, 16
    model1, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=0,
    )
    model2, _, _, _ = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=1,
    )
    original_model1 = copy.deepcopy(model1)

    torch.save(model1.state_dict(), "/tmp/model1.pth")
    torch.save(model1.state_dict(), "/tmp/dup_model1.pth")
    torch.save(model2.state_dict(), "/tmp/model2.pth")

    dup_checkpoints = ["/tmp/dup_model1.pth", "/tmp/model1.pth"]
    checkpoints = ["/tmp/model1.pth", "/tmp/model2.pth"]

    metric_list = ["dot", "cos"]
    for metric in metric_list:
        computer = GradientSimilarityComputer(
            model=model1,
            task=task,
            metric=metric,
        )
        grad_scores = computer.compute_scores_with_loader(
            test_loader=valid_loader, train_loader=train_loader
        )

        computer = TracinComputer(
            model=model1,
            task=task,
            metric=metric,
        )
        tracin_scores = computer.compute_scores_with_loader(
            checkpoints=dup_checkpoints,
            test_loader=valid_loader,
            train_loader=train_loader,
            lrs=1.0,
        )
        assert len(tracin_scores.shape) == 2
        assert tracin_scores.shape[0] == valid_size
        assert tracin_scores.shape[1] == train_size
        assert torch.allclose(grad_scores, tracin_scores, rtol=RTOL, atol=ATOL)

        computer = TracinComputer(
            model=model1,
            task=task,
            metric=metric,
        )
        tracin_scores = computer.compute_scores_with_loader(
            checkpoints=checkpoints,
            test_loader=valid_loader,
            train_loader=train_loader,
            lrs=1.0,
        )
        assert not torch.allclose(grad_scores, tracin_scores, rtol=RTOL, atol=ATOL)
        assert check_model_equivalence(original_model1, model1)


@pytest.mark.parametrize("test_name", ["mlp", "conv", "conv_bn"])
def test_influence_function(test_name: str) -> None:
    train_size, valid_size = 32, 16
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=0,
    )
    original_model = copy.deepcopy(model)

    computer = InfluenceFunctionComputer(
        model=model,
        task=task,
    )
    computer.build_curvature_blocks(loader=train_loader)
    scores = computer.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )
    assert len(scores.shape) == 2
    assert scores.shape[0] == valid_size
    assert scores.shape[1] == train_size
    assert check_model_equivalence(original_model, model)

    computer = InfluenceFunctionComputer(
        model=model,
        task=task,
    )
    computer.build_curvature_blocks(loader=train_loader)
    scores = computer.compute_scores_with_loader(
        test_loader=train_loader, train_loader=train_loader
    )
    self_scores = computer.compute_self_scores_with_loader(loader=train_loader)
    assert torch.allclose(self_scores, torch.diag(scores), rtol=RTOL, atol=ATOL)
