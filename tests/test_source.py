import copy
import os
import tempfile

import pytest
import torch

from src.influence_function import InfluenceFunctionComputer
from src.source import SourceComputer
from tests.utils import check_model_equivalence, prepare_test

RTOL = 1e-3
ATOL = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_checkpoint(model: torch.nn.Module, path: str) -> str:
    torch.save(model.state_dict(), path)
    return path


@pytest.mark.parametrize("test_name", ["mlp", "conv"])
def test_source_shape_and_finiteness(test_name: str, tmp_path) -> None:
    # Basic sanity: build curvature, compute scores, verify shape and that no
    # NaNs/Infs leak through the matrix-function math.
    train_size, valid_size = 32, 8
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=0,
    )
    original_model = copy.deepcopy(model)

    ckpt = _save_checkpoint(model, str(tmp_path / "ckpt.pt"))
    source = SourceComputer(
        model=model,
        task=task,
        checkpoints_per_segment=[[ckpt]],
        iters_per_segment=[100],
        lrs_per_segment=[0.01],
    )
    source.build_curvature_blocks(loader=train_loader)
    scores = source.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )
    assert scores.shape == (valid_size, train_size)
    assert torch.isfinite(scores).all()
    # SOURCE must not mutate the original final-parameters model.
    assert check_model_equivalence(original_model, model)


@pytest.mark.parametrize("test_name", ["mlp", "conv"])
def test_source_single_segment_matches_damped_if(test_name: str, tmp_path) -> None:
    # For L=1, the SOURCE estimator equals IF with damping lambda = 1/(eta*K) up
    # to the (1 - exp(-x))/x vs 1/(sigma+lambda) approximation (paper Eq. 48).
    # The two should rank-correlate strongly, which is a stable test even if
    # the absolute scores differ.
    from scipy.stats import spearmanr  # local import to keep optional dep optional

    train_size, valid_size = 32, 8
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=0,
    )

    ckpt = _save_checkpoint(model, str(tmp_path / "ckpt.pt"))
    eta, K = 0.01, 1000
    lam = 1.0 / (eta * K)

    source = SourceComputer(
        model=copy.deepcopy(model),
        task=task,
        checkpoints_per_segment=[[ckpt]],
        iters_per_segment=[K],
        lrs_per_segment=[eta],
    )
    source.build_curvature_blocks(loader=train_loader)
    source_scores = source.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )

    ifc = InfluenceFunctionComputer(
        model=copy.deepcopy(model),
        task=task,
        damping=lam,
    )
    ifc.build_curvature_blocks(loader=train_loader)
    if_scores = ifc.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )

    # Compare rankings, not absolute magnitudes. SOURCE uses the F_r matrix
    # function while IF uses (sigma + lambda)^-1; both are *qualitatively* the
    # same (paper Eq. 48 -- they agree at sigma -> 0 and sigma -> infty) but
    # differ in between, so we expect strong but not perfect rank correlation.
    # We average across queries to suppress single-query outliers.
    rhos = []
    for q in range(valid_size):
        rho, _ = spearmanr(source_scores[q].cpu().numpy(), if_scores[q].cpu().numpy())
        rhos.append(rho)
    mean_rho = float(sum(rhos) / len(rhos))
    assert (
        mean_rho > 0.85
    ), f"Mean Spearman correlation too low: {mean_rho} (per-query: {rhos})"


@pytest.mark.parametrize("test_name", ["mlp"])
def test_source_duplicate_checkpoints_equal_single(test_name: str, tmp_path) -> None:
    # Averaging across N identical checkpoints should give the same result as a
    # single checkpoint: catches indexing bugs where we'd accidentally double-count
    # gradients or apply S/F_r more than once.
    train_size, valid_size = 32, 8
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=0,
    )

    ckpt = _save_checkpoint(model, str(tmp_path / "ckpt.pt"))
    dup_ckpt = _save_checkpoint(model, str(tmp_path / "dup_ckpt.pt"))

    # `use_true_fisher=False` to keep covariance / Lambda fitting deterministic
    # (the default `True` samples targets from the model output, which is random
    # across calls and would defeat the bit-identity check below).
    source_one = SourceComputer(
        model=copy.deepcopy(model),
        task=task,
        checkpoints_per_segment=[[ckpt]],
        iters_per_segment=[100],
        lrs_per_segment=[0.01],
        use_true_fisher=False,
    )
    source_one.build_curvature_blocks(loader=train_loader)
    scores_one = source_one.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )

    source_two = SourceComputer(
        model=copy.deepcopy(model),
        task=task,
        checkpoints_per_segment=[[ckpt, dup_ckpt]],
        iters_per_segment=[100],
        lrs_per_segment=[0.01],
        use_true_fisher=False,
    )
    source_two.build_curvature_blocks(loader=train_loader)
    scores_two = source_two.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )

    assert torch.allclose(scores_one, scores_two, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("test_name", ["mlp"])
def test_source_multi_segment_runs(test_name: str, tmp_path) -> None:
    # Smoke test that the multi-segment path runs and produces a valid score table.
    train_size, valid_size = 24, 6
    model, train_loader, valid_loader, task = prepare_test(
        device=DEVICE,
        test_name=test_name,
        train_size=train_size,
        valid_size=valid_size,
        seed=0,
    )

    ckpt_early = _save_checkpoint(model, str(tmp_path / "early.pt"))
    ckpt_mid = _save_checkpoint(model, str(tmp_path / "mid.pt"))
    ckpt_late = _save_checkpoint(model, str(tmp_path / "late.pt"))

    source = SourceComputer(
        model=copy.deepcopy(model),
        task=task,
        checkpoints_per_segment=[[ckpt_early], [ckpt_mid], [ckpt_late]],
        iters_per_segment=[50, 50, 50],
        lrs_per_segment=[0.01, 0.005, 0.001],
    )
    source.build_curvature_blocks(loader=train_loader)
    scores = source.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=train_loader
    )
    assert scores.shape == (valid_size, train_size)
    assert torch.isfinite(scores).all()
