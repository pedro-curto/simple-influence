"""TRAK wrapper.

Thin adapter that funnels the project's `AbstractTask` and data loaders into
the upstream `TRAKer` implementation from
https://github.com/MadryLab/trak. The `task.get_model_output()` method must
return a `trak.modelout_functions.AbstractModelOutput` subclass.
"""

import copy
from typing import Any, List

import torch
import torch.nn as nn

from simple_influence.abstract_computer import AbstractComputer
from simple_influence.abstract_task import AbstractTask
from trak import TRAKer


class TrakComputer(AbstractComputer):
    """Compute TRAK attribution scores using the upstream `TRAKer`.

    For each checkpoint we run TRAK's two-phase API: ``featurize`` accumulates
    projected per-sample gradients across the training data, then ``score``
    is invoked for the query data. After all checkpoints have been processed
    the scores are pulled out via ``finalize_scores`` and the model's original
    parameters are restored.

    Attributes:
        original_state_dict (dict): the model's parameters at construction
            time (CPU-resident); restored after each score computation.
        proj_dim (int): TRAK random-projection dimension.
        save_dir (str): directory in which TRAK persists intermediate
            features and scores.
    """

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        proj_dim: int = 4096,
        save_dir: str = "temp/trak_results",
    ) -> None:
        """Initialize the TRAK computer.

        Args:
            model (nn.Module):
                Model whose final parameters are the starting point; the
                computer overwrites these with each checkpoint while scoring
                and restores them at the end.
            task (AbstractTask):
                Task adapter. ``task.get_model_output()`` must return a
                non-``None`` TRAK model-output adapter.
            proj_dim (int, optional):
                Random-projection dimension. Defaults to ``4096``.
            save_dir (str, optional):
                Directory for TRAK intermediates. Defaults to
                ``"temp/trak_results"``.
        """
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        # Snapshot the original parameters on CPU so we can restore them
        # after each score computation.
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        for name, tensor in self.original_state_dict.items():
            self.original_state_dict[name] = tensor.cpu()

        self.proj_dim = proj_dim
        self.save_dir = save_dir

    def _reload_original_params(self) -> None:
        """Restore the parameters captured at construction time."""
        self.model.load_state_dict(self.original_state_dict)
        self.model = self.model.to(self.task.device)

    def compute_scores_with_batch(
        self,
        batch1: Any,
        batch2: Any,
        expt_name: str,
        checkpoints: List[str],
    ) -> torch.Tensor:
        """Compute TRAK scores for a pair of batches.

        Args:
            batch1 (Any):
                Training batch (used in TRAK's featurize phase).
            batch2 (Any):
                Query batch (used in TRAK's score phase).
            expt_name (str):
                Name TRAK uses to scope its on-disk caches.
            checkpoints (List[str]):
                Paths to checkpoints to ensemble across.

        Returns:
            torch.Tensor: ``(|batch2|, |batch1|)`` score table (rows = query,
            cols = train), cast to ``self.score_dtype``.
        """
        self.model.eval()

        expt_name = expt_name + "_with_batch"
        traker = TRAKer(
            model=self.model,
            load_from_save_dir=True,
            task=self.task.get_model_output(),
            proj_dim=self.proj_dim,
            train_set_size=self.task.get_batch_size(batch1),
            device=self.task.device,
            save_dir=f"{self.save_dir}/{expt_name}",
            use_half_precision=False,
        )
        for model_id, ckpt in enumerate(checkpoints):
            traker.load_checkpoint(
                torch.load(ckpt, map_location=self.task.device), model_id=model_id
            )
            batch = [x.to(self.task.device) for x in batch1]
            traker.featurize(batch=batch, num_samples=self.task.get_batch_size(batch))
        traker.finalize_features()

        for model_id, ckpt in enumerate(checkpoints):
            traker.start_scoring_checkpoint(
                exp_name=expt_name,
                checkpoint=torch.load(ckpt, map_location=self.task.device),
                model_id=model_id,
                num_targets=self.task.get_batch_size(batch2),
            )
            batch = [x.to(self.task.device) for x in batch2]
            traker.score(batch=batch, num_samples=self.task.get_batch_size(batch2))
        scores = traker.finalize_scores(exp_name=expt_name)

        self._reload_original_params()
        return torch.from_numpy(scores).t().to(self.score_dtype)

    def compute_scores_with_loader(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        expt_name: str,
        checkpoints: List[str],
    ) -> torch.Tensor:
        """Compute TRAK scores for a pair of loaders.

        Args:
            test_loader (DataLoader):
                Loader yielding query (test) data.
            train_loader (DataLoader):
                Loader yielding training data.
            expt_name (str):
                Name TRAK uses to scope its on-disk caches.
            checkpoints (List[str]):
                Paths to checkpoints to ensemble across.

        Returns:
            torch.Tensor: ``(num_test, num_train)`` score table.
        """
        self.model.eval()

        expt_name = expt_name + "_with_loader"
        traker = TRAKer(
            model=self.model,
            load_from_save_dir=True,
            task=self.task.get_model_output(),
            proj_dim=self.proj_dim,
            train_set_size=len(train_loader.dataset),
            device=self.task.device,
            save_dir=f"{self.save_dir}/{expt_name}",
            use_half_precision=False,
        )
        for model_id, ckpt in enumerate(checkpoints):
            traker.load_checkpoint(
                torch.load(ckpt, map_location=self.task.device), model_id=model_id
            )
            for batch in train_loader:
                batch = [x.to(self.task.device) for x in batch]
                traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        traker.finalize_features()

        for model_id, ckpt in enumerate(checkpoints):
            traker.start_scoring_checkpoint(
                exp_name=expt_name,
                checkpoint=torch.load(ckpt, map_location=self.task.device),
                model_id=model_id,
                num_targets=len(test_loader.dataset),
            )
            for batch in test_loader:
                batch = [x.to(self.task.device) for x in batch]
                traker.score(batch=batch, num_samples=batch[0].shape[0])
        scores = traker.finalize_scores(exp_name=expt_name)

        self._reload_original_params()
        return torch.from_numpy(scores).t().to(self.score_dtype)

    def compute_self_scores_with_loader(
        self,
        loader: torch.utils.data.DataLoader,
        expt_name: str,
        checkpoints: List[str],
    ) -> torch.Tensor:
        """Compute TRAK self-similarity scores via the diagonal of the full table.

        TRAK does not expose a direct self-score path, so this is a
        convenience wrapper that runs :meth:`compute_scores_with_loader`
        with ``train == test`` and returns the diagonal.

        Args:
            loader (DataLoader):
                Loader yielding data points.
            expt_name (str):
                Name TRAK uses to scope its on-disk caches.
            checkpoints (List[str]):
                Paths to checkpoints to ensemble across.

        Returns:
            torch.Tensor: 1-D tensor of self-similarity scores.
        """
        self.model.eval()

        # TRAK doesn't have a direct self-influence path; compute the full
        # table and return the diagonal.
        expt_name = expt_name + "_self_scores"
        scores = self.compute_scores_with_loader(
            test_loader=loader,
            train_loader=loader,
            expt_name=expt_name,
            checkpoints=checkpoints,
        )
        return torch.diag(scores)
