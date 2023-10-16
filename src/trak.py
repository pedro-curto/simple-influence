import copy
from typing import Any, List, Optional

import torch
import torch.nn as nn

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask
from trak import TRAKer


class TrakComputer(AbstractComputer):
    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        proj_dim: int = 4096,
        save_dir: str = "temp/trak_results",
    ) -> None:
        """Initializes the `TrakComputer` class.

        Trak computers use the implementation provided by the authors. For details, please see:
        https://github.com/MadryLab/trak.

        Args:
            model (nn.Module):
                The PyTorch model for which gradient similarities are computed.
            task (AbstractTask):
                The task for the pipeline.
            proj_dim (int, optional):
                Random projection dimension.
            save_dir (str, optional):
                The directory to save intermediate caches.
        """
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        # Save original parameters to CPU.
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        for name, tensor in self.original_state_dict.items():
            self.original_state_dict[name] = tensor.cpu()

        self.proj_dim = proj_dim
        self.save_dir = save_dir

    def _reload_original_params(self) -> None:
        """Reload the initial parameters and buffers, given at the initialization stage."""
        self.model.load_state_dict(self.original_state_dict)
        self.model = self.model.to(self.task.device)

    def compute_scores_with_batch(
        self,
        batch1: Any,
        batch2: Any,
        expt_name: str,
        checkpoints: List[str],
    ) -> torch.Tensor:
        """Compute pairwise influence scores between data points in `batch1` and `batch2`.

        Args:
            batch1 (object):
                The first set of data points from the data loader.
            batch2 (object):
                The second set of data points from the data loader.
            expt_name (str):
                The name of the experiment.
            checkpoints (list):
                A list of paths to the checkpoints.
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
        """Compute pairwise similarity scores between `test_loader` and `train_loader`.

        Args:
            test_loader (DataLoader):
                The loader with test dataset.
            train_loader (DataLoader):
                The loader with training dataset.
            expt_name (str):
                The name of the experiment.
            checkpoints (list):
                A list of paths to the checkpoints.
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
        """Compute self-similarity scores of all data points in `loader`.

        Args:
            loader (DataLoader):
                The loader for which self-similarity scores are computed.
            checkpoints (list):
                A list of paths to the checkpoints.
            expt_name (str):
                The name of the experiment.
            checkpoints (list):
                A list of paths to the checkpoints.
        """
        self.model.eval()

        # There isn't an easy way to compute the self-influence scores.
        # Hence, compute all pairwise scores, and return the diagonal elements.
        expt_name = expt_name + "_self_scores"
        scores = self.compute_scores_with_loader(
            test_loader=loader,
            train_loader=loader,
            expt_name=expt_name,
            checkpoints=checkpoints,
        )
        return torch.diag(scores)
