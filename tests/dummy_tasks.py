from typing import List, Optional, Tuple

import torch

from examples.mnist.task import ClassificationTask
from examples.regression.task import RegressionTask

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]


class MLPTask(RegressionTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def influence_modules(self) -> Optional[List[str]]:
        return ["0", "2", "4"]

    def representation_module(self) -> str:
        return "2"


class ConvTask(ClassificationTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def influence_modules(self) -> Optional[List[str]]:
        return ["0", "2", "5"]

    def representation_module(self) -> str:
        return "2"


class ConvBNTask(ClassificationTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def influence_modules(self) -> Optional[List[str]]:
        # Ignore score computations on BN layers (note that this is just for testing; we can include
        # BN layers if needed).
        return ["0", "3", "7"]

    def representation_module(self) -> str:
        return "5"
