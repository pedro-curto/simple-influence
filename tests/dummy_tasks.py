from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from examples.mnist.task import ClassificationTask

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]


class MLPTask(ClassificationTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def influence_modules(self) -> Optional[List[str]]:
        # Ignore computation of BN parameters.
        # dummy_model = make_dummy_conv_bn_module()
        # for name, module in dummy_model.named_modules():
        #     pass
        # return [""]
        return ["0", "2", "4"]

    def representation_modules(self):
        return "2"


class ConvTask(ClassificationTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def influence_modules(self) -> Optional[List[str]]:
        # Ignore computation of BN parameters.
        # dummy_model = make_dummy_conv_bn_module()
        # for name, module in dummy_model.named_modules():
        #     pass
        # return [""]
        return None

    def representation_modules(self) -> str:
        # Ignore computation of BN parameters.
        # dummy_model = make_dummy_conv_bn_module()
        # for name, module in dummy_model.named_modules():
        #     pass
        # return [""]
        return "2"


class ConvBNTask(ClassificationTask):
    def __init__(
        self, device: torch.device = "cpu", generator: Optional[torch.Generator] = None
    ) -> None:
        super().__init__(device=device, generator=generator)

    def influence_modules(self) -> Optional[List[str]]:
        # Ignore computation of BN parameters.
        # dummy_model = make_dummy_conv_bn_module()
        # for name, module in dummy_model.named_modules():
        #     pass
        # return [""]
        return None

    def representation_modules(self) -> str:
        # Ignore computation of BN parameters.
        # dummy_model = make_dummy_conv_bn_module()
        # for name, module in dummy_model.named_modules():
        #     pass
        # return [""]
        return "5"
