"""Shared helpers used by every example pipeline."""

import gc
import os
import random
import struct

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set the global random seed for ``random``, NumPy, and PyTorch.

    Args:
        seed (int):
            Random seed.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def reset_seed() -> None:
    """Reset the seed to a fresh value drawn from ``os.urandom``."""
    rng_seed = struct.unpack("I", os.urandom(4))[0]
    set_seed(rng_seed)


def clear_gpu_cache() -> None:
    """Free GPU memory: run gc, empty the PyTorch CUDA cache, reset peak stats.

    No-op on CPU-only setups.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
