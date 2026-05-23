# Simple Influence

A lightweight library for trying out **training data attribution (TDA)** methods on PyTorch
models. Implements:

- **Influence functions** with EK-FAC curvature approximation (Grosse et al. 2023)
- **SOURCE** — segmented unrolled differentiation that handles non-converged and multi-stage training (Bae et al. 2024, [arXiv:2405.12186](https://arxiv.org/abs/2405.12186))
- Baselines: **gradient similarity**, **representation similarity**, **TracIn / GAS**, and a thin wrapper around **TRAK**

Designed to be easy to adapt to your own model and task — typically you only need to subclass
`AbstractTask` (see `examples/` for four worked end-to-end pipelines covering regression, image
classification, GLUE, and GPT-2 language modeling).

## Getting Started

1. Create a new Conda environment:
    ```bash
    conda create -n simple_influence python=3.10
    conda activate simple_influence
    ```
2. Install the package and its dependencies (use the `pytorch_cpu` extra if you don't have a GPU):
    ```bash
    pip install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/torch_stable.html'
    ```
   PyTorch `>=2.0.0` is required (the EK-FAC implementation uses `torch.func` / `functorch`).
3. Run the test suite to verify the install:
    ```bash
    pytest tests/
    ```
   The non-smoke tests run on CPU against synthetic data. To also run the slower GLUE/Wiki
   smoke tests, use `pytest -m smoke tests/`.

## Quickstart

```python
from src.influence_function import InfluenceFunctionComputer
from examples.mnist.task import ClassificationTask

# `model` is any nn.Module; `train_loader` / `valid_loader` are standard PyTorch DataLoaders.
task = ClassificationTask(device="cuda")
computer = InfluenceFunctionComputer(model=model, task=task)
computer.build_curvature_blocks(train_loader)                # one pass to build EK-FAC.
scores = computer.compute_scores_with_loader(valid_loader, train_loader)
# `scores[i, j]` is the influence of training point `j` on validation point `i`.
```

To swap in a different attribution method, replace `InfluenceFunctionComputer` with
`GradientSimilarityComputer`, `RepresentationSimilarityComputer`, `TracinComputer`, or
`TrakComputer` — each implements the same `compute_scores_with_loader(test_loader, train_loader)`
interface.

### SOURCE

For non-converged or multi-stage training, use `SourceComputer`. It needs a list of checkpoints
per training segment plus the (averaged) learning rate and total number of gradient updates per
segment:

```python
from src.source import SourceComputer

# Suppose you saved 6 checkpoints during training and want L = 3 segments
# (early / middle / late), with 2 checkpoints per segment.
source = SourceComputer(
    model=model,                             # holds the final parameters theta_s
    task=task,
    checkpoints_per_segment=[
        ["ckpts/epoch_2.pt", "ckpts/epoch_4.pt"],   # earliest segment
        ["ckpts/epoch_6.pt", "ckpts/epoch_8.pt"],   # middle segment
        ["ckpts/epoch_10.pt", "ckpts/epoch_12.pt"], # latest segment
    ],
    iters_per_segment=[K_1, K_2, K_3],       # gradient updates per segment
    lrs_per_segment=[eta_1, eta_2, eta_3],   # averaged learning rate per segment
)
source.build_curvature_blocks(train_loader)
scores = source.compute_scores_with_loader(valid_loader, train_loader)
```

With `L = 1` (a single segment), SOURCE reduces to an EK-FAC influence function with damping
`lambda = 1 / (eta * K)` — i.e. the damping is *derived* from the training schedule rather than
hand-tuned. Currently only `Linear` and `Conv2d` modules are supported by `SourceComputer`.

## Running the examples

Example scripts use absolute imports (`from examples.mnist.pipeline import ...`) and also
read/write paths relative to the example directory (`files/checkpoints/`, `files/results/`).
The easiest way to run them is to `cd` into the example directory and point `PYTHONPATH` at
the project root so the imports resolve:

```bash
cd examples/mnist
PYTHONPATH=../.. python -m examples.mnist.train
```

### Regression

CPU-only (small enough that GPU has no benefit):

```bash
cd examples/regression
PYTHONPATH=../.. python -m examples.regression.train               # writes files/checkpoints/
PYTHONPATH=../.. python -m examples.regression.compute_influences  # writes files/results/
PYTHONPATH=../.. python -m examples.regression.evaluate.visualize_distribution
```

### MNIST

```bash
cd examples/mnist
PYTHONPATH=../.. python -m examples.mnist.train
PYTHONPATH=../.. python -m examples.mnist.compute_influences        # IF + SOURCE
PYTHONPATH=../.. python -m examples.mnist.evaluate.visualize_influences
```

A small end-to-end smoke test (~1 minute on CPU) that trains briefly and runs both IF and
SOURCE on the result:

```bash
cd examples/mnist
PYTHONPATH=../.. python smoke_test_source.py
```

### GLUE (BERT)

Tested on an A100 80GB; reduce batch size for smaller GPUs.

```bash
cd examples/glue
PYTHONPATH=../.. python -m examples.glue.train
PYTHONPATH=../.. python -m examples.glue.compute_influences
PYTHONPATH=../.. python -m examples.glue.evaluate.inspect_influences
```

### WikiText-2 (GPT-2)

Tested on an A100 80GB; reduce batch size for smaller GPUs.

```bash
cd examples/wiki
PYTHONPATH=../.. python -m examples.wiki.train
PYTHONPATH=../.. python -m examples.wiki.compute_influences
PYTHONPATH=../.. python -m examples.wiki.evaluate.inspect_influences
```

## Getting Started with Development

1. Install the optional development dependencies:
    ```bash
    pip install -e '.[dev]'
    ```
2. Install the pre-commit hooks:
    ```bash
    pre-commit install
    ```
3. Run the tests to verify that everything is functioning correctly:
    ```bash
    pytest
    ```

## Looking for production-grade influence functions?

This library is intentionally minimal — it's a clean reference implementation aimed at making
the methods easy to read and adapt. For a production-grade implementation with query batching,
layer-wise and token-wise score breakdowns, multi-GPU support, and many more features, see
[**Kronfluence**](https://github.com/pomonam/kronfluence).

## Known Limitations

1. EK-FAC influence calculations are only compatible with the following modules: `Linear`,
   `Conv2d`, `LayerNorm`, `BatchNorm2d`, and `Embedding`. Custom modules that hold parameters
   directly (rather than wrapping one of the supported layers) won't have EK-FAC statistics
   collected for them:
   ```python
   import torch.nn as nn
   import torch

   class CustomLinear(nn.Module):
       def __init__(self, num_inputs, num_outputs):
           super().__init__()
           self.weight = nn.Parameter(torch.Tensor((num_inputs, num_outputs)))

       def forward(self, inputs):
            return inputs @ self.weight
   ```
   Replace these with one of the supported modules — see `replace_conv1d_modules` in
   `examples/wiki/pipeline.py` for a worked example (HuggingFace's `Conv1D` -> `nn.Linear`).
2. All TDA computers in this library are single-GPU.
3. To estimate the true Fisher for EK-FAC, the task's loss function must sample targets from
   the model output (see `examples/` for several worked examples).
4. `SourceComputer` currently supports only `Linear` and `Conv2d` modules — the LayerNorm /
   BatchNorm "full" and Embedding "diagonal" Fisher branches don't yet have a closed-form
   matrix function for the SOURCE preconditioner.

## Version Logs

1. 2023/10/14: Initial implementation, supporting four examples: `regression`, `mnist`, `glue`, `wiki`.
2. 2024-06: Added `SourceComputer` (Bae et al. 2024); standardized docstrings; bug fixes.
