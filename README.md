# Influence Functions Prototype

A lightweight prototype for trying out **training data attribution (TDA)** methods on PyTorch
models. Implements:

- **Influence functions** with EK-FAC curvature approximation (Grosse et al. 2023)
- **SOURCE** — segmented unrolled differentiation that handles non-converged and multi-stage training (Bae et al. 2024, [arXiv:2405.12186](https://arxiv.org/abs/2405.12186))
- Baselines: **gradient similarity**, **representation similarity**, **TracIn / GAS**, and a thin wrapper around **TRAK**

Designed to be easy to adapt to your own model and task — typically you only need to subclass
`AbstractTask` (see `examples/` for four worked end-to-end pipelines covering regression, image
classification, GLUE, and GPT-2 language modeling).

## Getting Started

To begin, follow these steps to set up your environment:

1. Create a new Conda environment:
    ```bash
    conda create -n influence_functions_env python=3.10
    conda activate influence_functions_env
    ```
2. Install this package and its dependencies within the newly created environment (if you are working on a machine without a GPU, you can choose the `pytorch_cpu` option instead):
    ```bash
    pip install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/torch_stable.html'
    ```
   PyTorch `>=2.0.0` is required (the EK-FAC implementation uses `torch.func` / `functorch`).
3. To verify that everything is functioning correctly, run the test suite:
    ```bash
    pytest tests/
    ```
   The non-smoke tests run on CPU and only depend on synthetic data. To also run the slower
   GLUE/Wiki smoke tests, use `pytest -m smoke tests/`.

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

All example scripts use absolute imports (`from examples.mnist.pipeline import ...`),
so they must be run as Python modules from the **project root** — not by `cd`-ing
into the example directory. Each example writes / reads relative paths under
`examples/<name>/files/`, so first `cd` into the example directory.

### Regression

CPU-only (small enough that GPU has no benefit):

```bash
cd examples/regression
python -m examples.regression.train               # writes checkpoints under files/checkpoints/
python -m examples.regression.compute_influences  # writes scores under files/results/
python -m examples.regression.evaluate.visualize_distribution
```

### MNIST

```bash
cd examples/mnist
python -m examples.mnist.train
python -m examples.mnist.compute_influences        # IF + SOURCE
python -m examples.mnist.evaluate.visualize_influences
```
```

A small end-to-end smoke test (~1 minute on CPU) that trains briefly and runs
both IF and SOURCE on the result:

```bash
cd examples/mnist
PYTHONPATH=$(pwd)/../.. python smoke_test_source.py
```

### GLUE (BERT)

Tested on an A100 80GB; reduce batch size for smaller GPUs.

```bash
cd examples/glue
python -m examples.glue.train
python -m examples.glue.compute_influences
python -m examples.glue.evaluate.inspect_influences
```

### WikiText-2 (GPT-2)

Tested on an A100 80GB; reduce batch size for smaller GPUs.

```bash
cd examples/wiki
python -m examples.wiki.train
python -m examples.wiki.compute_influences
python -m examples.wiki.evaluate.inspect_influences
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

## Known Limitations
1. EK-FAC influence calculations are only compatible with the following modules: `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm2d`, and `Embedding`. If a module is manually defined - like the CustomLinear module shown below - it won't support EK-FAC statistics:
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
   It's important to adapt your module to leverage the above-supported modules. An example of how to adapt existing modules to support EK-FAC is provided in `replace_conv1d_modules` within `examples/wiki/pipeline.py`, where Huggingface's `Conv1D` module is substituted with the `nn.Linear` module.
2. All TDA baseline techniques, including influence functions, are restricted to single GPU usage.
3. To estimate the actual Fisher for EK-FAC influence computation, a tailored loss function is needed. This function should sample the targets utilizing the outputs. Several examples of this are available in `examples/`, but feel free to reach out for help with specific use cases.
4. Some features like query batching, and layerwise & tokenwise visualization are currently not supported. However, I'm working on integrating these functionalities soon.

## Version Logs
1. 2023/10/14: Initial implementation, supporting four examples `regression`, `mnist`, `glue`, and `wiki`. 
