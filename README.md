# Influence Functions Prototype

Note that this repository is currently in development for a public release (which plan to happen in early 2024) and contains code related to active research projects.
Please ask if you would like to share the code.

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
    pip install -e . 
   ```
   Also, note that the strict minimum version requirement for PyTorch is `2.0.0` (to support functorch).
3. To verify that everything is functioning correctly, you can run one of the tests:
    ```bash
    python test_utils.py
    python test_baselines.py
    python test_ekfac.py
    ```

### Running Regression Experiments

Please note that the regression experiments exclusively utilize CPUs (without GPU) due its small size. To initiate the training process, execute the command below:
```bash
cd examples/regression
python train.py
```
Upon completion of the training, you can evaluate all the baseline TDA methods by running:
```bash
python compute_influences.py
```
For analysis of the influence distribution, I have provided a short script. You can execute it using:
```bash
cd evaluate
python visualize_distribution.py
```

### Running MNIST Experiments

To initiate model training, execute the command below:
```bash
cd examples/mnist
python train.py
```
Upon completing the training, you can run influence functions with:
```bash
python compute_influences.py
```
For visualization of the most influential training images, use:
```bash
python evaluate/visualize_influences.py
```

### Running GLUE Experiments

To initiate model training, follow the command below. Please be aware that this code has been tested on an A100 GPU with 80GB memory. For smaller GPUs, consider reducing the batch size:
```bash
cd examples/glue
python train.py
```
Upon completing the training, you can run influence function computations with:
```bash
python compute_influences.py
```
To display the most influential training sequences, use:
```bash
python evaluate/inspect_influences.py
```

### Running GPT-2 Experiments

To initiate model training, follow the command below. Please be aware that this code has been tested on an A100 GPU with 80GB memory. For smaller GPUs, consider reducing the batch size:
```bash
cd examples/wiki
python train.py
```
Upon completing the training, you can run influence function computations with:
```bash
python compute_influences.py
```
To display the most influential training sequences, use:
```bash
python evaluate/inspect_influences.py
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
   It's important to adapt your module to leverage the above-supported modules. An example of how to adapt existing modules to support EK-FAC is provided in replace_conv1d_modules within `examples/wiki/pipeline.py`, where Huggingface's `Conv1D` module is substituted with the `nn.Linear` module.
2. All TDA baseline techniques, including influence functions, are restricted to single GPU usage.
3. To estimate the actual Fisher for EK-FAC influence computation, a tailored loss function is needed. This function should sample the targets utilizing the outputs. Several examples of this are available in `examples/`, but feel free to reach out for help with specific use cases.
4. Some features like query batching, and layerwise & tokenwise visualization are currently not supported. However, I'm working on integrating these functionalities soon.

## Version Logs
1. 2023/10/14: Initial implementation, supporting four examples `regression`, `mnist`, `glue`, and `wiki`. 
