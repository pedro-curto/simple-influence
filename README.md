# Influence Functions Prototype

Note that this repository is currently in development for a public release and contains code related to active research projects.
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
python compute_baselines.py
```
For analysis of the influence distribution, I've provided a short script. You can execute it using:
```bash
cd evaluate
python visualize_distribution.py
```

### Running MNIST Experiments

To train the model, you can run the following command:
```bash
cd examples/mnist
python train.py
```
When the training run finishes, you can run all baseline modules using the command:
```bash
python compute_baselines.py
```
You can visualize the top influential training images by running:
```bash
python evaluate/visualize_influence.py
```

### Running GLUE Experiments

To train the model, you can run the following command:
```bash
cd examples/glue
python train.py
```
When the training run finishes, you can run all baseline modules using the command:
```bash
python compute_baselines.py
```
You can print the top influential training sequences by running:
```bash
python evaluate/visualize_influence.py
```

### Running GPT-2 Experiments

## Known Limitations
1. EK-FAC influence computations only support `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm2d`, and `Embedding` modules. For example, if the linear module is defined mantually, e.g.,
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
   the EK-FAC factors will not be computed on them. Please change your module so that you use the above configurations.
2. To approximate the true Fisher for EK-FAC influence computation, you require designing a custom loss function (where you need to sample the targets using the outputs). Some examples are provided in `examples/`.
3. Several functionalities such as query batching or layerwise & tokenwise visualization is not yet supported.
## Version Logs
1. 2023/10/14: Initial implementation, supporting four examples `regression`, `mnist`, `glue`, and `wiki`.
