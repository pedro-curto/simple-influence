# Influence Functions Prototype

Please note that this repository is currently under development for public release. We kindly request that you ask from sharing this implementation.

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
3. To verify that everything is functioning correctly, you can run one of the tests:
    ```bash
    python test_utils.py
    python test_baselines.py
    python test_ekfac.py
    ```

## Running Regression Experiments

## Running MNIST Experiments

## Running GLUE Experiments

## Running GPT-2 Experiments
