import time
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask
from src.ekfac_utils import (
    ActivationHandler,
    GradientHandler,
    InvalidModuleError,
    make_grad_dict_to_matrix,
)


class InfluenceFunctionComputer(AbstractComputer):
    _supported_kronecker_modules = {"Linear", "Conv2d"}
    _supported_full_modules = {"LayerNorm", "BatchNorm2d"}
    _supported_diag_modules = {"Embedding"}
    eig_dtype: torch.dtype = torch.float64

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        damping: Optional[float] = None,
        n_epoch: int = 1,
        use_true_fisher: bool = True,
        cov_dtype: torch.dtype = torch.float32,
        grads_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initializes the `InfluenceFunctionComputer` class.

        This class performs TDA using influence functions. More specifically, instead of using expensive
        iterative computers such as LiSSA, the class uses EK-FAC approximation. The details can be found
        in https://arxiv.org/pdf/2308.03296.pdf.

        Args:
            model (nn.Module):
                The PyTorch model for which representations are computed.
            task (AbstractTask):
                The task for the pipeline.
            damping (float, optional):
                The damping term. Defaults to None, where the module-wise damping is set to be
                0.1 x (mean of the eigenvalues).
            n_epoch (int, optional):
                Number of epochs to compute covariance and lambda statistics. Defaults to 1.
            use_true_fisher (bool, optional):
                If set to False, the class uses empirical Fisher, where the targets are set to the true targets.
                Defalts to True, where the targets are sampled from the outputs.
            cov_dtype (dtype, optional):
                Dtype to store covariance statistics.
            grads_dtype (dtype, optional):
                Dtype to choose what dtype to use for preconditioning.
        """
        super().__init__(model, task)
        self.func_params = dict(self.model.named_parameters())
        self.func_buffers = dict(self.model.named_buffers())

        self.damping = damping
        self.n_epoch = n_epoch
        self.use_true_fisher = use_true_fisher
        self.cov_dtype = cov_dtype
        self.grads_dtype = grads_dtype

        # Define activation and gradients handler.
        self._activation_handler = ActivationHandler()
        self._gradient_handler = GradientHandler()

        # List of attributes to navigate modules.
        self._name_to_module = dict(self.model.named_modules())
        self._module_to_name = {v: k for k, v in self._name_to_module.items()}
        self.modules = []
        self.modules_name = []
        # Modules, where influnece are computed using EK-FAC (e.g., Linear, Conv2d).
        self.kronecker_modules_name = []
        # Modules, where influences are computed using the full Fisher (e.g., LayerNorm2d).
        self.full_modules_name = []
        # Modules, where influence are computied using the diagonal Fisher (e.g., Embedding).
        self.diag_modules_name = []

        # List of attributes to keep track EK-FAC computation.
        self.activation_cov, self.pseudograd_cov = {}, {}
        self.activation_cov_eigvecs, self.pseudograd_cov_eigvecs = {}, {}
        self.activation_cov_eigvals, self.pseudograd_cov_eigvals = {}, {}
        self.kronecker_eigvals, self.full_factors, self.diag_factors = {}, {}, {}
        self.damping_factors = {}
        self._activation_masks = None

        self._handles = []
        self.initialize()

    def initialize(self) -> None:
        for name, module in self.model.named_modules():
            classname = module.__class__.__name__

            if name in self.task.influence_modules():
                self.logger.info(f"Found module {name}.")

                if classname in self._supported_kronecker_modules:
                    # Register all modules.
                    self.modules.append(module)
                    self.modules_name.append(name)
                    self.kronecker_modules_name.append(name)

                    # Register forward hooks.
                    handle = module.register_forward_pre_hook(self._forward_hook)
                    self._handles.append(handle)

                    # Register backward hooks.
                    handle = module.register_full_backward_hook(self._backward_hook)
                    self._handles.append(handle)

                if classname in self._supported_full_modules:
                    # This is only supported for LayerNorm & BatchNorm parameters (with affine set to True).
                    if module.weight is not None:
                        self.modules.append(module)
                        self.modules_name.append(name)
                        self.full_modules_name.append(name)

                if classname in self._supported_diag_modules:
                    # This is supported for embedding parameters.
                    self.modules.append(module)
                    self.modules_name.append(name)
                    self.diag_modules_name.append(name)

        if len(self.modules_name) == 0:
            error_msg = f"Cannot find any modules in {self.task.influence_modules()}."
            self.logger.error(error_msg)
            raise AttributeError(error_msg)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor]) -> None:
        assert len(inputs) == 1

        with torch.no_grad():
            module_name = self._module_to_name[module]
            acts = self._activation_handler.extract_activations(
                inputs[0].data.to(dtype=self.cov_dtype), module, self._activation_masks
            ).to(dtype=self.cov_dtype)
            if module_name not in self.activation_cov:
                last_dim = acts.shape[-1]
                self.activation_cov[module_name] = torch.zeros(
                    (last_dim, last_dim), dtype=self.cov_dtype, device=acts.device
                )
            self.activation_cov[module_name].addmm_(acts.t(), acts)

    def _backward_hook(
        self,
        module: nn.Module,
        grad_inputs: Tuple[torch.Tensor],
        grad_outputs: Tuple[torch.Tensor],
    ) -> None:
        del grad_inputs
        assert len(grad_outputs) == 1

        with torch.no_grad():
            module_name = self._module_to_name[module]
            pseudograds = self._gradient_handler(
                grad_outputs[0].data.to(dtype=self.cov_dtype), module
            ).to(dtype=self.cov_dtype)
            if module_name not in self.pseudograd_cov:
                last_dim = pseudograds.shape[-1]
                self.pseudograd_cov[module_name] = torch.zeros(
                    (last_dim, last_dim),
                    dtype=self.cov_dtype,
                    device=pseudograds.device,
                )
            self.pseudograd_cov[module_name].addmm_(pseudograds.t(), pseudograds)

    def _perform_forward_and_backward_pass(
        self, batch: Any, sample: bool = False, reduction: str = "sum"
    ) -> None:
        """Perform the forward and backward pass with the given `batch`."""
        loss = self.task.get_train_loss(
            model=self.model,
            batch=batch,
            sample=sample,
            reduction=reduction,
        )
        loss.backward()

    def fit_covariances(self, loader: torch.utils.data.DataLoader) -> None:
        t0 = time.time()

        self.model.eval()
        examples_seen = 0
        for _ in range(self.n_epoch):
            for batch in loader:
                self.model.zero_grad()
                self._activation_masks = self.task.get_activation_masks(batch)
                self._perform_forward_and_backward_pass(
                    batch, sample=self.use_true_fisher, reduction="sum"
                )
                examples_seen += self.task.get_batch_size(batch)

        with torch.no_grad():
            for name in self.kronecker_modules_name:
                self.activation_cov[name] /= examples_seen
                self.activation_cov[name] = self.activation_cov[name].to(
                    self.grads_dtype
                )
                self.pseudograd_cov[name] /= examples_seen
                self.pseudograd_cov[name] = self.pseudograd_cov[name].to(
                    self.grads_dtype
                )
        self.logger.info(f"Seen {examples_seen} examples.")
        self.logger.info(f"Time for computing covariances: {time.time() - t0}")

    def fit_eigendecompositions(self, keep_cache: bool = False) -> None:
        t1 = time.time()

        with torch.no_grad():
            for name in self.kronecker_modules_name:
                orig_dtype = self.activation_cov[name].dtype
                eigvals, eigvecs = torch.linalg.eigh(
                    self.activation_cov[name].to(dtype=self.eig_dtype)
                )
                self.activation_cov_eigvals[name] = eigvals.to(dtype=orig_dtype)
                self.activation_cov_eigvecs[name] = eigvecs.to(dtype=orig_dtype)

                if not keep_cache:
                    del self.activation_cov[name]
                    del self.activation_cov_eigvals[name]

                orig_dtype = self.pseudograd_cov[name].dtype
                eigvals, eigvecs = torch.linalg.eigh(
                    self.pseudograd_cov[name].to(dtype=self.eig_dtype)
                )
                self.pseudograd_cov_eigvals[name] = eigvals.to(dtype=orig_dtype)
                self.pseudograd_cov_eigvecs[name] = eigvecs.to(dtype=orig_dtype)

                if not keep_cache:
                    del self.pseudograd_cov[name]
                    del self.pseudograd_cov_eigvals[name]
        self.logger.info(f"Time for eigendecomposition: {time.time() - t1}")

    def compute_kronecker_lambda(
        self, module_name: str, per_batch_grads: torch.Tensor
    ) -> None:
        if module_name not in self.kronecker_eigvals:
            self.kronecker_eigvals[module_name] = torch.zeros(
                (per_batch_grads.shape[1], per_batch_grads.shape[2]),
                dtype=self.grads_dtype,
                device=per_batch_grads.device,
            )
        grads_rot = torch.matmul(
            per_batch_grads, self.activation_cov_eigvecs[module_name]
        )

        batch_size = grads_rot.shape[0]
        for i in range(batch_size):
            weight_grad_rot = torch.matmul(
                self.pseudograd_cov_eigvecs[module_name].t(), grads_rot[i, :, :]
            )
            self.kronecker_eigvals[module_name].add_(torch.square(weight_grad_rot))

    def compute_full_factors(
        self, module_name: str, per_batch_grads: torch.Tensor
    ) -> None:
        if module_name not in self.full_factors:
            last_dim = per_batch_grads.shape[1]
            self.full_factors[module_name] = torch.zeros(
                (last_dim, last_dim),
                dtype=self.grads_dtype,
                device=per_batch_grads.device,
            )
        self.full_factors[module_name].addmm_(per_batch_grads.t(), per_batch_grads)

    def compute_diag_lambda(
        self, module_name: str, per_batch_grads: torch.Tensor
    ) -> None:
        if module_name not in self.diag_factors:
            self.diag_factors[module_name] = torch.zeros(
                (per_batch_grads.shape[1], per_batch_grads.shape[2]),
                dtype=self.grads_dtype,
                device=per_batch_grads.device,
            )
        self.diag_factors[module_name].add_(torch.square(per_batch_grads).sum(dim=0))

    def fit_additional_factors(self, loader: torch.utils.data.DataLoader) -> None:
        t2 = time.time()

        def compute_loss(_params, _buffers, _batch):
            return self.task.get_train_loss(
                model=self.model,
                batch=_batch,
                parameter_and_buffer_dicts=(_params, _buffers),
                sample=self.use_true_fisher,
                reduction="sum",
            )

        ft_compute_grad = torch.func.grad(compute_loss, has_aux=False)

        self.model.eval()
        examples_seen = 0
        for _ in range(self.n_epoch):
            for batch in loader:
                examples_seen += self.task.get_batch_size(batch)
                grads_dict = torch.func.vmap(
                    ft_compute_grad,
                    in_dims=(None, None, 0),
                    randomness="different",
                )(self.func_params, self.func_buffers, batch)

                with torch.no_grad():
                    for name, module in zip(self.modules_name, self.modules):
                        if name in self.kronecker_modules_name:
                            per_batch_grads = make_grad_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_kronecker_lambda(name, per_batch_grads)
                        elif name in self.full_modules_name:
                            per_batch_grads = make_grad_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_full_factors(name, per_batch_grads)
                        elif name in self.diag_modules_name:
                            per_batch_grads = make_grad_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_diag_lambda(name, per_batch_grads)
                        else:
                            raise InvalidModuleError()

        with torch.no_grad():
            for name in self.kronecker_modules_name:
                self.kronecker_eigvals[name] /= examples_seen
                self.kronecker_eigvals[name] = self.kronecker_eigvals[name].to(
                    self.grads_dtype
                )
                if self.damping is None:
                    self.damping_factors[name] = 0.1 * torch.mean(
                        self.kronecker_eigvals[name]
                    )
                else:
                    self.damping_factors[name] = self.damping

            for name in self.full_modules_name:
                self.full_factors[name] /= examples_seen

                eigvals, eigvecs = torch.linalg.eigh(
                    self.full_factors[name].to(dtype=self.eig_dtype)
                )
                if self.damping is None:
                    # Use the heuristic approach where we set damping to be.
                    self.damping_factors[name] = 0.01 * torch.mean(eigvals).to(
                        dtype=self.grads_dtype
                    )
                else:
                    self.damping_factors[name] = self.damping

                inv_eigvals = eigvals + self.damping_factors[name]
                self.full_factors[name] = torch.matmul(
                    torch.matmul(eigvecs, torch.diag(inv_eigvals)), eigvecs.t()
                )
                self.full_factors[name] = self.full_factors[name].to(self.grads_dtype)

            for name in self.diag_modules_name:
                self.diag_factors[name] /= examples_seen
                self.diag_factors[name] = self.diag_factors[name].to(self.grads_dtype)
                if isinstance(self.damping, str) and self.damping == "heuristic":
                    self.damping_factors[name] = 0.1 * torch.mean(
                        self.diag_factors[name]
                    )
                else:
                    self.damping_factors[name] = self.damping
        print("Time for computing Lambda:", time.time() - t2)

    def build_curvature_blocks(
        self, loader: torch.utils.data.DataLoader, keep_cache: bool = False
    ) -> None:
        self.fit_covariances(loader=loader)
        self.fit_eigendecompositions(keep_cache=keep_cache)
        for handle in self._handles:
            handle.remove()
        self.fit_additional_factors(loader=loader)

    def precondition_grads(
        self,
        module_name: str,
        grads: torch.Tensor,
    ) -> torch.Tensor:
        if module_name in self.kronecker_modules_name:
            grads_rot = torch.matmul(
                self.pseudograd_cov_eigvecs[module_name].t(),
                torch.matmul(
                    grads.to(dtype=self.grads_dtype),
                    self.activation_cov_eigvecs[module_name],
                ),
            )

            precond_grads_rot = grads_rot / (
                self.kronecker_eigvals[module_name] + self.damping_factors[module_name]
            )

            precond_grads = torch.matmul(
                self.pseudograd_cov_eigvecs[module_name],
                torch.matmul(
                    precond_grads_rot,
                    self.activation_cov_eigvecs[module_name].t(),
                ),
            )

        elif module_name in self.full_modules_name:
            precond_grads = torch.matmul(
                grads.to(dtype=self.grads_dtype), self.full_factors[module_name]
            )

        elif module_name in self.diag_modules_name:
            precond_grads = grads / (
                self.diag_factors[module_name] + self.damping_factors[module_name]
            )

        else:
            raise InvalidModuleError()
        return precond_grads

    def compute_influence(
        self,
        valid_batch: Any,
        train_batch: Any,
    ) -> torch.Tensor:
        self.model.eval()

        pass

    def compute_total_influence(
        self,
        valid_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        disable_precondition: bool = False,
    ) -> torch.Tensor:
        self.model.eval()

        with torch.no_grad():
            score_table = torch.zeros(
                (len(valid_loader.dataset), len(train_loader.dataset)),
                dtype=self.grads_dtype,
                device=self.task.device,
                requires_grad=False,
            )

        ft_compute_train_grad = torch.func.grad(self._compute_train_loss, has_aux=False)
        ft_compute_measurement_grad = torch.func.grad(
            self._compute_measurement, has_aux=False
        )
        num_valid_data = len(valid_loader.dataset)

        num_processed_valid = 0
        for valid_batch in valid_loader:
            print(
                f"Processed {num_processed_valid} validation data points (out of {num_valid_data})."
            )
            valid_batch_size = self.task.get_batch_size(valid_batch)

            precond_query_grads_dict = {}
            query_grads_dict = torch.func.vmap(
                ft_compute_measurement_grad,
                in_dims=(None, None, 0),
                randomness="different",
            )(self.func_params, self.func_buffers, valid_batch)

            with torch.no_grad():
                for name, module in zip(self.modules_name, self.modules):
                    query_grads = make_grad_dict_to_matrix(
                        module, name, query_grads_dict
                    )
                    if disable_precondition:
                        precond_query_grads_dict[name] = query_grads.reshape(
                            valid_batch_size, -1
                        ).to(dtype=self.grads_dtype)
                    else:
                        precond_query_grads_dict[name] = (
                            self.precondition_grads(name, query_grads)
                            .reshape(valid_batch_size, -1)
                            .to(dtype=self.grads_dtype)
                        )

            num_processed_train = 0
            for train_batch in train_loader:
                train_batch_size = self.task.get_batch_size(train_batch)
                train_grads_dict = torch.func.vmap(
                    ft_compute_train_grad,
                    in_dims=(None, None, 0),
                    randomness="different",
                )(self.func_params, self.func_buffers, train_batch)

                with torch.no_grad():
                    for name, module in zip(self.modules_name, self.modules):
                        train_grads = (
                            make_grad_dict_to_matrix(module, name, train_grads_dict)
                            .reshape(train_batch_size, -1)
                            .to(dtype=self.grads_dtype)
                        )

                        v_start = num_processed_valid
                        t_start = num_processed_train
                        score_table[
                            v_start : v_start + valid_batch_size,
                            t_start : t_start + train_batch_size,
                        ].addmm_(precond_query_grads_dict[name], train_grads.t())
                num_processed_train += train_batch_size
            num_processed_valid += valid_batch_size
        return score_table.to(dtype=self.score_dtype)
