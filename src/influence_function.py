"""Influence functions with EK-FAC curvature approximation.

Implementation of the EK-FAC-approximated influence function of Grosse et al.
(2023, https://arxiv.org/pdf/2308.03296.pdf). Given a trained model, this
computer attributes a query measurement `f(z_q, theta)` to each training point
`z_m` by computing

    tau(z_q, z_m) = grad f(z_q, theta)^T H^{-1} grad L(z_m, theta)

where `H` is the (Gauss-Newton) Hessian and the inverse Hessian-vector product
is approximated using EK-FAC. The implementation supports three module
families:

  * Kronecker (``Linear``, ``Conv2d``): full EK-FAC factorization.
  * Full (``LayerNorm``, ``BatchNorm2d``): explicit dense Fisher inversion.
  * Diagonal (``Embedding``): diagonal Fisher.

Only these module types are picked up; other parameters in the model do not
contribute to the attribution score.
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask
from src.ekfac_utils import (
    InvalidModuleError,
    extract_activations,
    extract_gradients,
    make_grads_dict_to_matrix,
)


class InfluenceFunctionComputer(AbstractComputer):
    """Compute influence-function scores via EK-FAC.

    The class is used in three phases:

      1. Construct the computer (registers forward / backward hooks on the
         supported modules).
      2. Call :meth:`build_curvature_blocks` once with a training data loader.
         This populates the EK-FAC factors (covariances, eigenbasis, corrected
         eigenvalues, full / diagonal Fisher) used for preconditioning.
      3. Call :meth:`compute_scores_with_loader` (or
         :meth:`compute_scores_with_batch`, or
         :meth:`compute_self_scores_with_loader`) to obtain attribution scores.

    Attributes:
        damping (Optional[float]): module-wise damping term. If ``None``, the
            module-wise damping is set to ``0.1 * mean(eigvals)`` (a common
            heuristic) per module.
        n_epoch (int): number of epochs over the data loader used while fitting
            covariances and the corrected eigenvalues.
        use_true_fisher (bool): whether to sample targets from the model output
            (true Fisher) instead of using the actual labels (empirical Fisher).
    """

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
    ) -> None:
        """Initialize the influence-function computer.

        Args:
            model (nn.Module):
                Model whose final parameters serve as ``theta`` for attribution.
            task (AbstractTask):
                Task adapter describing the loss, measurement, and the module
                set over which scores are computed.
            damping (float, optional):
                Module-wise damping ``lambda`` added to the EK-FAC eigenvalues
                before inversion. If ``None`` (default), uses
                ``0.1 * mean(eigvals)`` per module.
            n_epoch (int, optional):
                Number of passes over the loader for covariance / corrected-
                eigenvalue estimation. Defaults to ``1``.
            use_true_fisher (bool, optional):
                If ``True`` (default), sample targets from the model output
                while fitting curvature (true Fisher); if ``False``, use the
                labels (empirical Fisher).
        """
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        self.func_params = dict(self.model.named_parameters())
        self.func_buffers = dict(self.model.named_buffers())

        self.damping = damping
        self.n_epoch = n_epoch
        self.use_true_fisher = use_true_fisher

        # Reverse map for hooks: hooks receive the module object, but we index
        # all our caches by the qualified module name.
        self._module_to_name = {
            v: k for k, v in dict(self.model.named_modules()).items()
        }

        # Ordered lists holding the modules participating in attribution and
        # their qualified names. The three `_*_modules_name` sub-lists indicate
        # which Fisher branch each module belongs to.
        self.modules: list = []
        self.modules_name: list = []
        self.kronecker_modules_name: list = []
        self.full_modules_name: list = []
        self.diag_modules_name: list = []

        # Caches populated by `fit_covariances`, `fit_eigendecompositions`,
        # `fit_additional_factors`.
        self.activation_cov: Dict[str, torch.Tensor] = {}
        self.pseudograd_cov: Dict[str, torch.Tensor] = {}
        self.activation_cov_eigvecs: Dict[str, torch.Tensor] = {}
        self.pseudograd_cov_eigvecs: Dict[str, torch.Tensor] = {}
        self.activation_cov_eigvals: Dict[str, torch.Tensor] = {}
        self.pseudograd_cov_eigvals: Dict[str, torch.Tensor] = {}
        self.kronecker_eigvals: Dict[str, torch.Tensor] = {}
        self.full_factors: Dict[str, torch.Tensor] = {}
        self.diag_factors: Dict[str, torch.Tensor] = {}
        self.damping_factors: Dict[str, torch.Tensor] = {}
        self._activation_masks: Optional[torch.Tensor] = None

        # Flags so each fitting step is idempotent.
        self._covariance_done = False
        self._eigendecompositon_done = False
        self._additional_factors_done = False

        self._handles: list = []
        self.initialize()

    def initialize(self) -> None:
        """Register hooks and populate the module lists.

        Walks the model once; for every module named in
        ``task.influence_modules()`` whose class is supported, the module is
        appended to :attr:`modules` and the appropriate Fisher-branch list.
        Linear / Conv2d modules also get forward and backward hooks installed
        for activation- and pseudo-gradient-covariance accumulation.

        Raises:
            AttributeError: if no supported module was found.
        """
        for name, module in self.model.named_modules():
            classname = module.__class__.__name__

            if name in self.task.influence_modules():
                self.logger.info(f"Found module {name}.")

                if classname in self._supported_kronecker_modules:
                    self.modules.append(module)
                    self.modules_name.append(name)
                    self.kronecker_modules_name.append(name)

                    handle = module.register_forward_pre_hook(self._forward_hook)
                    self._handles.append(handle)
                    handle = module.register_full_backward_hook(self._backward_hook)
                    self._handles.append(handle)

                if classname in self._supported_full_modules:
                    # `affine=False` LayerNorm/BatchNorm has no weight / bias
                    # to attribute against; skip silently.
                    if module.weight is not None:
                        self.modules.append(module)
                        self.modules_name.append(name)
                        self.full_modules_name.append(name)

                if classname in self._supported_diag_modules:
                    self.modules.append(module)
                    self.modules_name.append(name)
                    self.diag_modules_name.append(name)

        if len(self.modules_name) == 0:
            error_msg = f"Cannot find any modules in {self.task.influence_modules()}."
            self.logger.error(error_msg)
            raise AttributeError(error_msg)

    # ------------------------------------------------------------------ hooks

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor]) -> None:
        """Forward pre-hook: accumulate the per-module activation covariance.

        Args:
            module (nn.Module):
                Module whose forward pass we are intercepting.
            inputs (tuple):
                Positional inputs to the module (expected to be a 1-tuple).
        """
        assert len(inputs) == 1

        with torch.no_grad():
            module_name = self._module_to_name[module]
            # `.detach().clone()` defensively copies out of the autograd graph:
            # the hook's input shares storage with tensors that may be mutated
            # later in the backward pass, which would silently corrupt the
            # accumulated covariance.
            acts = extract_activations(
                inputs[0].detach().clone().to(dtype=self.stats_dtype),
                module,
                self._activation_masks,
            ).to(dtype=self.stats_dtype)
            if module_name not in self.activation_cov:
                last_dim = acts.shape[-1]
                self.activation_cov[module_name] = torch.zeros(
                    (last_dim, last_dim), dtype=self.stats_dtype, device=acts.device
                )
            self.activation_cov[module_name].addmm_(acts.t(), acts)

    def _backward_hook(
        self,
        module: nn.Module,
        grad_inputs: Tuple[torch.Tensor],
        grad_outputs: Tuple[torch.Tensor],
    ) -> None:
        """Full backward hook: accumulate the per-module pseudo-gradient covariance.

        Args:
            module (nn.Module):
                Module whose backward pass we are intercepting.
            grad_inputs (tuple):
                Unused (PyTorch passes ``grad_input`` here for full backward
                hooks, but we only need ``grad_output``).
            grad_outputs (tuple):
                Gradient of the loss w.r.t. each output (expected to be a
                1-tuple).
        """
        del grad_inputs
        assert len(grad_outputs) == 1

        with torch.no_grad():
            module_name = self._module_to_name[module]
            # Defensive copy: same reason as in `_forward_hook`.
            pseudograds = extract_gradients(
                grad_outputs[0].detach().clone().to(dtype=self.stats_dtype), module
            ).to(dtype=self.stats_dtype)
            if module_name not in self.pseudograd_cov:
                last_dim = pseudograds.shape[-1]
                self.pseudograd_cov[module_name] = torch.zeros(
                    (last_dim, last_dim),
                    dtype=self.stats_dtype,
                    device=pseudograds.device,
                )
            self.pseudograd_cov[module_name].addmm_(pseudograds.t(), pseudograds)

    def _perform_forward_and_backward_pass(
        self, batch: Any, sample: bool = False, reduction: str = "sum"
    ) -> None:
        """Run a forward + backward pass to drive the EK-FAC hooks.

        Args:
            batch (Any):
                A single batch from the loader.
            sample (bool, optional):
                Whether to sample targets from the model output (true Fisher).
            reduction (str, optional):
                Reduction strategy passed to the task's training loss.
        """
        loss = self.task.get_train_loss(
            model=self.model,
            batch=batch,
            sample=sample,
            reduction=reduction,
        )
        loss.backward()

    # ------------------------------------------------------------------ EK-FAC fitting

    def fit_covariances(self, loader: torch.utils.data.DataLoader) -> None:
        """Fit the activation and pseudo-gradient covariances.

        Runs ``n_epoch`` passes over the loader, triggering the forward /
        backward hooks to accumulate uncentered second-moment statistics; the
        accumulators are then normalized by the total number of examples.

        Args:
            loader (DataLoader):
                Loader yielding the training data over which covariances are
                computed.
        """
        if self._covariance_done:
            self.logger.info("Covariance computation is already done. Skipping.")
            return

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
        self._covariance_done = True
        self.logger.info(f"Seen {examples_seen} examples.")
        self.logger.info(f"Time for computing covariances: {time.time() - t0}")

    def fit_eigendecompositions(self, keep_cache: bool = False) -> None:
        """Eigendecompose the fitted covariances.

        Produces the per-module eigenvectors (``activation_cov_eigvecs`` /
        ``pseudograd_cov_eigvecs``) used as the EK-FAC basis. The covariance
        matrices themselves are dropped after decomposition unless
        ``keep_cache`` is set.

        Args:
            keep_cache (bool, optional):
                If ``False`` (default), drop the raw covariance matrices and
                their eigenvalues from memory once the eigenbasis has been
                extracted (downstream EK-FAC steps only need the
                eigenvectors). Set to ``True`` to retain everything for
                inspection.
        """
        if self._eigendecompositon_done:
            self.logger.info(
                "Eigendecomposition computation is already done. Skipping."
            )
            return

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
        self._eigendecompositon_done = True
        self.logger.info(f"Time for eigendecomposition: {time.time() - t1}")

    def compute_kronecker_lambda(
        self, module_name: str, per_batch_grads: torch.Tensor
    ) -> None:
        """Accumulate the EK-FAC corrected eigenvalues (Lambda) for one module.

        The corrected eigenvalues are the squared per-sample gradient
        coordinates in the EK-FAC eigenbasis, averaged over the data; see
        Eq. (53) of Grosse et al. (2023). Computed in vectorized form across
        the batch axis.

        Args:
            module_name (str):
                Qualified module name.
            per_batch_grads (torch.Tensor):
                Per-sample gradients shaped ``(batch_size, out_dim, in_dim)``.
        """
        assert len(per_batch_grads.shape) == 3

        if module_name not in self.kronecker_eigvals:
            self.kronecker_eigvals[module_name] = torch.zeros(
                (per_batch_grads.shape[1], per_batch_grads.shape[2]),
                dtype=self.grads_dtype,
                device=per_batch_grads.device,
            )
        # Rotate gradients into the EK-FAC eigenbasis: U_S^T @ grads @ U_A.
        # The Lambda factor is the squared rotated gradient summed across the
        # batch. This vectorized form avoids a Python-level batch loop.
        grads_rot = torch.matmul(
            self.pseudograd_cov_eigvecs[module_name].t(),
            torch.matmul(per_batch_grads, self.activation_cov_eigvecs[module_name]),
        )
        self.kronecker_eigvals[module_name].add_(torch.square(grads_rot).sum(dim=0))

    def compute_full_factors(
        self, module_name: str, per_batch_grads: torch.Tensor
    ) -> None:
        """Accumulate the full Fisher for a LayerNorm / BatchNorm module.

        Args:
            module_name (str):
                Qualified module name.
            per_batch_grads (torch.Tensor):
                Per-sample gradients shaped ``(batch_size, num_params)``.
        """
        assert len(per_batch_grads.shape) == 2
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
        """Accumulate the diagonal Fisher for an Embedding module.

        Args:
            module_name (str):
                Qualified module name.
            per_batch_grads (torch.Tensor):
                Per-sample gradients shaped ``(batch_size, num_emb, emb_dim)``.
        """
        assert len(per_batch_grads.shape) == 3
        if module_name not in self.diag_factors:
            self.diag_factors[module_name] = torch.zeros(
                (per_batch_grads.shape[1], per_batch_grads.shape[2]),
                dtype=self.grads_dtype,
                device=per_batch_grads.device,
            )
        self.diag_factors[module_name].add_(torch.square(per_batch_grads).sum(dim=0))

    def fit_additional_factors(self, loader: torch.utils.data.DataLoader) -> None:
        """Fit the per-module Fisher quantities used for preconditioning.

        Runs ``n_epoch`` passes over the loader, computing per-sample gradients
        via ``torch.func.vmap`` and dispatching to ``compute_kronecker_lambda``
        / ``compute_full_factors`` / ``compute_diag_lambda`` per module. Also
        sets up the per-module damping factor: ``self.damping`` if provided,
        otherwise the ``0.1 * mean(eigvals)`` heuristic.

        Args:
            loader (DataLoader):
                Loader yielding the training data.
        """
        if self._additional_factors_done:
            self.logger.info(
                "Additional factors computation is already done. Skipping."
            )
            return

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
                            per_batch_grads = make_grads_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_kronecker_lambda(name, per_batch_grads)
                        elif name in self.full_modules_name:
                            per_batch_grads = make_grads_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_full_factors(name, per_batch_grads)
                        elif name in self.diag_modules_name:
                            per_batch_grads = make_grads_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_diag_lambda(name, per_batch_grads)
                        else:
                            raise InvalidModuleError()

        with torch.no_grad():
            # Normalize and set up damping per module.
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
                    self.damping_factors[name] = 0.1 * torch.mean(eigvals).to(
                        dtype=self.grads_dtype
                    )
                else:
                    self.damping_factors[name] = self.damping

                # Invert and re-densify in one shot: (V diag(eigvals + lambda) V^T).
                # `self.full_factors[name]` will be used directly as the
                # damped inverse downstream.
                inv_eigvals = eigvals + self.damping_factors[name]
                self.full_factors[name] = torch.matmul(
                    torch.matmul(eigvecs, torch.diag(inv_eigvals)), eigvecs.t()
                )
                self.full_factors[name] = self.full_factors[name].to(self.grads_dtype)

            for name in self.diag_modules_name:
                self.diag_factors[name] /= examples_seen
                self.diag_factors[name] = self.diag_factors[name].to(self.grads_dtype)
                # Match the heuristic damping behavior used by the Kronecker /
                # full branches: a `None` damping triggers the
                # ``0.1 * mean(eigvals)`` heuristic.
                if self.damping is None:
                    self.damping_factors[name] = 0.1 * torch.mean(
                        self.diag_factors[name]
                    )
                else:
                    self.damping_factors[name] = self.damping

        self._additional_factors_done = True
        self.logger.info(f"Time for computing Lambda: {time.time() - t2}")

    def build_curvature_blocks(
        self, loader: torch.utils.data.DataLoader, keep_cache: bool = False
    ) -> None:
        """Fit every EK-FAC factor end-to-end.

        Runs :meth:`fit_covariances`, :meth:`fit_eigendecompositions`, removes
        the hooks, and runs :meth:`fit_additional_factors`. After this call the
        computer is ready for score computation.

        Args:
            loader (DataLoader):
                Loader yielding the training data.
            keep_cache (bool, optional):
                Forwarded to :meth:`fit_eigendecompositions`. Defaults to
                ``False``.
        """
        self.fit_covariances(loader=loader)
        self.fit_eigendecompositions(keep_cache=keep_cache)
        self.remove_handles()
        self.fit_additional_factors(loader=loader)

    def remove_handles(self) -> None:
        """Remove all forward and backward hooks installed by :meth:`initialize`."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------ preconditioning

    def precondition_grads(
        self,
        module_name: str,
        grads: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the inverse-Hessian preconditioner to one module's gradients.

        Dispatches by Fisher branch:

          * Kronecker: rotate ``grads`` into the EK-FAC eigenbasis, divide by
            ``eigvals + lambda``, rotate back.
          * Full: matmul with the precomputed damped-inverse dense factor.
          * Diagonal: divide elementwise by ``diag + lambda``.

        Args:
            module_name (str):
                Qualified module name.
            grads (torch.Tensor):
                Per-sample gradients for this module, with leading batch axis.

        Returns:
            torch.Tensor: the preconditioned gradients, same shape as ``grads``.

        Raises:
            InvalidModuleError: if ``module_name`` is unknown.
        """
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
            # `full_factors[module_name]` is the already-inverted damped Fisher.
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

    def _get_grads_dict(
        self, batch: Any, use_measurement: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample gradients (training-loss or measurement) via vmap.

        Args:
            batch (Any):
                A single batch.
            use_measurement (bool, optional):
                If ``True``, differentiate ``get_measurement``; if ``False``
                (default), differentiate ``get_train_loss``.

        Returns:
            Dict[str, torch.Tensor]: per-parameter per-sample gradients.
        """
        grads_dict = torch.func.vmap(
            self._compute_measurement_grad()
            if use_measurement
            else self._compute_train_loss_grad(),
            in_dims=(None, None, 0),
            randomness="different",
        )(self.func_params, self.func_buffers, batch)
        return grads_dict

    def _get_precond_grads_dict(
        self,
        batch: Any,
        use_measurement: bool = False,
        disable_precondition: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample gradients and apply the preconditioner per module.

        Args:
            batch (Any):
                A single batch.
            use_measurement (bool, optional):
                If ``True``, differentiate the measurement function; otherwise
                the training loss.
            disable_precondition (bool, optional):
                If ``True``, skip preconditioning (i.e. treat the Hessian as
                the identity).

        Returns:
            Dict[str, torch.Tensor]: per-module preconditioned gradients,
            flattened to ``(batch_size, num_params)``.
        """
        batch_size = self.task.get_batch_size(batch)
        grads_dict = self._get_grads_dict(batch=batch, use_measurement=use_measurement)

        with torch.no_grad():
            precond_grads_dict = {}
            for name, module in zip(self.modules_name, self.modules):
                grads = make_grads_dict_to_matrix(
                    module, name, grads_dict, remove_grads=True
                )
                if disable_precondition:
                    precond_grads_dict[name] = grads.reshape(batch_size, -1).to(
                        dtype=self.grads_dtype
                    )
                else:
                    precond_grads_dict[name] = (
                        self.precondition_grads(module_name=name, grads=grads)
                        .reshape(batch_size, -1)
                        .to(dtype=self.grads_dtype)
                    )
        del grads_dict
        return precond_grads_dict

    # ------------------------------------------------------------------ scoring

    def compute_scores_with_batch(
        self, batch1: Any, batch2: Any, disable_precondition: bool = False
    ) -> torch.Tensor:
        """Compute pairwise influence scores for a pair of batches.

        ``batch1`` is treated as the query batch (its gradients are
        preconditioned using ``get_measurement``); ``batch2`` is treated as
        the training batch (raw ``get_train_loss`` gradients).

        Args:
            batch1 (Any):
                Query (test) batch.
            batch2 (Any):
                Training batch.
            disable_precondition (bool, optional):
                If ``True``, skip preconditioning (identity Hessian).

        Returns:
            torch.Tensor: ``(|batch1|, |batch2|)`` table of pairwise scores.
        """
        self.model.eval()
        precond_grads_dict1 = self._get_precond_grads_dict(
            batch=batch1,
            use_measurement=True,
            disable_precondition=disable_precondition,
        )
        grads_dict2 = self._get_grads_dict(batch=batch2, use_measurement=False)

        with torch.no_grad():
            batch_size = self.task.get_batch_size(batch2)
            total_score = 0.0

            for name, module in zip(self.modules_name, self.modules):
                grads = (
                    make_grads_dict_to_matrix(module, name, grads_dict2)
                    .reshape(batch_size, -1)
                    .to(dtype=self.grads_dtype)
                )
                if isinstance(total_score, float):
                    total_score = torch.matmul(precond_grads_dict1[name], grads.t())
                else:
                    total_score.addmm_(precond_grads_dict1[name], grads.t())
                del precond_grads_dict1[name], grads
        return total_score

    def compute_scores_with_loader(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        disable_precondition: bool = False,
    ) -> torch.Tensor:
        """Compute pairwise influence scores between two loaders.

        Iterates the test loader in the outer loop and the train loader in
        the inner loop. The query side uses ``get_measurement``; the train
        side uses ``get_train_loss``.

        Args:
            test_loader (DataLoader):
                Loader yielding query (test) data.
            train_loader (DataLoader):
                Loader yielding training data.
            disable_precondition (bool, optional):
                If ``True``, skip preconditioning (identity Hessian).

        Returns:
            torch.Tensor: ``(num_test, num_train)`` table of pairwise scores.
        """
        self.model.eval()

        score_table = torch.zeros(
            (len(test_loader.dataset), len(train_loader.dataset)),
            dtype=self.score_dtype,
            device=self.task.device,
            requires_grad=False,
        )

        num_processed_test = 0
        for test_batch in test_loader:
            test_batch_size = self.task.get_batch_size(test_batch)
            self.logger.info(
                f"Processing test batch [{num_processed_test}, "
                f"{num_processed_test + test_batch_size}) of "
                f"{len(test_loader.dataset)}."
            )
            precond_test_grads_dict = self._get_precond_grads_dict(
                batch=test_batch,
                use_measurement=True,
                disable_precondition=disable_precondition,
            )

            num_processed_train = 0
            for train_batch in train_loader:
                train_batch_size = self.task.get_batch_size(train_batch)
                train_grads_dict = self._get_grads_dict(
                    batch=train_batch, use_measurement=False
                )

                with torch.no_grad():
                    for name, module in zip(self.modules_name, self.modules):
                        train_grads = (
                            make_grads_dict_to_matrix(module, name, train_grads_dict)
                            .reshape(train_batch_size, -1)
                            .to(dtype=self.grads_dtype)
                        )
                        score_table[
                            num_processed_test : num_processed_test + test_batch_size,
                            num_processed_train : num_processed_train
                            + train_batch_size,
                        ].addmm_(precond_test_grads_dict[name], train_grads.t())
                        del train_grads
                num_processed_train += train_batch_size
            del precond_test_grads_dict
            num_processed_test += test_batch_size
        return score_table

    def compute_self_scores_with_loader(
        self, loader: torch.utils.data.DataLoader, disable_precondition: bool = False
    ) -> torch.Tensor:
        """Compute self-influence scores for every data point in a loader.

        For each data point ``z``, returns ``grad L(z)^T H^{-1} grad L(z)``,
        i.e. the diagonal of the IF score table when ``train == test`` and
        both use the training loss.

        Args:
            loader (DataLoader):
                Loader yielding the data points for which self-influences
                will be computed.
            disable_precondition (bool, optional):
                If ``True``, skip preconditioning (identity Hessian).

        Returns:
            torch.Tensor: 1-D tensor of self-influence scores, length equal
            to the size of the loader's dataset.
        """
        self.model.eval()

        scores = []
        for batch in loader:
            batch_size = self.task.get_batch_size(batch)
            current_score = torch.zeros(
                (batch_size,),
                dtype=self.score_dtype,
                device=self.task.device,
                requires_grad=False,
            )
            grads_dict = self._get_grads_dict(batch=batch, use_measurement=False)

            with torch.no_grad():
                for name, module in zip(self.modules_name, self.modules):
                    grads = make_grads_dict_to_matrix(
                        module, name, grads_dict, remove_grads=True
                    )
                    if disable_precondition:
                        precond_grads = grads.reshape(batch_size, -1).to(
                            dtype=self.grads_dtype
                        )
                    else:
                        precond_grads = (
                            self.precondition_grads(module_name=name, grads=grads)
                            .reshape(batch_size, -1)
                            .to(dtype=self.grads_dtype)
                        )
                    grads = grads.reshape(batch_size, -1).to(dtype=self.grads_dtype)
                    current_score.add_(torch.sum(precond_grads * grads, dim=-1))

                del grads_dict
                scores.append(current_score)
        return torch.cat(scores)
