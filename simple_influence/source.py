"""SOURCE: training data attribution via approximate unrolled differentiation.

Reference:
    Bae, Lin, Lorraine, Grosse. "Training Data Attribution via Approximate Unrolled
    Differentiation." arXiv:2405.12186 (2024).

SOURCE partitions the training trajectory into L segments and approximates the
average gradient and Hessian within each segment as stationary. The final score
(Equation 24 in the paper) is

    tau(z_q, z_m) = grad f(z_q, theta_s)^T  *  sum_{l=1..L} ( prod_{l'=L..l+1} S_l' ) r_l

where, per segment l:

    S_l   = exp(-eta_l * K_l * H_l)                              (matrix function)
    r_l   = (1 / N) * (I - exp(-eta_l * K_l * H_l)) * H_l^{-1} * g_l

H_l is the segment-averaged Hessian, g_l is the segment-averaged training gradient,
eta_l is the averaged learning rate, and K_l is the number of training iterations
in the segment. We approximate H_l with EK-FAC, so both S_l and r_l reduce to
elementwise matrix functions applied to the corrected eigenvalues.

For L = 1, SOURCE collapses to an EK-FAC influence function with damping
lambda = 1 / (eta_1 * K_1), i.e. the damping is *derived* rather than hand-tuned
(see Section 3.2 of the paper).

Differences from the paper:
    - The global `1/N` factor in r_l is omitted, matching the convention used by
      `InfluenceFunctionComputer` in this repo. Rankings are unaffected; absolute
      scores differ by a factor of N.
    - SGD-with-momentum learning-rate scaling (Appendix D, paragraph below
      Eq. 54: `eta_l <- eta_l / (1 - beta)`) is NOT applied automatically. If you
      trained with momentum, pass the pre-scaled learning rate via
      `lrs_per_segment`.
    - The Adam-aware variant of SOURCE (Appendix D, last paragraph, involving the
      Adam preconditioner `P_l`) is not implemented. This computer assumes plain
      SGD-style updates.
    - The F_r function `(1 - exp(-eta * K * sigma)) / sigma` is `0/0` at
      `sigma = 0`; we guard with `nan_to_num` + `clamp(max=eta*K)` (which is the
      analytic limit) for numerical stability.
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from simple_influence.abstract_computer import AbstractComputer
from simple_influence.abstract_task import AbstractTask
from simple_influence.ekfac_utils import make_grads_dict_to_matrix
from simple_influence.influence_function import InfluenceFunctionComputer


class _SegmentFactors:
    """EK-FAC factors averaged across the checkpoints of a single segment.

    Holds, per supported (Linear / Conv2d) module:
      - `activation_cov_eigvecs`: Q_A from the eigendecomposition of the averaged
        activation covariance.
      - `pseudograd_cov_eigvecs`: Q_S from the eigendecomposition of the averaged
        pseudo-gradient covariance.
      - `kronecker_eigvals`: the averaged corrected eigenvalues (Lambda) in the
        Q_A (x) Q_S basis.

    Also caches the per-module nn.Module handles so callers can iterate over them
    in the same order at score time.
    """

    def __init__(self) -> None:
        self.modules_name: List[str] = []
        self.modules: List[nn.Module] = []
        self.activation_cov_eigvecs: Dict[str, torch.Tensor] = {}
        self.pseudograd_cov_eigvecs: Dict[str, torch.Tensor] = {}
        self.kronecker_eigvals: Dict[str, torch.Tensor] = {}


class SourceComputer(AbstractComputer):
    """SOURCE training-data attribution.

    Args:
        model (nn.Module):
            PyTorch model whose current parameters are the *final* parameters
            (theta_s in the paper). The query gradient is computed at these
            parameters, so make sure the final checkpoint is loaded before
            constructing this class.
        task (AbstractTask):
            Task definition for the pipeline.
        checkpoints_per_segment (list[list[str]]):
            One inner list per segment, in chronological order
            (`checkpoints_per_segment[0]` is the earliest segment,
             `checkpoints_per_segment[-1]` is the latest). Each inner list holds
            the paths to the model checkpoints whose EK-FAC factors will be
            averaged for that segment.
        iters_per_segment (list[int]):
            Number of gradient updates K_l within each segment.
        lrs_per_segment (list[float]):
            Averaged learning rate eta_l within each segment.
        n_epoch (int, optional):
            Number of epochs over `loader` used to estimate per-checkpoint
            covariances. Defaults to 1, matching `InfluenceFunctionComputer`.
        use_true_fisher (bool, optional):
            If True, sample targets from the model output when computing
            pseudo-gradients (true Fisher); if False, use the actual labels
            (empirical Fisher). Defaults to True.

    Notes:
        - Only `nn.Linear` and `nn.Conv2d` modules are supported. SOURCE applies
          a matrix function (eta_l * K_l * H_l + ...) to the EK-FAC eigenvalues;
          the LayerNorm/BatchNorm "full" and Embedding "diagonal" Fisher branches
          do not have a closed form for this matrix function, so they are not
          covered here.
        - Compared to influence functions, SOURCE is roughly C times more
          expensive, where C is the total number of checkpoints across all
          segments.
    """

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        checkpoints_per_segment: List[List[str]],
        iters_per_segment: List[int],
        lrs_per_segment: List[float],
        n_epoch: int = 1,
        use_true_fisher: bool = True,
    ) -> None:
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        if not (
            len(checkpoints_per_segment)
            == len(iters_per_segment)
            == len(lrs_per_segment)
        ):
            raise ValueError(
                "checkpoints_per_segment, iters_per_segment, and lrs_per_segment "
                "must have the same length (= number of segments)."
            )
        if len(checkpoints_per_segment) == 0:
            raise ValueError("Need at least one segment.")
        for seg_idx, ckpts in enumerate(checkpoints_per_segment):
            if len(ckpts) == 0:
                raise ValueError(f"Segment {seg_idx} has no checkpoints.")

        self.checkpoints_per_segment = checkpoints_per_segment
        self.iters_per_segment = iters_per_segment
        self.lrs_per_segment = lrs_per_segment
        self.n_epoch = n_epoch
        self.use_true_fisher = use_true_fisher

        # Snapshot the final parameters: they are theta_s used for the query
        # gradient. We restore these whenever we have to mutate model state to
        # load a checkpoint for covariance / gradient computation.
        self.final_state_dict = copy.deepcopy(self.model.state_dict())
        self.final_func_params = copy.deepcopy(dict(self.model.named_parameters()))
        self.final_func_buffers = copy.deepcopy(dict(self.model.named_buffers()))

        # One `_SegmentFactors` per segment, populated by `build_curvature_blocks`.
        self.segments: List[_SegmentFactors] = []

    # ------------------------------------------------------------------ utils

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a state dict into `self.model` and place it on `task.device`."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.task.device)
        self.model.eval()

    def _restore_final_params(self) -> None:
        """Restore the original final parameters (theta_s) onto `self.model`."""
        self.model.load_state_dict(self.final_state_dict)
        self.model = self.model.to(self.task.device)
        self.model.eval()

    def _build_inner_computer(self) -> InfluenceFunctionComputer:
        """Construct an `InfluenceFunctionComputer` whose state we will reuse.

        We immediately strip the hooks it registers so we control when they fire.
        `damping=0.0` keeps preconditioning identity-like; SOURCE supplies its
        own matrix function in `compute_scores_with_loader`.
        """
        ifc = InfluenceFunctionComputer(
            model=self.model,
            task=self.task,
            damping=0.0,
            n_epoch=self.n_epoch,
            use_true_fisher=self.use_true_fisher,
        )
        for handle in ifc._handles:
            handle.remove()
        ifc._handles = []
        return ifc

    # ------------------------------------------------------------------ curvature

    def build_curvature_blocks(self, loader: torch.utils.data.DataLoader) -> None:
        """Build per-segment averaged EK-FAC factors.

        For each segment l, this:
          1. Loads each checkpoint in the segment and fits per-checkpoint
             activation and pseudo-gradient covariances.
          2. Averages the covariances across checkpoints and eigendecomposes
             the averages to obtain Q_A and Q_S (the segment's eigenbasis).
          3. Re-visits each checkpoint and computes the corrected EK-FAC
             eigenvalues in this shared eigenbasis, then averages them.

        See Appendix D.2 of the paper.
        """
        self.segments.clear()

        for seg_idx, ckpts in enumerate(self.checkpoints_per_segment):
            self.logger.info(
                f"Building curvature for segment {seg_idx + 1}/"
                f"{len(self.checkpoints_per_segment)} ({len(ckpts)} checkpoints)."
            )

            # Step 1: per-checkpoint covariances, collected on CPU to bound peak memory.
            act_cov_acc: Dict[str, List[torch.Tensor]] = {}
            grad_cov_acc: Dict[str, List[torch.Tensor]] = {}
            kronecker_module_names: Optional[List[str]] = None
            kronecker_modules: Optional[List[nn.Module]] = None

            for ckpt in ckpts:
                self._load_checkpoint(ckpt)
                ifc = InfluenceFunctionComputer(
                    model=self.model,
                    task=self.task,
                    damping=0.0,
                    n_epoch=self.n_epoch,
                    use_true_fisher=self.use_true_fisher,
                )
                ifc.fit_covariances(loader=loader)
                # The hooks are no longer needed once covariances are accumulated;
                # remove them so the next computation does not double-accumulate.
                for handle in ifc._handles:
                    handle.remove()
                ifc._handles = []

                if kronecker_module_names is None:
                    kronecker_module_names = list(ifc.kronecker_modules_name)
                    # Cache the module objects in the same order; needed for
                    # `make_grads_dict_to_matrix` later.
                    name_to_module = dict(zip(ifc.modules_name, ifc.modules))
                    kronecker_modules = [
                        name_to_module[n] for n in kronecker_module_names
                    ]

                for name in kronecker_module_names:
                    act_cov_acc.setdefault(name, []).append(
                        ifc.activation_cov[name].detach().cpu()
                    )
                    grad_cov_acc.setdefault(name, []).append(
                        ifc.pseudograd_cov[name].detach().cpu()
                    )
                del ifc

            assert kronecker_module_names is not None
            assert kronecker_modules is not None
            if len(kronecker_module_names) == 0:
                raise RuntimeError(
                    "SOURCE found no supported (Linear / Conv2d) influence modules. "
                    "Check `task.influence_modules()`."
                )

            # Step 2: average covariances, eigendecompose to get the segment's
            # shared eigenbasis. We do this through a fresh `InfluenceFunctionComputer`
            # mostly to reuse `fit_eigendecompositions`.
            seg_ifc = self._build_inner_computer()
            device = self.task.device
            with torch.no_grad():
                for name in kronecker_module_names:
                    seg_ifc.activation_cov[name] = (
                        torch.stack(act_cov_acc[name]).mean(0).to(device=device)
                    )
                    seg_ifc.pseudograd_cov[name] = (
                        torch.stack(grad_cov_acc[name]).mean(0).to(device=device)
                    )
            seg_ifc._covariance_done = True
            # Keep the covariance cache (set to True) so the eigvecs are written
            # without trying to free state we never populated for non-kronecker modules.
            seg_ifc.fit_eigendecompositions(keep_cache=True)

            # Step 3: per-checkpoint corrected eigenvalues, computed in the
            # segment's averaged eigenbasis. We swap the eigenbasis into a
            # disposable IFC instance and let `fit_additional_factors` do the work.
            lambda_acc: Dict[str, List[torch.Tensor]] = {}
            for ckpt in ckpts:
                self._load_checkpoint(ckpt)
                ckpt_ifc = self._build_inner_computer()
                # Use the *averaged* eigenbasis from this segment, not whatever
                # `ckpt_ifc` would otherwise pick up.
                with torch.no_grad():
                    for name in kronecker_module_names:
                        ckpt_ifc.activation_cov_eigvecs[
                            name
                        ] = seg_ifc.activation_cov_eigvecs[name]
                        ckpt_ifc.pseudograd_cov_eigvecs[
                            name
                        ] = seg_ifc.pseudograd_cov_eigvecs[name]
                ckpt_ifc.fit_additional_factors(loader=loader)
                with torch.no_grad():
                    for name in kronecker_module_names:
                        lambda_acc.setdefault(name, []).append(
                            ckpt_ifc.kronecker_eigvals[name].detach().cpu()
                        )
                del ckpt_ifc

            seg_factors = _SegmentFactors()
            seg_factors.modules_name = kronecker_module_names
            seg_factors.modules = kronecker_modules
            with torch.no_grad():
                for name in kronecker_module_names:
                    seg_factors.activation_cov_eigvecs[
                        name
                    ] = seg_ifc.activation_cov_eigvecs[name]
                    seg_factors.pseudograd_cov_eigvecs[
                        name
                    ] = seg_ifc.pseudograd_cov_eigvecs[name]
                    seg_factors.kronecker_eigvals[name] = (
                        torch.stack(lambda_acc[name]).mean(0).to(device=device)
                    )
            self.segments.append(seg_factors)
            del seg_ifc

        # Restore final parameters so subsequent forward passes use theta_s.
        self._restore_final_params()

    # ------------------------------------------------------------------ matrix functions

    @staticmethod
    def _apply_matrix_function(
        grads_per_module: Dict[str, torch.Tensor],
        segment: _SegmentFactors,
        factor_fn,
    ) -> Dict[str, torch.Tensor]:
        """Apply a scalar matrix function of H_l to a dict of per-module gradients.

        For each module, the EK-FAC representation says H_l (acting on the
        flattened weight gradient) has eigenvalues `kronecker_eigvals` in the
        Q_A (x) Q_S basis. So applying f(H_l) is just:
            f(H_l) * grad   ==   Q_S * ( f(Lambda) ⊙ ( Q_S^T * grad * Q_A ) ) * Q_A^T
        where grad has shape (batch, out_dim, in_dim) and the elementwise factor
        broadcasts over the batch axis.

        `factor_fn(eigvals)` returns the (out_dim, in_dim) tensor of scalar
        matrix-function values applied to each corrected eigenvalue.
        """
        out: Dict[str, torch.Tensor] = {}
        for name in segment.modules_name:
            q_a = segment.activation_cov_eigvecs[name]
            q_s = segment.pseudograd_cov_eigvecs[name]
            eigvals = segment.kronecker_eigvals[name]
            factor = factor_fn(eigvals)
            grads = grads_per_module[name]
            rotated = torch.matmul(q_s.t(), torch.matmul(grads, q_a))
            rotated = rotated * factor.unsqueeze(0)
            out[name] = torch.matmul(q_s, torch.matmul(rotated, q_a.t()))
        return out

    @staticmethod
    def _r_factor(eigvals: torch.Tensor, eta: float, k: int) -> torch.Tensor:
        """The F_r matrix function (paper Eq. 21): (1 - exp(-eta * K * sigma)) / sigma.

        We use `-expm1(-x)` for numerical stability when `eta * K * sigma` is small.
        The limit at sigma -> 0 is `eta * K`; we clamp/fill any sigmas at or below
        zero (which can happen from float rounding on an empirical Fisher
        eigenvalue) to that value to avoid NaNs.
        """
        scale = eta * k
        # `-expm1(-x)` = 1 - exp(-x); both numerator and denominator vanish at 0,
        # so guard with `nan_to_num` (sigma == 0) and `clamp` (sigma very small).
        result = -torch.expm1(-scale * eigvals) / eigvals
        result = torch.nan_to_num(result, nan=scale, posinf=scale, neginf=scale)
        return result.clamp(max=scale)

    @staticmethod
    def _s_factor(eigvals: torch.Tensor, eta: float, k: int) -> torch.Tensor:
        """The decay matrix function (paper Eq. 16): exp(-eta * K * sigma)."""
        return torch.exp(-eta * float(k) * eigvals)

    # ------------------------------------------------------------------ scoring

    def _measurement_grads_dict(
        self,
        batch: Any,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-sample measurement gradients via `torch.func.vmap`."""
        grad_fn = self._compute_measurement_grad()
        return torch.func.vmap(
            grad_fn, in_dims=(None, None, 0), randomness="different"
        )(params, buffers, batch)

    def _train_loss_grads_dict(
        self,
        batch: Any,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-sample training-loss gradients via `torch.func.vmap`."""
        grad_fn = self._compute_train_loss_grad()
        return torch.func.vmap(
            grad_fn, in_dims=(None, None, 0), randomness="different"
        )(params, buffers, batch)

    def compute_scores_with_loader(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Compute pairwise SOURCE scores between every test and training point.

        Implements Equation 24 of the paper. The outer iteration is over the
        test loader; for each test batch we walk segments from latest to earliest,
        accumulating training-gradient contributions.

        Args:
            test_loader (DataLoader): loader providing query data points.
            train_loader (DataLoader): loader providing training data points.

        Returns:
            A `(num_test, num_train)` tensor of attribution scores.
        """
        if not self.segments:
            raise RuntimeError(
                "No curvature factors available. Call `build_curvature_blocks` first."
            )

        self._restore_final_params()

        num_test = len(test_loader.dataset)
        num_train = len(train_loader.dataset)
        score_table = torch.zeros(
            (num_test, num_train),
            dtype=self.score_dtype,
            device=self.task.device,
            requires_grad=False,
        )

        num_processed_test = 0
        for test_batch in test_loader:
            test_batch_size = self.task.get_batch_size(test_batch)
            self.logger.info(
                f"Processing test batch [{num_processed_test}, "
                f"{num_processed_test + test_batch_size}) of {num_test}."
            )

            # Query gradient at the final parameters theta_s. Same convention as
            # `InfluenceFunctionComputer.compute_scores_with_loader`: query uses
            # the measurement function f(z_q, theta_s).
            self._restore_final_params()
            test_grads_dict = self._measurement_grads_dict(
                test_batch, self.final_func_params, self.final_func_buffers
            )
            # Convert per-parameter grads into per-module (batch, out_dim, in_dim)
            # matrices for any segment's module list (the module set is the same
            # across segments since it derives from `task.influence_modules()`).
            example_segment = self.segments[0]
            running_grads: Dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for name, module in zip(
                    example_segment.modules_name, example_segment.modules
                ):
                    running_grads[name] = make_grads_dict_to_matrix(
                        module=module,
                        module_name=name,
                        grads_dict=test_grads_dict,
                        remove_grads=True,
                    ).to(dtype=self.grads_dtype)
            del test_grads_dict

            # Walk segments from latest (highest index) to earliest. At each
            # step, `running_grads` already has all S_l' factors for later
            # segments applied; we apply F_r at the current segment to get
            # `preconditioned_grads`, score against this segment's training
            # gradients, then update `running_grads` with S for this segment
            # before moving to the earlier one.
            num_segments = len(self.segments)
            for seg_idx in range(num_segments - 1, -1, -1):
                segment = self.segments[seg_idx]
                eta = self.lrs_per_segment[seg_idx]
                k = self.iters_per_segment[seg_idx]

                # Default-arg trick captures the *current* values of `eta` /
                # `k`. The lambdas are called immediately by
                # `_apply_matrix_function`, so closing over the loop variables
                # would also be safe, but pylint flags `cell-var-from-loop`.
                preconditioned_grads = self._apply_matrix_function(
                    running_grads,
                    segment,
                    lambda eig, _eta=eta, _k=k: self._r_factor(eig, _eta, _k),
                )
                # Flatten for matmul against per-sample train gradients.
                preconditioned_flat: Dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    for name in segment.modules_name:
                        preconditioned_flat[name] = preconditioned_grads[name].reshape(
                            test_batch_size, -1
                        )
                del preconditioned_grads

                self._accumulate_segment_scores(
                    segment=segment,
                    seg_idx=seg_idx,
                    preconditioned_flat=preconditioned_flat,
                    train_loader=train_loader,
                    score_table=score_table,
                    num_processed_test=num_processed_test,
                    test_batch_size=test_batch_size,
                )
                del preconditioned_flat

                # Apply S for this segment to prepare `running_grads` for the
                # next earlier segment. Skip on the earliest segment (no more
                # iterations remain).
                if seg_idx > 0:
                    running_grads = self._apply_matrix_function(
                        running_grads,
                        segment,
                        lambda eig, _eta=eta, _k=k: self._s_factor(eig, _eta, _k),
                    )

            num_processed_test += test_batch_size

        self._restore_final_params()
        return score_table

    def _accumulate_segment_scores(
        self,
        segment: _SegmentFactors,
        seg_idx: int,
        preconditioned_flat: Dict[str, torch.Tensor],
        train_loader: torch.utils.data.DataLoader,
        score_table: torch.Tensor,
        num_processed_test: int,
        test_batch_size: int,
    ) -> None:
        """Average training-gradient inner products across this segment's checkpoints.

        For each checkpoint in the segment we load its parameters, compute
        per-sample training-loss gradients on the train loader, and accumulate
        `<preconditioned_query, train_grad>` into the score table - divided by
        the number of checkpoints in the segment so we end up with the segment's
        averaged gradient as required by SOURCE.
        """
        ckpts = self.checkpoints_per_segment[seg_idx]
        weight = 1.0 / float(len(ckpts))

        for ckpt in ckpts:
            self._load_checkpoint(ckpt)
            ckpt_params = dict(self.model.named_parameters())
            ckpt_buffers = dict(self.model.named_buffers())

            num_processed_train = 0
            for train_batch in train_loader:
                train_batch_size = self.task.get_batch_size(train_batch)
                train_grads_dict = self._train_loss_grads_dict(
                    train_batch, ckpt_params, ckpt_buffers
                )

                with torch.no_grad():
                    for name, module in zip(segment.modules_name, segment.modules):
                        train_grads = (
                            make_grads_dict_to_matrix(
                                module=module,
                                module_name=name,
                                grads_dict=train_grads_dict,
                                remove_grads=True,
                            )
                            .reshape(train_batch_size, -1)
                            .to(dtype=self.grads_dtype)
                        )
                        score_table[
                            num_processed_test : num_processed_test + test_batch_size,
                            num_processed_train : num_processed_train
                            + train_batch_size,
                        ].addmm_(
                            preconditioned_flat[name], train_grads.t(), alpha=weight
                        )
                        del train_grads
                num_processed_train += train_batch_size
                del train_grads_dict
