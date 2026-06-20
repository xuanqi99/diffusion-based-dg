"""Sharpness-Aware Gradient Matching for domain generalization.

SAGM augments pooled-source ERM with a second forward/backward pass at a
gradient-dependent parameter perturbation. The final update averages the
original and perturbed gradients, encouraging low empirical risk, local
flatness, and a small surrogate gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Type

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from erm import ERMTrainer, Minibatch, combine_source_minibatches


RhoSchedule = Callable[[int], float]
OptimizerType = Type[torch.optim.Optimizer]


@dataclass(frozen=True)
class SAGMOutput:
    """Values produced by one SAGM optimization step."""

    loss: Tensor
    perturbed_loss: Tensor
    logits: Tensor
    targets: Tensor
    accuracy: Tensor
    rho: float


class SAGMTrainer(ERMTrainer):
    """An ERM classifier intended to be optimized with ``SAGMOptimizer``."""


class LinearRhoScheduler:
    """Linearly interpolate rho over a fixed number of optimization steps."""

    def __init__(
        self,
        total_steps: int,
        *,
        start_value: float = 0.05,
        end_value: float = 0.05,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1.")
        if start_value < 0 or end_value < 0:
            raise ValueError("rho values must be non-negative.")
        self.total_steps = total_steps
        self.start_value = start_value
        self.end_value = end_value

    def __call__(self, step: int) -> float:
        progress = min(max(step, 0), self.total_steps) / self.total_steps
        return self.start_value + progress * (self.end_value - self.start_value)


def _disable_running_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            if not hasattr(module, "_sagm_backup_momentum"):
                module._sagm_backup_momentum = module.momentum
            module.momentum = 0.0


def _enable_running_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, _BatchNorm) and hasattr(module, "_sagm_backup_momentum"):
            module.momentum = module._sagm_backup_momentum
            del module._sagm_backup_momentum


class SAGMOptimizer(torch.optim.Optimizer):
    """Wrap a PyTorch optimizer with the two-pass SAGM update.

    ``alpha`` controls the gradient-matching offset and ``rho`` controls the
    sharpness neighborhood. The official SAGM configuration uses non-adaptive
    perturbations with ``rho=0.05`` and dataset-dependent ``alpha`` values.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        *,
        alpha: float = 0.001,
        rho: float = 0.05,
        rho_scheduler: RhoSchedule | None = None,
        adaptive: bool = False,
        eps: float = 1e-12,
        sync_distributed: bool = True,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if rho < 0:
            raise ValueError("rho must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        parameters = [
            parameter
            for group in base_optimizer.param_groups
            for parameter in group["params"]
        ]
        if not parameters:
            raise ValueError("base_optimizer does not contain any parameters.")

        super().__init__(parameters, defaults={"adaptive": adaptive})
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.model = model
        self.alpha = alpha
        self.initial_rho = rho
        self.rho_scheduler = rho_scheduler
        self.adaptive = adaptive
        self.eps = eps
        self.sync_distributed = sync_distributed
        self.step_index = 0

    @property
    def rho(self) -> float:
        if self.rho_scheduler is None:
            return self.initial_rho
        value = float(self.rho_scheduler(self.step_index))
        if value < 0:
            raise ValueError("rho_scheduler returned a negative value.")
        return value

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _gradient_norm(self) -> Tensor:
        norms = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if self.adaptive:
                    gradient = parameter.detach().abs() * gradient
                norms.append(gradient.norm(p=2))
        if not norms:
            raise RuntimeError("SAGM received no gradients from the first pass.")
        return torch.stack(norms).norm(p=2)

    @torch.no_grad()
    def _perturb(self, rho: float) -> None:
        gradient_norm = self._gradient_norm()
        scale = rho / (gradient_norm + self.eps) - self.alpha

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                state["original_gradient"] = parameter.grad.detach().clone()
                perturbation = parameter.grad * scale.to(parameter)
                if self.adaptive:
                    perturbation = perturbation * parameter.detach().pow(2)
                parameter.add_(perturbation)
                state["perturbation"] = perturbation

    @torch.no_grad()
    def _restore_parameters(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                perturbation = self.state[parameter].get("perturbation")
                if perturbation is not None:
                    parameter.sub_(perturbation)
                    del self.state[parameter]["perturbation"]

    @torch.no_grad()
    def _merge_gradients(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                original = self.state[parameter].get("original_gradient")
                if original is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = original.clone()
                else:
                    parameter.grad.mul_(0.5).add_(original, alpha=0.5)
                del self.state[parameter]["original_gradient"]

    @torch.no_grad()
    def _clear_saved_gradients(self) -> None:
        for state in self.state.values():
            state.pop("original_gradient", None)

    @torch.no_grad()
    def _sync_gradients(self) -> None:
        if not self.sync_distributed or not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                torch.distributed.all_reduce(
                    parameter.grad,
                    op=torch.distributed.ReduceOp.SUM,
                )
                parameter.grad.div_(world_size)

    def step(
        self,
        closure: Callable[[], tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        """Perform one SAGM update using a logits-and-loss closure."""

        if closure is None:
            raise ValueError("SAGMOptimizer.step requires a closure.")

        current_rho = self.rho
        _enable_running_stats(self.model)
        self.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits, loss = closure()
        loss.backward()
        first_logits = logits.detach()
        first_loss = loss.detach()

        try:
            self._perturb(current_rho)
            _disable_running_stats(self.model)
            self.zero_grad(set_to_none=True)
            with torch.enable_grad():
                _perturbed_logits, perturbed_loss = closure()
            perturbed_loss.backward()
            detached_perturbed_loss = perturbed_loss.detach()
            self._merge_gradients()
            self._restore_parameters()
            self._sync_gradients()
            self.base_optimizer.step()
        except Exception:
            self._restore_parameters()
            self._clear_saved_gradients()
            self.zero_grad(set_to_none=True)
            raise
        finally:
            _enable_running_stats(self.model)

        self.step_index += 1
        return first_logits, first_loss, detached_perturbed_loss, current_rho

    def state_dict(self) -> dict[str, object]:
        return {
            "sagm": super().state_dict(),
            "base_optimizer": self.base_optimizer.state_dict(),
            "step_index": self.step_index,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        super().load_state_dict(state_dict["sagm"])
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups
        self.step_index = int(state_dict.get("step_index", 0))


def build_sagm_optimizer(
    trainer: SAGMTrainer,
    *,
    optimizer_type: OptimizerType = torch.optim.Adam,
    lr: float = 3e-5,
    weight_decay: float = 1e-4,
    alpha: float = 0.001,
    rho: float = 0.05,
    rho_scheduler: RhoSchedule | None = None,
    adaptive: bool = False,
    **optimizer_kwargs: object,
) -> SAGMOptimizer:
    """Construct a base optimizer and wrap it with SAGM."""

    base_optimizer = optimizer_type(
        trainer.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )
    return SAGMOptimizer(
        base_optimizer,
        trainer,
        alpha=alpha,
        rho=rho,
        rho_scheduler=rho_scheduler,
        adaptive=adaptive,
    )


def sagm_step(
    trainer: SAGMTrainer,
    optimizer: SAGMOptimizer,
    inputs: Tensor | None = None,
    targets: Tensor | None = None,
    *,
    minibatches: Sequence[Minibatch] | None = None,
) -> SAGMOutput:
    """Run one two-pass SAGM update on one batch or pooled source domains."""

    uses_batch = inputs is not None or targets is not None
    if uses_batch == (minibatches is not None):
        raise ValueError(
            "Provide either inputs and targets or minibatches, but not both."
        )
    if minibatches is not None:
        inputs, targets = combine_source_minibatches(minibatches)
    elif inputs is None or targets is None:
        raise ValueError("Both inputs and targets are required for a single batch.")

    trainer.train()

    def closure() -> tuple[Tensor, Tensor]:
        output = trainer.training_loss(inputs, targets)
        return output.logits, output.loss

    logits, loss, perturbed_loss, rho = optimizer.step(closure)
    accuracy = logits.argmax(dim=1).eq(targets).float().mean()
    return SAGMOutput(
        loss=loss,
        perturbed_loss=perturbed_loss,
        logits=logits,
        targets=targets.detach(),
        accuracy=accuracy,
        rho=rho,
    )
