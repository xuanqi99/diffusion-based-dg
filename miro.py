"""PyTorch utilities for MIRO domain generalization.

MIRO (Mutual Information Regularization with Oracle) trains a target model with
the usual supervised loss plus a feature regularizer against a frozen
pre-trained model that approximates an oracle representation.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FeatureMap = Mapping[str, Tensor]
LossFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class MIROOutput:
    """Container returned by ``MIROTrainer.training_loss``."""

    total_loss: Tensor
    task_loss: Tensor
    miro_loss: Tensor
    logits: Tensor


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in eval mode and disable all gradients."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _as_tensor(output: object, layer_name: str) -> Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        f"Layer {layer_name!r} returned {type(output)!r}; expected a Tensor "
        "or a tuple/list whose first item is a Tensor."
    )


def _extract_logits(output: object) -> Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "pred", "prediction", "output"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    raise TypeError(
        "Could not infer logits from model output. Return a Tensor, put logits "
        "first in a tuple/list, or expose a 'logits' key."
    )


class HookedFeatureModel(nn.Module):
    """Run a model and collect outputs from named intermediate layers."""

    def __init__(self, model: nn.Module, feature_layers: Sequence[str]) -> None:
        super().__init__()
        if not feature_layers:
            raise ValueError("feature_layers must contain at least one layer name.")

        modules = dict(model.named_modules())
        missing = [name for name in feature_layers if name not in modules]
        if missing:
            available = ", ".join(list(modules.keys())[:20])
            raise ValueError(
                "Unknown feature layer(s): "
                f"{missing}. First available module names: {available}"
            )

        self.model = model
        self.feature_layers = tuple(feature_layers)
        self._features: MutableMapping[str, Tensor] = OrderedDict()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for name in self.feature_layers:
            self._handles.append(modules[name].register_forward_hook(self._save(name)))

    def _save(self, name: str) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            self._features[name] = _as_tensor(output, name)

        return hook

    def forward(self, inputs: Tensor) -> tuple[Tensor, OrderedDict[str, Tensor]]:
        self._features.clear()
        output = self.model(inputs)
        missing = [name for name in self.feature_layers if name not in self._features]
        if missing:
            raise RuntimeError(f"Forward pass did not produce feature(s): {missing}")
        features = OrderedDict((name, self._features[name]) for name in self.feature_layers)
        return _extract_logits(output), features

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class MIROFeatureRegularizer(nn.Module):
    """Gaussian variational MIRO regularizer with bias-only log variance.

    The mean encoder is the identity, and each selected feature level owns a
    learnable log-variance scalar that is broadcast over the full feature map.
    This keeps the implementation architecture-agnostic while following the
    lightweight encoder design used by MIRO.
    """

    def __init__(
        self,
        feature_layers: Sequence[str],
        *,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if not feature_layers:
            raise ValueError("feature_layers must contain at least one layer name.")

        self.feature_layers = tuple(feature_layers)
        self.eps = eps
        self.reduction = reduction
        self.log_variances = nn.ParameterDict(
            {name.replace(".", "__"): nn.Parameter(torch.zeros(1)) for name in feature_layers}
        )

    def _log_variance(self, name: str) -> Tensor:
        return self.log_variances[name.replace(".", "__")]

    def forward(self, student_features: FeatureMap, oracle_features: FeatureMap) -> Tensor:
        losses: list[Tensor] = []
        for name in self.feature_layers:
            student = student_features[name]
            oracle = oracle_features[name].detach()
            if student.shape != oracle.shape:
                raise ValueError(
                    f"Feature shape mismatch at {name!r}: "
                    f"student {tuple(student.shape)} vs oracle {tuple(oracle.shape)}"
                )

            log_var = self._log_variance(name).to(dtype=student.dtype, device=student.device)
            inv_var = torch.exp(-log_var).clamp_min(self.eps)
            squared_error = (student - oracle).pow(2)
            losses.append((log_var + squared_error * inv_var).mean())

        stacked = torch.stack(losses)
        if self.reduction == "sum":
            return stacked.sum()
        return stacked.mean()


class MIROTrainer(nn.Module):
    """Wrap a classifier with MIRO loss computation.

    Parameters
    ----------
    student:
        Trainable model used for prediction.
    oracle:
        Frozen pre-trained model with the same selected feature shapes as the
        student. It can be a deep copy of the initialization checkpoint.
    feature_layers:
        Names from ``model.named_modules()`` to regularize, such as
        ``("layer1", "layer2", "layer3", "layer4")`` for ResNet-style models.
    lambda_miro:
        Weight applied to the MIRO regularizer.
    task_loss:
        Supervised loss for the source-domain task. Defaults to cross entropy.
    """

    def __init__(
        self,
        student: nn.Module,
        oracle: nn.Module,
        feature_layers: Sequence[str],
        *,
        lambda_miro: float = 0.01,
        task_loss: LossFn | None = None,
    ) -> None:
        super().__init__()
        self.student = HookedFeatureModel(student, feature_layers)
        self.oracle = HookedFeatureModel(freeze_module(oracle), feature_layers)
        self.regularizer = MIROFeatureRegularizer(feature_layers)
        self.lambda_miro = lambda_miro
        self.task_loss = task_loss or F.cross_entropy

    def forward(self, inputs: Tensor) -> Tensor:
        logits, _features = self.student(inputs)
        return logits

    def train(self, mode: bool = True) -> "MIROTrainer":
        super().train(mode)
        self.oracle.eval()
        return self

    def training_loss(self, inputs: Tensor, targets: Tensor) -> MIROOutput:
        logits, student_features = self.student(inputs)
        self.oracle.eval()
        with torch.no_grad():
            _oracle_logits, oracle_features = self.oracle(inputs)

        task_loss = self.task_loss(logits, targets)
        miro_loss = self.regularizer(student_features, oracle_features)
        total_loss = task_loss + self.lambda_miro * miro_loss

        return MIROOutput(
            total_loss=total_loss,
            task_loss=task_loss.detach(),
            miro_loss=miro_loss.detach(),
            logits=logits,
        )

    def close(self) -> None:
        self.student.close()
        self.oracle.close()


def miro_step(
    trainer: MIROTrainer,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
) -> MIROOutput:
    """Run one optimization step for a MIRO-wrapped model."""

    trainer.train()
    optimizer.zero_grad(set_to_none=True)
    output = trainer.training_loss(inputs, targets)
    output.total_loss.backward()
    optimizer.step()
    return output


def named_modules_containing(model: nn.Module, tokens: Iterable[str]) -> list[str]:
    """Convenience helper for discovering feature layer names."""

    lowered = tuple(token.lower() for token in tokens)
    return [
        name
        for name, _module in model.named_modules()
        if name and any(token in name.lower() for token in lowered)
    ]
