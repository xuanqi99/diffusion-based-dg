"""PyTorch implementation of Empirical Risk Minimization for DG.

ERM pools labeled examples from all available source domains and minimizes the
standard supervised task loss. It is the canonical domain generalization
baseline and the final recognition-training stage used by many augmentation
methods, including CDGA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


LossFn = Callable[[Tensor, Tensor], Tensor]
Minibatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class ERMOutput:
    """Values produced by an ERM loss or optimization step."""

    loss: Tensor
    logits: Tensor
    targets: Tensor
    accuracy: Tensor


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


def combine_source_minibatches(minibatches: Sequence[Minibatch]) -> Minibatch:
    """Pool minibatches from multiple source domains for standard ERM."""

    if not minibatches:
        raise ValueError("minibatches must contain at least one source domain.")

    inputs = [batch[0] for batch in minibatches]
    targets = [batch[1] for batch in minibatches]
    if any(x.ndim == 0 for x in inputs) or any(y.ndim == 0 for y in targets):
        raise ValueError("Each source minibatch must include a batch dimension.")

    reference_shape = inputs[0].shape[1:]
    mismatched = [tuple(x.shape) for x in inputs if x.shape[1:] != reference_shape]
    if mismatched:
        raise ValueError(
            "All source-domain inputs must share the same non-batch shape; "
            f"expected {tuple(reference_shape)}, received {mismatched}."
        )

    return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)


class ERMTrainer(nn.Module):
    """Wrap a prediction model with ERM loss computation.

    Parameters
    ----------
    model:
        Trainable classifier whose output is a logits tensor, a tuple/list with
        logits first, or a mapping containing a ``logits``-like key.
    task_loss:
        Supervised source-domain loss. Defaults to cross entropy.
    """

    def __init__(self, model: nn.Module, *, task_loss: LossFn | None = None) -> None:
        super().__init__()
        self.model = model
        self.task_loss = task_loss or F.cross_entropy

    def forward(self, inputs: Tensor) -> Tensor:
        return _extract_logits(self.model(inputs))

    def training_loss(self, inputs: Tensor, targets: Tensor) -> ERMOutput:
        logits = self(inputs)
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                "Logits and targets must have the same batch size; "
                f"received {logits.shape[0]} and {targets.shape[0]}."
            )

        loss = self.task_loss(logits, targets)
        accuracy = logits.detach().argmax(dim=1).eq(targets.detach()).float().mean()
        return ERMOutput(
            loss=loss,
            logits=logits,
            targets=targets,
            accuracy=accuracy,
        )

    def training_loss_from_domains(self, minibatches: Sequence[Minibatch]) -> ERMOutput:
        """Pool source-domain minibatches and compute the ERM objective."""

        inputs, targets = combine_source_minibatches(minibatches)
        return self.training_loss(inputs, targets)


def erm_step(
    trainer: ERMTrainer,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor | None = None,
    targets: Tensor | None = None,
    *,
    minibatches: Sequence[Minibatch] | None = None,
    max_grad_norm: float | None = None,
) -> ERMOutput:
    """Run one ERM optimization step on one batch or multiple source domains."""

    uses_batch = inputs is not None or targets is not None
    if uses_batch == (minibatches is not None):
        raise ValueError(
            "Provide either inputs and targets or minibatches, but not both."
        )
    if minibatches is None and (inputs is None or targets is None):
        raise ValueError("Both inputs and targets are required for a single batch.")

    trainer.train()
    optimizer.zero_grad(set_to_none=True)
    if minibatches is not None:
        output = trainer.training_loss_from_domains(minibatches)
    else:
        output = trainer.training_loss(inputs, targets)

    output.loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(trainer.parameters(), max_grad_norm)
    optimizer.step()
    return output


@torch.no_grad()
def evaluate_erm(
    trainer: ERMTrainer,
    batches: Iterable[Minibatch],
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Return sample-weighted mean loss and accuracy for a data loader."""

    trainer.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for inputs, targets in batches:
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)

        logits = trainer(inputs)
        batch_size = targets.shape[0]
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("batches did not yield any examples.")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
    }
