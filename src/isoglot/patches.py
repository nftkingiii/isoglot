"""Fixes to the Lexsi stack found while building Isoglot, applied at runtime.

SafeTune 0.1.12's LISA wrapper sums per-parameter norms into one value without
moving them to a common device. With the model split across two GPUs
(device_map="auto"), the first cross-device addition raises
"Expected all tensors to be on the same device". The two methods below keep
LISA's math exactly and only move each term to one device before summing.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def _switch_state(self, new_state):
    if new_state == self.status:
        return
    sum_drift = 0.0
    leaving, other = ((self.alignment_weights, self.finetune_weights) if self.status == "alignment"
                      else (self.finetune_weights, self.alignment_weights))
    for name, param in self.model.named_parameters():
        if param.requires_grad:
            leaving[name] = param.data.detach().clone()
            sum_drift += float(torch.norm(leaving[name] - other[name]) ** 2)
    self.status = new_state


@contextmanager
def _apply_proximal_penalty(self, loss: torch.Tensor, scale: float = 1.0):
    modified_loss = loss.clone()
    if self.current_step > self.warmup_steps and self.rho > 0:
        anchor = self.finetune_weights if self.status == "alignment" else self.alignment_weights
        penalty = torch.zeros((), device=loss.device, dtype=torch.float32)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                term = (self.rho / 2) * torch.norm(param - anchor[name]) ** 2
                penalty = penalty + term.to(loss.device, torch.float32)
        modified_loss = modified_loss + penalty * scale
    yield modified_loss


def patch_lisa_multi_device() -> None:
    """Make SafeTune's LISA work on a model split across several GPUs."""
    from safetune.core.optim.lisa import LisaOptimizerWrapper

    LisaOptimizerWrapper.switch_state = _switch_state
    LisaOptimizerWrapper.apply_proximal_penalty = _apply_proximal_penalty
