import torch
import torch.nn as nn
import torch.nn.functional as F


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, soft: bool = False) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.soft = soft
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        if ctx.soft:
            grad_input = grad_output * F.softsign(input)  # Soft gradient
        else:
            grad_input = grad_output.clamp(min=-1, max=1)  # Hard gradient
        return grad_input


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, soft: bool = False) -> torch.Tensor:
        return STEFunction.apply(x, soft)
