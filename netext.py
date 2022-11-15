import torch
import torch.nn as nn
from torch import Tensor
from compressai.ops import LowerBound
from typing import Any, List, Optional, Tuple, Union
from compressai.entropy_models import GaussianConditional
from compressai.models import ScaleHyperprior
from compressai.models.utils import conv, deconv

import torch
import torch.nn as nn

from torch import Tensor


def upperbound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.min(x, bound)

def upperbound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x <= bound) | (grad_output > 0)
    return pass_through_if * grad_output, None

class UpperBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return upperbound_fwd(x, bound)
    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return upperbound_bwd(x, bound, grad_output)

class UpperBound(nn.Module):
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))
    def forward(self, x):
        return UpperBoundFunction.apply(x, self.bound)

class GaussianConditionalExt(GaussianConditional):
    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]]
    ):
        super().__init__(scale_table)
        self.lower_bound_deltas = LowerBound(0.5)
        self.upper_bound_deltas = UpperBound(1.5)

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, deltas: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        half = 0.5 * deltas
        dist = torch.distributions.normal.Normal(0, scales)
        likelihood = dist.cdf(values + half) - dist.cdf(values - half)
        return likelihood

    def quantize(
        self, inputs: Tensor, mode: str, deltas: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise * deltas
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs/deltas) * deltas

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        deltas: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        scales = self.lower_bound_scale(scales)
        deltas = self.lower_bound_deltas(deltas)
        deltas = self.upper_bound_deltas(deltas)
        outputs = self.quantize(inputs, "noise" if training else "dequantize", deltas, means)
        likelihood = self._likelihood(outputs, scales, deltas, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood


class ScaleHyperpriorExt(ScaleHyperprior):
    def __init__(self, N=128, M=192, **kwargs):
        super().__init__(N, M, **kwargs)

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.gaussian_conditional = GaussianConditionalExt(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, deltas_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, deltas_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

if __name__ == "__main__":
    ScaleHyperpriorExt()
    pass