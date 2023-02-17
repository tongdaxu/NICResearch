from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from compressai.ops import LowerBound
from compressai.entropy_models import GaussianConditional
from compressai.models import ScaleHyperprior, MeanScaleHyperprior
from compressai.models.utils import conv, deconv
from compressai.layers import GDN, MaskedConv2d


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
        self.lower_bound_deltas = LowerBound(0.8)
        self.upper_bound_deltas = UpperBound(1.25)

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, deltas: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        half = 0.5 * deltas
        dist = torch.distributions.normal.Normal(0, scales)
        # likelihood = dist.cdf(values + half) - dist.cdf(values - half)
        likelihood = dist.cdf(values/deltas + 0.5) - dist.cdf(values/deltas - 0.5)
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

class ScaleHyperpriorWithY(ScaleHyperprior):
    def __init__(self, N=128, M=192, **kwargs):
        super().__init__(N, M, **kwargs)
    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        # note that we can not use y-mu+mu here!
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "y_hat": y_hat.detach(), # get the aun bitstream
            "y_hat_i": torch.round(y_hat.detach()), # get the bitstream
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

class MeanScaleHyperpriorWithY(MeanScaleHyperprior):
    def __init__(self, N=128, M=192, **kwargs):
        super().__init__(N, M, **kwargs)
    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat_i = torch.round(y - means_hat) + means_hat
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "y_hat": y_hat.detach(), # get the aun bitstream
            "y_hat_i": y_hat_i.detach(), # get the bitstream
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

class ConMeanScaleHyperpriorWithY(MeanScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_s1 = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
        )
        self.g_s2 = nn.Sequential(
            deconv(N+28, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
    def forward(self, x, label):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat_i = torch.round(y - means_hat) + means_hat
        x_mid = self.g_s1(y_hat)
        x_hat = self.g_s2(torch.cat([x_mid,label],dim=1))
        return {
            "x_hat": x_hat,
            "y_hat": y_hat.detach(), # get the aun bitstream
            "y_hat_i": y_hat_i.detach(), # get the bitstream
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

class ScaleHyperpriorYDecoder(nn.Module):
    def __init__(self, N=128, M=192) -> None:
        super(ScaleHyperpriorYDecoder, self).__init__()
        self.g_s = nn.Sequential(
            deconv(2 * M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, y_hat_i):
        eps = torch.randn_like(y_hat_i)
        y_hat_input = torch.cat([y_hat_i, eps],dim=1)
        x_hat = self.g_s(y_hat_input)
        return x_hat
        
class ConScaleHyperpriorYDecoder(nn.Module):
    def __init__(self, N=128, M=192) -> None:
        super(ConScaleHyperpriorYDecoder, self).__init__()
        self.g_s1 = nn.Sequential(
            deconv(M*2, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
        )
        self.g_s2 = nn.Sequential(
            deconv(N+28, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, y_hat_i, label):
        eps = torch.randn_like(y_hat_i)
        y_hat_input = torch.cat([y_hat_i, eps],dim=1)
        x_mid = self.g_s1(y_hat_input)
        x_hat = self.g_s2(torch.cat([x_mid, label],dim=1))
        return x_hat


class Discriminator(nn.Module):
    def __init__(self, image_dims, context_dims, C, spectral_norm=True):
        super(Discriminator, self).__init__()
        
        self.image_dims = image_dims
        self.context_dims = context_dims
        im_channels = self.image_dims[0]
        kernel_dim = 4
        context_C_out = 12
        filters = (64, 128, 256, 512)

        # Upscale encoder output - (C, 16, 16) -> (12, 256, 256)
        self.context_conv = nn.Conv2d(C, context_C_out, kernel_size=3, padding=1, padding_mode='reflect')
        self.context_upsample = nn.Upsample(scale_factor=16, mode='nearest')

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # (C_in + C_in', 256,256) -> (64,128,128), with implicit padding
        # TODO: Check if removing spectral norm in first layer works
        self.conv1 = norm(nn.Conv2d(im_channels + context_C_out, filters[0], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.conv2 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.conv3 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (32,32) -> (16,16)
        self.conv4 = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))

        self.conv_out = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)

    def forward(self, x, y):
        """
        x: Concatenated real/gen images on batch size dim
        y: Quantized latents
        """
        batch_size = x.size()[0]

        # Concatenate upscaled encoder output y as contextual information
        y = self.activation(self.context_conv(y))
        y = self.context_upsample(y)

        x = torch.cat((x,y), dim=1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        
        out = self.conv_out(x).view(-1,1)
        # out = torch.sigmoid(out_logits)
        
        return out


class ConDiscriminator(nn.Module):
    def __init__(self, image_dims, context_dims, C, spectral_norm=True):
        super(ConDiscriminator, self).__init__()
        
        self.image_dims = image_dims
        self.context_dims = context_dims
        im_channels = self.image_dims[0]
        kernel_dim = 4
        context_C_out = 12
        filters = (64, 128, 256, 512)

        self.context_conv = nn.Conv2d(C, context_C_out, kernel_size=3, padding=1, padding_mode='reflect')
        self.context_upsample = nn.Upsample(scale_factor=16, mode='nearest')
        self.label_upsample = nn.Upsample(scale_factor=4, mode='nearest')

        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        self.conv1 = norm(nn.Conv2d(im_channels + context_C_out + 28, filters[0], kernel_dim, **cnn_kwargs))
        self.conv2 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))
        self.conv3 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))
        self.conv4 = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))
        self.conv_out = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)

    def forward(self, x, y, label):
        batch_size = x.size()[0]
        y = self.activation(self.context_conv(y))
        y = self.context_upsample(y)
        label = self.label_upsample(label)
        x = torch.cat([x,y,label], dim=1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        
        out_logits = self.conv_out(x).view(-1,1)
        out = torch.sigmoid(out_logits)
        
        return out

if __name__ == "__main__":
    ScaleHyperpriorExt()
    pass