from typing import Optional

import random
import torch
from torch import Tensor
from torch import nn


class GradientNormalizeIn(nn.Module):
    def __init__(self, epsilon: float = 0.01) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Tensor of dimension (batch_size * 2, num_features, input_length) that
            contains two copies of each sequence, arranged as (0 1 2 3 ...0 1 2 3 ..),
            where the two copies have a random difference added (in both directions,
            i.e. added and subtracted).
        """
        (b, f, l) = x.shape
        r = 0.5 * self.epsilon * torch.randn((b, f, l), dtype=x.dtype, device=x.device, requires_grad=False)
        ans = torch.cat((x + r, x - r))
        return ans


class GradientNormalizeOut(nn.Module):

    def __init__(self, epsilon: float = 0.01) -> None:
        '''Note: you are supposed to give this the same epsilon as a GradientNormalizeIn
        module that was earlier in the network.'''
        super().__init__()
        # actually epsilon is ignored right now.
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        '''
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size * 2, num_features, input_length).

        Returns:
            Tensor: Tensor of dimension (batch_size, num_features, input_length) where
            we have averaged over the two copies created by the GradientNormalizeIn
            module.   We do something else which is a no-op that affects the
            back-propagated gradients: we make it as if we were dividing by the
            rms of the difference between the two perturbed copies we created
            in class GradientNormalizeIn, but we actually cancel that out
            by multiplying by a constant so the scaling factor is effectively 1.0.
            In typical situations, e.g. when weight decay is in place, the network
            "wants" to increase the scale of the output layer, so if this is
            done at the output layer it will have the effect of trying to
            shrink the difference between the perturbed copies, which will
            act like a penalty on the Jacobian of the entire network.
        '''
        (b2, f, l) = x.shape
        assert b2 % 2 == 0
        b = b2 // 2
        x1 = x[0:b]
        x2 = x[b:2*b]

        # sumsq_diff will be for each sequence.
        sumsq_diff = ((x1 - x2) ** 2).sum(dim=(1,2))
        with torch.no_grad():
            # remove outliers e.g. where LSTM went to different paths.
            limit = 2.0 * torch.median(sumsq_diff)
        indexes = torch.where(sumsq_diff < limit)
        rms = sumsq_diff[indexes].sum().sqrt()
        inv_rms = 1.0 / rms
        rms_nograd = rms.detach()
        avg_x = 0.5 * (x1 + x2)
        return avg_x * inv_rms * rms_nograd
