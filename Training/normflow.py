"""Conditional RealNVP proposal used by the neural gradient estimator."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small conditional network used for scale and translation outputs."""

    def __init__(self, dim, cond_dim, use_cond):
        super().__init__()
        self.use_cond = use_cond
        in_dim = dim + cond_dim if use_cond else dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, (in_dim + dim) // 2),
            nn.ReLU(),
            nn.Linear((in_dim + dim) // 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, cond):
        """Evaluate the MLP with the optional conditioning vector."""

        if self.use_cond and cond is not None:
            x = torch.cat([x, cond], dim=-1)
        return self.net(x)


class CouplingLayer(nn.Module):
    """One affine RealNVP coupling layer."""

    def __init__(self, dim, cond_dim, mask, use_cond):
        super().__init__()
        self.register_buffer("mask", mask)
        self.s_net = MLP(dim, cond_dim, use_cond)
        self.t_net = MLP(dim, cond_dim, use_cond)

    def forward(self, x, cond, inverse=False):
        """Apply the forward or inverse affine coupling transform."""

        # Split the input using the fixed alternating mask.
        x1 = x * self.mask
        x2 = x * (1 - self.mask)

        st_in = x1
        s = self.s_net(st_in, cond) * (1 - self.mask)
        t = self.t_net(st_in, cond) * (1 - self.mask)

        if not inverse:
            y2 = x2 * torch.exp(s) + t
            y = x1 + y2
            logdet = s.sum(dim=-1)
            return y, logdet
        else:
            y2 = (x2 - t) * torch.exp(-s)
            y = x1 + y2
            logdet = (-s).sum(dim=-1)
            return y, logdet


class RealNVP(nn.Module):
    """Conditional normalizing flow and its importance weights."""

    def __init__(self, dim, cond_dim, n_layer, device):
        super().__init__()
        self.device = device
        self.dim = dim
        masks = []
        for i in range(n_layer):
            m = torch.zeros(dim, device=device)
            m[i % 2 :: 2] = 1
            masks.append(m)
        self.register_buffer("masks", torch.stack(masks))
        self.layers = nn.ModuleList(
            [
                CouplingLayer(dim, cond_dim, self.masks[i], use_cond=True)
                for i in range(n_layer)
            ]
        )

    def fwd(self, z, cond):
        """Map a standard-normal sample to proposal samples."""

        logdet_total = torch.zeros(z.shape[0], device=self.device)
        x = z
        for layer in self.layers:
            x, logdet = layer(x, cond, inverse=False)
            logdet_total += logdet
        return x, logdet_total

    def forward(self, z, cond):
        """Return proposal samples and log importance weights."""

        eps, logdet = self.fwd(z, cond)
        logw = logdet - 0.5 * (torch.sum(eps**2, dim=1) - torch.sum(z**2, dim=1))

        return eps, logw
