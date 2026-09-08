"""Batched exponential-family distributions."""

import torch


class Gaussian:
    """Multivariate Gaussian."""

    def __init__(self):
        self.mean = None
        self.cov = None
        self.eta = None
        self.Lambda = None
        self.natural_param = None

    def set_by_Mean_Cov(self, mean, cov):
        """Set from mean ``(B,d,1)`` and covariance ``(B,d,d)``."""

        self.mean = mean.clone()
        self.cov = cov.clone()
        self.Lambda = torch.linalg.inv(self.cov)
        self.eta = self.Lambda @ self.mean
        self.set_natural_param()

    def set_by_Canonical(self, eta, Lambda):
        """Set from information vector and precision matrix."""

        self.eta = eta.clone()
        self.Lambda = Lambda.clone()
        self.cov = torch.linalg.inv(self.Lambda)
        self.mean = self.cov @ self.eta
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        """Set from ``[eta; vec(-0.5 Lambda)]``."""

        self.natural_param = natural_param.clone()
        B, n, _ = self.natural_param.shape
        d = int(((-1 + (1 + 4 * n) ** 0.5) / 2))

        self.eta = self.natural_param[:, :d]  # (B, d, 1)
        self.Lambda = -2 * self.natural_param[:, d:].reshape(B, d, d)
        self.cov = torch.linalg.inv(self.Lambda)  # (B, d, d)
        self.mean = torch.matmul(self.cov, self.eta)  # (B, d, 1)

    def set_natural_param(self):
        """Pack the current canonical parameters into one tensor."""

        B, d, _ = self.mean.shape
        refdevice = self.mean.device
        self.natural_param = torch.zeros(
            (B, d + d * d, 1), dtype=torch.float32, device=refdevice
        )
        self.natural_param[:, :d, 0] = self.eta.reshape(B, d)
        self.natural_param[:, d:, 0] = -0.5 * self.Lambda.reshape(B, -1)


class invWishart:
    """Inverse-Wishart distribution."""

    def __init__(self):
        self.t = None
        self.T = None
        self.natural_param = None

    def set(self, t, T):
        self.t = t.clone()
        self.T = T.clone()
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        refdevice = self.t.device

        self.natural_param = natural_param.clone()
        n = self.natural_param.shape[1]
        dim = int(
            torch.sqrt(torch.tensor(n - 1, dtype=torch.float32, device=refdevice))
        )

        self.t = (-2 * self.natural_param[:, 0, 0] - dim - 1).reshape(-1, 1, 1)
        self.T = -2 * self.natural_param[:, 1:].reshape(-1, dim, dim)

    def set_natural_param(self):
        B, _, _ = self.t.shape
        refdevice = self.t.device
        dim = self.T.shape[1]
        self.natural_param = torch.zeros(
            (B, 1 + dim * dim, 1), dtype=torch.float32, device=refdevice
        )
        self.natural_param[:, 0, 0] = -0.5 * (self.t + dim + 1).flatten()
        self.natural_param[:, 1:, 0] = -0.5 * self.T.reshape(B, -1, 1).squeeze(-1)

    def inv_var_expectation(self):
        return self.t * torch.linalg.inv(self.T)


class Gamma:
    """Gamma distribution using shape ``a`` and rate ``b``."""

    def __init__(self):
        self.a = None
        self.b = None
        self.natural_param = None

    def set(self, a, b):
        self.a = a.clone()
        self.b = b.clone()
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        self.natural_param = natural_param.clone()

        self.a = self.natural_param[:, 0, 0].reshape(-1, 1, 1) + 1
        self.b = -self.natural_param[:, 1, 0].reshape(-1, 1, 1)

    def set_natural_param(self):
        B = self.a.shape[0]
        refdevice = self.a.device
        self.natural_param = torch.zeros(
            (B, 2, 1), dtype=torch.float32, device=refdevice
        )
        self.natural_param[:, 0, 0] = self.a.flatten() - 1
        self.natural_param[:, 1, 0] = -self.b.flatten()

    def expectation(self):
        return self.a / self.b

    def ln_expectation(self):
        epsilon = 1e-8
        return torch.digamma(self.a + epsilon) - torch.log(self.b + epsilon)


class Beta:
    """Beta distribution using shape parameters ``a`` and ``b``."""

    def __init__(self):
        self.a = None
        self.b = None
        self.natural_param = None

    def set(self, a, b):
        self.a = a.clone()
        self.b = b.clone()
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        self.natural_param = natural_param.clone()

        self.a = self.natural_param[:, 0, 0].reshape(-1, 1, 1) + 1
        self.b = self.natural_param[:, 1, 0].reshape(-1, 1, 1) + 1

    def set_natural_param(self):
        B = self.a.shape[0]
        refdevice = self.a.device
        self.natural_param = torch.zeros(
            (B, 2, 1), dtype=torch.float32, device=refdevice
        )
        self.natural_param[:, 0, 0] = self.a.flatten() - 1
        self.natural_param[:, 1, 0] = self.b.flatten() - 1

    def ln_expectation(self):
        epsilon = 1e-8
        return torch.digamma(self.a + epsilon) - torch.digamma(
            self.a + self.b + epsilon
        )

    def ln_1_expectation(self):
        epsilon = 1e-8
        return torch.digamma(self.b + epsilon) - torch.digamma(
            self.a + self.b + epsilon
        )


class Bernoulli:
    """Bernoulli distribution parameterized by success probability ``mu``."""

    def __init__(self):
        self.mu = None
        self.natural_param = None

    def set(self, mu):
        self.mu = mu.clone()
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        self.set(torch.special.expit(natural_param))

    def set_natural_param(self):
        B = self.mu.shape[0]
        self.natural_param = torch.log(self.mu / (1 - self.mu)).reshape(B, 1, 1)

    def expectation(self):
        return self.mu
