"""Exponential-family distributions used by the message passing solver."""

import numpy as np
import scipy


class Gaussian:
    """Multivariate Gaussian."""

    def __init__(self):
        self.mean = None
        self.cov = None
        self.eta = None
        self.Lambda = None
        self.natural_param = None

    def set_by_Mean_Cov(self, mean, cov):
        """Set ``(d, 1)`` mean and ``(d, d)`` covariance matrix."""

        self.mean = mean.copy()
        self.cov = cov.copy()
        self.Lambda = np.linalg.inv(self.cov)
        self.eta = self.Lambda @ self.mean
        self.set_natural_param()

    def set_by_Canonical(self, eta, Lambda):
        """Set canonical vector ``eta`` and precision matrix ``Lambda``."""

        self.eta = eta.copy()
        self.Lambda = Lambda.copy()
        self.cov = np.linalg.inv(self.Lambda)
        self.mean = self.cov @ self.eta
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        """Set the stacked natural parameter ``[eta; -0.5 vec(Lambda)]``."""

        self.natural_param = natural_param.copy()
        n = len(self.natural_param)
        d = int(((-1 + (1 + 4 * n) ** 0.5) / 2))

        self.eta = self.natural_param[:d]                  # (d, 1)
        self.Lambda = -2 * self.natural_param[d:].reshape(d, d)
        self.cov = np.linalg.inv(self.Lambda)              # (d, d)
        self.mean = self.cov @ self.eta          # (d, 1)

    def set_natural_param(self):
        """Synchronize the stacked natural parameter."""

        d = len(self.mean)
        self.natural_param = np.zeros((d + d*d, 1))
        self.natural_param[:d, 0] = self.eta.reshape(d, 1).flatten()
        self.natural_param[d:, 0] = (-0.5 * self.Lambda.reshape(-1, 1)).flatten()


class invWishart:
    """Inverse-Wishart distribution."""

    def __init__(self):
        self.t = None
        self.T = None
        self.natural_param = None

    def set(self, t, T):
        """Set degrees-of-freedom parameter ``t`` and scale matrix ``T``."""

        self.t = t.copy()
        self.T = T.copy()
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        """Recover ``t`` and ``T`` from the stacked natural parameter."""

        self.natural_param = natural_param.copy()
        n = self.natural_param.shape[0]
        dim = int(np.sqrt(n - 1))

        self.t = -2 * self.natural_param[0, 0] - dim - 1
        self.T = -2 * self.natural_param[1:].reshape(dim, dim)

    def set_natural_param(self):
        """Synchronize the natural parameter from ``t`` and ``T``."""

        dim = self.T.shape[0]
        self.natural_param = np.zeros((1 + dim * dim, 1))
        self.natural_param[0, 0] = -0.5 * (self.t + dim + 1)
        self.natural_param[1:, 0] = -0.5 * self.T.reshape(-1, 1).flatten()

    def inv_var_expectation(self):
        """Return the expected precision matrix, ``E[P^{-1}]``."""

        return self.t * np.linalg.inv(self.T)
    

class Gamma:
    """Gamma distribution using shape ``a`` and rate ``b``."""

    def __init__(self):
        self.a = None
        self.b = None
        self.natural_param = None

    def set(self, a, b):
        """Set shape ``a`` and rate ``b``."""

        self.a = a.copy()
        self.b = b.copy()
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        """Recover shape and rate from the natural parameter."""

        self.natural_param = natural_param.copy()

        self.a = self.natural_param[0, 0] + 1
        self.b = -self.natural_param[1, 0]

    def set_natural_param(self):
        """Synchronize the natural parameter from shape and rate."""

        self.natural_param = np.zeros((2, 1))
        self.natural_param[0, 0] = self.a - 1
        self.natural_param[1, 0] = -self.b

    def expectation(self):
        """Return ``E[x]``."""

        return self.a/self.b

    def ln_expectation(self):
        """Return ``E[log(x)]``."""

        return scipy.special.psi(self.a) - np.log(self.b)


class Beta:
    """Beta distribution using shape parameters ``a`` and ``b``."""

    def __init__(self):
        self.a = None
        self.b = None
        self.natural_param = None

    def set(self, a, b):
        """Set the two shape parameters."""

        self.a = a.copy()
        self.b = b.copy()
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        """Recover shape parameters from the natural parameter."""

        self.natural_param = natural_param.copy()

        self.a = self.natural_param[0, 0] + 1
        self.b = self.natural_param[1, 0] + 1

    def set_natural_param(self):
        """Synchronize the natural parameter from the shape parameters."""

        self.natural_param = np.zeros((2, 1))
        self.natural_param[0, 0] = self.a - 1
        self.natural_param[1, 0] = self.b - 1

    def ln_expectation(self):
        """Return ``E[log(x)]``."""

        return scipy.special.psi(self.a) - scipy.special.psi(self.a+self.b)
    
    def ln_1_expectation(self):
        """Return ``E[log(1-x)]``."""

        return scipy.special.psi(self.b) - scipy.special.psi(self.a+self.b)
    

class Bernoulli:
    """Bernoulli distribution parameterized by success probability ``mu``."""

    def __init__(self):
        self.mu = None
        self.natural_param = None

    def set(self, mu):
        """Set the success probability."""

        self.mu = mu.copy()
        self.set_natural_param()

    def set_by_natural_param(self, natural_param):
        """Recover the success probability through the logistic function."""

        self.set(1/(1+np.exp(-natural_param)))

    def set_natural_param(self):
        """Synchronize the log-odds natural parameter."""

        self.natural_param = np.log(self.mu/(1-self.mu)).reshape(1,1)

    def expectation(self):
        """Return ``E[x]``."""

        return self.mu
