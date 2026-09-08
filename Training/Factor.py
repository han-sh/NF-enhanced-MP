"""Factor parameter containers shared by the training graph and solver."""

import copy

import torch

from Transform import State


class MeasurementFactor:
    """Measurement/state transition factor."""

    def __init__(self, sensor_name, meas, model, cov, ext_param, if_linear):
        if isinstance(meas, State):
            self.meas = copy.deepcopy(meas)
        else:
            self.meas = meas.detach().clone()
        self.inv_cov = torch.linalg.inv(cov.detach())
        self.inv_cov_init = torch.linalg.inv(cov.detach())
        self.ext_param = ext_param
        self.sensor_name = sensor_name
        self.model_fcn = model.model_fcn
        self.Jaco_fcn = model.Jaco_fcn
        self.if_linear = if_linear

    def model(self, x, meas, ext_param=None):
        """Evaluate the configured residual model."""

        return self.model_fcn(x, meas, ext_param)

    def Jaco(self, x, meas, ext_param=None):
        """Evaluate the configured residual Jacobian."""

        return self.Jaco_fcn(x, meas, ext_param)


class EF_PriorFactor:
    """Exponential family prior factor."""

    def __init__(self, prior_natural_param):
        self.prior_natural_param = prior_natural_param.clone()


class EF_ConnectFactor:
    """Connecting Factor."""

    pass
