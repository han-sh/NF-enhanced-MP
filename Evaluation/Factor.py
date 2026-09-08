"""Factor parameter containers used to construct the factor graph."""

import copy

import numpy as np


class MeasurementFactor:
    """Measurement/state transition factor."""

    def __init__(self, sensor_name, meas, model, cov, ext_param, if_Taylor_Expansion):
        self.meas = copy.deepcopy(meas)
        self.inv_cov = np.linalg.inv(cov)
        self.inv_cov_init = np.linalg.inv(cov)
        self.ext_param = ext_param
        self.sensor_name = sensor_name
        self.model_fcn = model.model_fcn
        self.Jaco_fcn = model.Jaco_fcn
        self.if_Taylor_Expansion = if_Taylor_Expansion

    def model(self, x, meas, ext_param=None):
        """Evaluate the configured residual model."""

        return self.model_fcn(x, meas, ext_param)

    def Jaco(self, x, meas, ext_param=None):
        """Evaluate the configured residual Jacobian."""

        return self.Jaco_fcn(x, meas, ext_param)


class EF_PriorFactor:
    """Exponential family prior factor."""

    def __init__(self, prior_natural_param):
        self.prior_natural_param = prior_natural_param.copy()


class EF_ConnectFactor:
    """Connecting Factor."""

    pass
