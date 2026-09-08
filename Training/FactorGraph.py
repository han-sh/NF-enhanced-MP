"""Factor graph data structures."""

import torch

from Distribution import Gaussian
from Factor import MeasurementFactor
from Transform import State


class FactorGraph:
    """Construct a factor graph."""

    def __init__(self):
        self.var_node = {}
        self.factor_node = {}

    def add_var_node(self, var_name, belief, tangent_point, time_step, robot_ID, BP_MF):
        """Add a variable node."""

        if var_name in self.var_node:
            raise ValueError(f"Variable '{var_name}' already exists in FactorGraph!")

        new_node = VariableNode(var_name, belief, tangent_point, time_step, robot_ID, BP_MF)
        self.var_node[var_name] = new_node

    def add_factor_node(self, factor_name, adj_var_name, factor_param, NN):
        """Add a factor node."""

        if factor_name in self.factor_node:
            raise ValueError(f"Factor '{factor_name}' already exists in FactorGraph!")

        adj_var = {}
        for name in adj_var_name:
            if name not in self.var_node:
                raise KeyError(f"No adjoint variable '{name}'")
            adj_var[name] = self.var_node[name]

        new_factor = FactorNode(factor_name, adj_var, factor_param, NN)
        self.factor_node[factor_name] = new_factor

        for name in adj_var_name:
            var = self.var_node[name]
            var.adj_factor[factor_name] = new_factor
            var.message[factor_name] = None
            new_factor.message[var.var_name] = None

    def delete_var_node(self, var_name):
        """Delete a variable and every adjacent factor."""

        if var_name not in self.var_node:
            raise KeyError(f"Variable '{var_name}' to be deleted does not exist!")

        var = self.var_node[var_name]

        for adj_factor in list(var.adj_factor.values()):
            self.delete_factor_node(adj_factor.factor_name)

        del self.var_node[var_name]

    def delete_factor_node(self, factor_name):
        """Delete a factor and unlink it from all adjacent variables."""

        if factor_name not in self.factor_node:
            raise KeyError(f"Factor '{factor_name}' to be deleted does not exist!")

        factor = self.factor_node[factor_name]

        for var in list(factor.adj_var.values()):
            del var.adj_factor[factor_name]
            del var.message[factor_name]

        del self.factor_node[factor_name]

    def detach_all_belief(self):
        """Detach graph."""

        for var in self.var_node.values():
            var.detach_belief()

        for factor in self.factor_node.values():
            if isinstance(factor.factor_param, MeasurementFactor):
                factor.detach_belief()


class FactorNode:
    """Factor node."""

    def __init__(self, name, adj_var, factor_param, NN):
        self.factor_name = name

        self.adj_var = adj_var
        self.message = {}

        self.belief = None

        self.factor_param = factor_param

        self.contain_BP_var = False

        if isinstance(factor_param, MeasurementFactor):
            self.initial_belief()
            orientation_list = []
            position_list = []
            state_space = "R3"
            for var in self.adj_var.values():
                if var.BP_MF == "BP":
                    self.contain_BP_var = True
                    if var.tangent_point is not None:
                        state_space = "Lie group"
                        orientation_list.append(var.tangent_point.orientation)
                        position_list.append(var.tangent_point.position)

            if state_space == "Lie group":
                orientation = torch.stack(orientation_list, dim=0)
                position = torch.stack(position_list, dim=0)
                self.tangent_point = State(orientation, position)
            else:
                self.tangent_point = None

            if factor_param.sensor_name in NN.keys():
                self.NN_model = NN[factor_param.sensor_name]
            else:
                self.NN_model = None

    def initial_belief(self):
        """Initialize a measurement factor belief according to connected Gaussian variables."""

        if isinstance(self.factor_param, MeasurementFactor):
            dim = 0
            batch_size = None

            for var in self.adj_var.values():
                if isinstance(var.belief, Gaussian):
                    if batch_size is None:
                        batch_size = var.belief.mean.shape[0]
                        ref_device = var.belief.mean.device
                    dim += var.belief.mean.shape[1]

            mean = torch.zeros(
                (batch_size, dim, 1), dtype=torch.float32, device=ref_device
            )
            cov = torch.zeros(
                (batch_size, dim, dim), dtype=torch.float32, device=ref_device
            )

            count = 0
            for var in self.adj_var.values():
                if isinstance(var.belief, Gaussian):
                    dimi = var.belief.mean.shape[1]
                    mean[:, count : count + dimi, :] = var.belief.mean
                    cov[:, count : count + dimi, count : count + dimi] = var.belief.cov
                    count += dimi

            self.belief = Gaussian()
            self.belief.set_by_Mean_Cov(mean, cov)
        else:
            print("Factor type error when initializing factor beliefs!")

    def detach_belief(self):
        """Detach this factor belief and tangent point from autograd history."""

        natural_param_detach = self.belief.natural_param.detach().clone()
        self.belief.set_by_natural_param(natural_param_detach)

        if isinstance(self.tangent_point, State):
            ori = self.tangent_point.orientation.detach().clone()
            pos = self.tangent_point.position.detach().clone()
            self.tangent_point = State(ori, pos)


class VariableNode:
    """Variable node."""

    def __init__(self, name, belief, tangent_point, time_step, robot_ID, BP_MF):
        self.var_name = name

        self.adj_factor = {}

        self.belief = belief

        self.message = {}

        self.time_step = time_step
        self.robot_ID = robot_ID

        self.BP_MF = BP_MF

        self.tangent_point = tangent_point

    def detach_belief(self):
        """Detach this variable belief and tangent point from autograd history."""

        natural_param_detach = self.belief.natural_param.detach().clone()
        self.belief.set_by_natural_param(natural_param_detach)

        if isinstance(self.tangent_point, State):
            ori = self.tangent_point.orientation.detach().clone()
            pos = self.tangent_point.position.detach().clone()
            self.tangent_point = State(ori, pos)
