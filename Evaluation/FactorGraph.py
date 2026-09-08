"""Factor graph data structures."""

import numpy as np

from Distribution import Gaussian
from Factor import MeasurementFactor
from Transform import State


class FactorGraph:
    """Construct a factor graph."""

    def __init__(self):
        self.var_node = {}
        self.factor_node = {}

        self.extern_var_node = {}
        self.extern_factor_node = {}
        self.extern_var_node_fg = {}
        self.extern_factor_node_fg = {}

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

    def add_cross_factor_node(self, factor_name, self_adj_var_name, extern_adj_var_name, extern_fg, factor_param, NN):
        """Add a factor that also connects to variables owned by other robots."""

        if factor_name in self.factor_node:
            raise ValueError(f"Factor '{factor_name}' already exists in FactorGraph!")

        adj_var = {}
        for name in self_adj_var_name:
            if name not in self.var_node:
                raise KeyError(f"No adjoint variable '{name}'")
            adj_var[name] = self.var_node[name]

        for idx, name in enumerate(extern_adj_var_name):
            if name not in extern_fg[idx].var_node:
                raise KeyError(f"No adjoint variable '{name}'")
            adj_var[name] = extern_fg[idx].var_node[name]
            self.extern_var_node[(factor_name, name)] = adj_var[name]
            self.extern_var_node_fg[(factor_name, name)] = extern_fg[idx]

        new_factor = FactorNode(factor_name, adj_var, factor_param, NN)
        self.factor_node[factor_name] = new_factor

        for name in self_adj_var_name:
            var = self.var_node[name]
            var.adj_factor[factor_name] = new_factor
            var.message[factor_name] = None
            new_factor.message[var.var_name] = None
        
        for idx, name in enumerate(extern_adj_var_name):
            var = extern_fg[idx].var_node[name]
            var.adj_factor[factor_name] = new_factor
            var.message[factor_name] = None
            new_factor.message[var.var_name] = None
            extern_fg[idx].extern_factor_node[(name,factor_name)] = new_factor
            extern_fg[idx].extern_factor_node_fg[(name,factor_name)] = self

    def delete_var_node(self, var_name):
        """Delete a variable and every adjacent factor."""

        if var_name not in self.var_node:
            raise KeyError(f"Variable '{var_name}' to be deleted does not exist!")
        
        var = self.var_node[var_name]

        for adj_factor in list(var.adj_factor.values()):
            if (var_name, adj_factor.factor_name) in self.extern_factor_node:
                extern_fg = self.extern_factor_node_fg[(var_name,adj_factor.factor_name)]
                extern_fg.delete_factor_node(adj_factor.factor_name)
            else:
                self.delete_factor_node(adj_factor.factor_name)

        del self.var_node[var_name]

    def delete_factor_node(self, factor_name):
        """Delete a factor and unlink it from all adjacent variables."""

        if factor_name not in self.factor_node:
            raise KeyError(f"Factor '{factor_name}' to be deleted does not exist!")
        
        factor = self.factor_node[factor_name]
        
        for var in list(factor.adj_var.values()):
            if (factor_name,var.var_name) in self.extern_var_node:
                extern_fg = self.extern_var_node_fg[(factor_name,var.var_name)]
                del extern_fg.extern_factor_node[(var.var_name,factor_name)]
                del extern_fg.extern_factor_node_fg[(var.var_name,factor_name)]

                del self.extern_var_node[(factor_name,var.var_name)]
                del self.extern_var_node_fg[(factor_name,var.var_name)]
            del var.adj_factor[factor_name]
            del var.message[factor_name]

        del self.factor_node[factor_name]


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
                    if var.tangent_point != None:
                        state_space = "Lie group"
                        orientation_list.append(var.tangent_point.orientation)
                        position_list.append(var.tangent_point.position)

            if state_space == "Lie group":
                orientation = np.stack(orientation_list, axis=0)
                position = np.stack(position_list, axis=0)
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

            for var in self.adj_var.values():
                if isinstance(var.belief, Gaussian) and var.BP_MF=="BP":
                    dim += len(var.belief.mean)

            mean = np.zeros((dim, 1))
            cov = np.zeros((dim, dim))

            count = 0
            for var in self.adj_var.values():
                if isinstance(var.belief, Gaussian) and var.BP_MF=="BP":
                    dimi = len(var.belief.mean)
                    mean[count:count+dimi, 0] = var.belief.mean.flatten()
                    cov[count:count+dimi, count:count+dimi] = var.belief.cov
                    count += dimi

            self.belief = Gaussian()
            self.belief.set_by_Mean_Cov(mean, cov)
        else:
            print("Factor type error when initializing factor beliefs!")

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
