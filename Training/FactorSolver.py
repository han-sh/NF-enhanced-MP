"""Batched BP-MF solver."""

from collections import defaultdict

import torch

from Distribution import Gaussian, invWishart, Gamma, Beta, Bernoulli
from Factor import MeasurementFactor, EF_PriorFactor, EF_ConnectFactor
from Transform import State, rotm2rotvec, composite_add, composite_minus, composite_rightJaco


class fg_solver:
    """Solve factor graph inference."""

    def __init__(self, fg, nonlinear_approx_mode, state_space, sampling_num):
        """Initialize solver."""

        self.iteration = 0
        self.fg = fg
        self.dim = {}
        self.sampling_num = sampling_num
        self.state_space = state_space
        self.nonlinear_approx_mode = nonlinear_approx_mode
        self.A = {}

        self._initialize_factor_batches()
        var_offsets = self._initialize_variable_beliefs()
        self._initialize_factor_message_offsets()
        self._initialize_variable_message_offsets()

        back_factor_msgs_BP, back_factor_msgs_MF, back_var_msgs = (
            self._build_reverse_message_indices()
        )
        self._build_mapping_matrices(back_var_msgs, var_offsets, back_factor_msgs_BP, back_factor_msgs_MF)

    def _initialize_factor_batches(self):
        """Pack factor beliefs and sensor data into batches."""

        self.factor_offsets = {}
        self.dim["factor"] = 0
        factor_natural_param_list = []
        self.factor_groups = defaultdict(list)

        meas_dict = defaultdict(list)
        meas_state_dict = defaultdict(lambda: defaultdict(list))
        factor_tangent_point_dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        ext_param_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for factor in self.fg.factor_node.values():
            if not factor.contain_BP_var:
                continue

            sensor_name = factor.factor_param.sensor_name
            B = factor.belief.natural_param.shape[0]
            dim = factor.belief.natural_param.shape[1]
            offset = self.dim["factor"]

            self.factor_offsets[factor.factor_name] = (offset, dim)
            self.factor_groups[(sensor_name, dim)].append(
                (
                    factor.factor_name,
                    offset,
                    factor.factor_param.meas,
                    factor.factor_param.ext_param,
                )
            )
            self.dim["factor"] += dim
            factor_natural_param_list.append(factor.belief.natural_param)

            if self.state_space == "Lie_Group":
                for d in range(factor.tangent_point.orientation.shape[0]):
                    factor_tangent_point_dict[sensor_name][d]["orientation"].append(
                        factor.tangent_point.orientation[d].unsqueeze(1)
                    )
                    factor_tangent_point_dict[sensor_name][d]["position"].append(
                        factor.tangent_point.position[d].unsqueeze(1)
                    )

            if isinstance(factor.factor_param.meas, State):
                meas_state_dict[sensor_name]["orientation"].append(
                    factor.factor_param.meas.orientation.unsqueeze(1)
                )
                meas_state_dict[sensor_name]["position"].append(
                    factor.factor_param.meas.position.unsqueeze(1)
                )
            else:
                meas_dict[sensor_name].append(factor.factor_param.meas.unsqueeze(1))

            if factor.factor_param.ext_param is not None:
                for idx, ext_param in enumerate(factor.factor_param.ext_param):
                    for Rt, value in ext_param.items():
                        ext_param_dict[sensor_name][idx][Rt].append(
                            value.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
                        )

        self.factor_natural_param = torch.cat(factor_natural_param_list, dim=1)
        self._stack_sensor_batches(
            factor_tangent_point_dict, meas_state_dict, meas_dict, ext_param_dict
        )

    def _stack_sensor_batches(
        self, factor_tangent_point_dict, meas_state_dict, meas_dict, ext_param_dict
    ):
        """Stack the temporary per-factor lists collected for each sensor."""

        self.factor_tangent_point = defaultdict(lambda: defaultdict(dict))
        self.meas_batch = {}
        self.ext_param_batch = defaultdict(lambda: defaultdict(dict))
        self.iteration_NN = {}

        for sensor_name, _ in self.factor_groups:
            if self.state_space == "Lie_Group":
                for idx in factor_tangent_point_dict[sensor_name]:
                    orientation = torch.cat(
                        factor_tangent_point_dict[sensor_name][idx]["orientation"],
                        dim=1,
                    ).reshape(-1, 3, 3)
                    position = torch.cat(
                        factor_tangent_point_dict[sensor_name][idx]["position"],
                        dim=1,
                    ).reshape(-1, 3, 1)
                    self.factor_tangent_point[sensor_name][idx] = State(
                        orientation, position
                    )

            if sensor_name in meas_state_dict:
                orientation = torch.cat(
                    meas_state_dict[sensor_name]["orientation"], dim=1
                ).reshape(-1, 3, 3)
                position = torch.cat(
                    meas_state_dict[sensor_name]["position"], dim=1
                ).reshape(-1, 3, 1)
                self.meas_batch[sensor_name] = State(orientation, position)

            if sensor_name in meas_dict:
                dimz = meas_dict[sensor_name][0].shape[2]
                self.meas_batch[sensor_name] = torch.cat(
                    meas_dict[sensor_name], dim=1
                ).reshape(-1, dimz, 1)

            if sensor_name in ext_param_dict:
                for idx in ext_param_dict[sensor_name]:
                    for Rt in ext_param_dict[sensor_name][idx]:
                        dim_param = ext_param_dict[sensor_name][idx][Rt][0].shape
                        self.ext_param_batch[sensor_name][idx][Rt] = torch.cat(
                            ext_param_dict[sensor_name][idx][Rt], dim=1
                        ).reshape(-1, dim_param[2], dim_param[3])

            self.iteration_NN[sensor_name] = 0

    def _initialize_variable_beliefs(self):
        """Pack local variable beliefs and return their vector offsets."""

        var_offsets = {}
        self.dim["var"] = 0
        var_natural_param_list = []

        for var in self.fg.var_node.values():
            dim = var.belief.natural_param.shape[1]
            var_offsets[var.var_name] = (self.dim["var"], dim)
            self.dim["var"] += dim
            var_natural_param_list.append(var.belief.natural_param)

        self.var_natural_param = torch.cat(var_natural_param_list, dim=1)
        return var_offsets

    def _initialize_factor_message_offsets(self):
        """Allocate factor-to-variable blocks for BP and MF messages."""

        self.factor_msg_offsets_BP = {}
        self.dim["factor_msg_BP"] = 0
        self.factor_msg_groups_BP = defaultdict(list)

        for factor in self.fg.factor_node.values():
            if not factor.contain_BP_var:
                continue
            for var_name in factor.adj_var:
                if factor.adj_var[var_name].BP_MF != "BP":
                    continue
                dim = factor.adj_var[var_name].belief.natural_param.shape[1]
                offset = self.dim["factor_msg_BP"]
                self.factor_msg_offsets_BP[(factor.factor_name, var_name)] = (
                    offset,
                    dim,
                )
                self.factor_msg_groups_BP[dim].append(
                    (factor.factor_name, var_name, offset)
                )
                self.dim["factor_msg_BP"] += dim

        self.factor_msg_offsets_MF = {}
        self.dim["factor_msg_MF"] = 0

        for factor in self.fg.factor_node.values():
            for var_name in factor.adj_var:
                if factor.adj_var[var_name].BP_MF == "MF":
                    dim = factor.adj_var[var_name].belief.natural_param.shape[1]
                    self.factor_msg_offsets_MF[(factor.factor_name, var_name)] = (
                        self.dim["factor_msg_MF"],
                        dim,
                    )
                    self.dim["factor_msg_MF"] += dim

    def _initialize_variable_message_offsets(self):
        """Allocate and initialize BP variable-to-factor message blocks."""

        self.var_msg_offsets = {}
        self.dim["var_msg"] = 0
        var_msg_list = []

        for var in self.fg.var_node.values():
            if var.BP_MF != "BP":
                continue
            dim = var.belief.natural_param.shape[1]
            for factor_name in var.adj_factor:
                self.var_msg_offsets[(var.var_name, factor_name)] = (
                    self.dim["var_msg"],
                    dim,
                )
                self.dim["var_msg"] += dim
                var_msg_list.append(var.belief.natural_param)

        self.var_msg = torch.cat(var_msg_list, dim=1)

    def _build_reverse_message_indices(self):
        """Index packed edge messages by their destination node."""

        back_factor_msgs_BP = {}
        for (factor_name, var_name), (col, dim) in self.factor_msg_offsets_BP.items():
            back_factor_msgs_BP.setdefault(var_name, []).append((factor_name, col, dim))

        back_factor_msgs_MF = {}
        for (factor_name, var_name), (col, dim) in self.factor_msg_offsets_MF.items():
            back_factor_msgs_MF.setdefault(var_name, []).append((factor_name, col, dim))

        back_var_msgs = {}
        for (var_name, factor_name), (col, dim) in self.var_msg_offsets.items():
            back_var_msgs.setdefault(factor_name, []).append((var_name, col, dim))

        return back_factor_msgs_BP, back_factor_msgs_MF, back_var_msgs

    @staticmethod
    def _expectation_parameter_projection(factor, var, mean_offsets, dim, dima, device):
        """The returned matrix Lbar extracts the selected variable's expectation parameter from the factor's expectation parameter:

            variable expectation parameter = Lbar @ factor expectation parameter

        ----------
        dim: Length of the variable's expectation parameter
        dima: Length of the factor's expectation parameter.
        """

        dim_factor_mean = factor.belief.mean.shape[1]
        dim_var_mean = var.belief.mean.shape[1]
        start = mean_offsets[var.var_name]

        L = torch.zeros((dim_var_mean, dim_factor_mean), dtype=torch.float32, device=device)
        L[:, start : start + dim_var_mean] = torch.eye(dim_var_mean, dtype=torch.float32, device=device)

        Lbar = torch.zeros((dim, dima), dtype=torch.float32, device=device)
        Lbar[:dim_var_mean, :dim_factor_mean] = L
        Lbar[dim_var_mean:, dim_factor_mean:] = torch.kron(L, L)
        return Lbar

    @staticmethod
    def _make_sparse_matrix(rows, cols, vals, shape, device):
        """Create one detached COO mapping matrix on the solver device."""

        indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
        values = torch.tensor(vals, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, shape, device=device).detach()

    def _build_mapping_matrices(
        self,
        back_var_msgs,
        var_offsets,
        back_factor_msgs_BP,
        back_factor_msgs_MF,
    ):

        device = self.var_natural_param.device
        L_factor = torch.zeros((self.dim["factor"], self.dim["var_msg"]), dtype=torch.float32, device=device)
        L_1_factor_msg = torch.zeros((self.dim["factor_msg_BP"], self.dim["factor"]), dtype=torch.float32, device=device)

        rows, cols, vals = [], [], []
        for factor in self.fg.factor_node.values():
            if not factor.contain_BP_var:
                continue

            mean_offsets = {}
            count = 0
            for var in factor.adj_var.values():
                if var.BP_MF == "BP":
                    dimj = var.belief.mean.shape[1]
                    mean_offsets[var.var_name] = count
                    count += dimj
    
            factor_row, dima = self.factor_offsets[factor.factor_name]

            for var_name, col, dim in back_var_msgs.get(factor.factor_name, []):
                var = factor.adj_var[var_name]
                Lbar = self._expectation_parameter_projection(
                    factor, var, mean_offsets, dim, dima, device
                )
                L_factor[factor_row : factor_row + dima, col : col + dim] = (
                    Lbar.transpose(1, 0)
                )

            for factor_msg_name in factor.message:
                if not self.fg.var_node[factor_msg_name].BP_MF=="BP":
                    continue

                row, dim = self.factor_msg_offsets_BP[
                    (factor.factor_name, factor_msg_name)
                ]
                var = factor.adj_var[factor_msg_name]
                Lbar = self._expectation_parameter_projection(
                    factor, var, mean_offsets, dim, dima, device
                )
                L_1_factor_msg[row : row + dim, factor_row : factor_row + dima] = Lbar

                for var_name, col, dim in back_var_msgs.get(factor.factor_name, []):
                    if factor_msg_name == var_name:
                        r = torch.arange(dim) + row
                        c = torch.arange(dim) + col
                        rows.extend(r.tolist())
                        cols.extend(c.tolist())
                        vals.extend([1.0] * dim)

        self.L_factor = L_factor.to_sparse().detach()
        self.L_1_factor_msg = L_1_factor_msg.to_sparse().detach()
        self.L_2_factor_msg = self._make_sparse_matrix(rows, cols, vals, (self.dim["factor_msg_BP"], self.dim["var_msg"]), device)

        rows, cols, vals = [], [], []
        for var in self.fg.var_node.values():
            row, _ = var_offsets[var.var_name]
            for _, col, dim in back_factor_msgs_BP.get(var.var_name, []):
                r = torch.arange(dim) + row
                c = torch.arange(dim) + col
                rows.extend(r.tolist())
                cols.extend(c.tolist())
                vals.extend([1.0] * dim)
        self.L_var_BP = self._make_sparse_matrix(rows, cols, vals, (self.dim["var"], self.dim["factor_msg_BP"]), device)

        rows, cols, vals = [], [], []
        for var in self.fg.var_node.values():
            row, _ = var_offsets[var.var_name]
            for _, col, dim in back_factor_msgs_MF.get(var.var_name, []):
                r = torch.arange(dim) + row
                c = torch.arange(dim) + col
                rows.extend(r.tolist())
                cols.extend(c.tolist())
                vals.extend([1.0] * dim)
        self.L_var_MF = self._make_sparse_matrix(rows, cols, vals, (self.dim["var"], self.dim["factor_msg_MF"]), device)

        rows, cols, vals = [], [], []
        for var in self.fg.var_node.values():
            if var.BP_MF != "BP":
                continue
            for var_msg_name in var.message:
                row, _ = self.var_msg_offsets[(var.var_name, var_msg_name)]
                for factor_name, col, dim in back_factor_msgs_BP.get(var.var_name, []):
                    if var_msg_name != factor_name:
                        r = torch.arange(dim) + row
                        c = torch.arange(dim) + col
                        rows.extend(r.tolist())
                        cols.extend(c.tolist())
                        vals.extend([1.0] * dim)

        self.L_var_msg = self._make_sparse_matrix(rows, cols, vals, (self.dim["var_msg"], self.dim["factor_msg_BP"]), device)

    def solve_one_step(self):
        """Run one synchronous BP-MF iteration."""

        self.iteration += 1
        if self.iteration > 1:
            self.compute_all_BP_var_messages()
        self.compute_all_factor_messages_MF()
        self.update_all_factor_belief()
        self.compute_all_factor_messages_BP()
        self.update_all_var_belief()
        self.set_all_MF_var_belief()

    def compute_all_BP_var_messages(self):
        """Update internal BP variable-to-factor message."""

        B = self.factor_message_BP.shape[0]
        self.var_msg = self.L_var_msg.to_dense().unsqueeze(0).expand(B, -1, -1) @ self.factor_message_BP

    def update_all_factor_belief(self):
        """Update concatenated BP factor beliefs in natural coordinates."""

        B = self.factor_natural_param.shape[0]
        self.factor_natural_param = self.gradient + self.L_factor.to_dense().unsqueeze(0).expand(B, -1, -1) @ self.var_msg

    def compute_gradient_A(self):
        """Evaluate batched gradients and residual second moments."""

        # Update effective measurement precisions from the current MF beliefs.
        for factor in self.fg.factor_node.values():
            if not isinstance(factor.factor_param, MeasurementFactor):
                continue

            E_Gamma = None
            E_Bernoulli = None
            for var in factor.adj_var.values():
                if isinstance(var.belief, invWishart):
                    factor.factor_param.inv_cov = var.belief.inv_var_expectation()
                if isinstance(var.belief, Gamma):
                    E_Gamma = var.belief.expectation()
                if isinstance(var.belief, Bernoulli):
                    E_Bernoulli = var.belief.expectation()

            if E_Gamma is not None and E_Bernoulli is not None:
                factor.factor_param.inv_cov = (
                    E_Bernoulli * factor.factor_param.inv_cov_init
                    + (1-E_Bernoulli)
                    * E_Gamma
                    * factor.factor_param.inv_cov_init0
                )

        B = self.factor_natural_param.shape[0]
        device = self.factor_natural_param.device

        results = {}

        # Process together all factors that share the same sensor model.
        for [sensor_name, dim_natural], items in self.factor_groups.items():
            d = int(((-1 + (1 + 4 * dim_natural) ** 0.5) / 2))

            rows = [torch.arange(row_idx, row_idx + dim_natural) for _, row_idx, _, _ in items]
            rows = torch.cat(rows)

            N_batch = len(items)
            factor_natural_param_batch = self.factor_natural_param[:, rows].reshape(B * N_batch, -1, 1)

            factor_eta = factor_natural_param_batch[:, :d]
            factor_Lambda = -2 * factor_natural_param_batch[:, d:].reshape(B * N_batch, d, d)

            factor_cov = torch.linalg.inv(factor_Lambda)
            factor_mean = factor_cov @ factor_eta

            factor_name = items[0][0]

            factor = self.fg.factor_node[factor_name]
            dima = factor.belief.mean.shape[1]
            if isinstance(self.meas_batch[sensor_name], State):
                dimz = 6
            else:
                dimz = self.meas_batch[sensor_name].shape[1]

            invR_batch = torch.zeros((B, N_batch, dimz, dimz), dtype=torch.float32, device=device)
            count = 0
            for factor_name, _, _, _ in items:
                invR_batch[:, count, :, :] = self.fg.factor_node[
                    factor_name
                ].factor_param.inv_cov
                count += 1

            gradient_batch = torch.zeros((B, N_batch, dim_natural, 1), dtype=torch.float32, device=device)

            if (
                self.nonlinear_approx_mode == "Taylor_Expansion"
                or factor.factor_param.if_linear
                or factor.NN_model == None
            ):
                if self.state_space == "R3" or (sensor_name == "prior" and self.state_space == "Lie_Group"):
                    residual_batch = factor.factor_param.model_fcn(
                        factor_mean,
                        self.meas_batch[sensor_name],
                        self.ext_param_batch[sensor_name],
                    )
                    J_batch = factor.factor_param.Jaco_fcn(
                        factor_mean,
                        self.meas_batch[sensor_name],
                        self.ext_param_batch[sensor_name],
                    )
                if self.state_space == "Lie_Group" and sensor_name != "prior":
                    tangent_point = []
                    var_num = len(self.factor_tangent_point[sensor_name])

                    for var_id in range(var_num):
                        state_vec = factor_mean[:, var_id*6:(var_id+1)*6]
                        tangent_point_ori = self.factor_tangent_point[sensor_name][var_id].orientation
                        tangent_point_pos = self.factor_tangent_point[sensor_name][var_id].position
                        tangent_point.append(composite_add(State(tangent_point_ori, tangent_point_pos), state_vec))

                    residual_batch = factor.factor_param.model_fcn(tangent_point, self.meas_batch[sensor_name], self.ext_param_batch[sensor_name],)
                    J_batch = factor.factor_param.Jaco_fcn(tangent_point, self.meas_batch[sensor_name], self.ext_param_batch[sensor_name])
                    J_batch = J_batch @ composite_rightJaco(factor_mean)

                residual_batch = residual_batch.reshape(B, N_batch, dimz, 1)
                J_batch = J_batch.reshape(B, N_batch, dimz, dima)

                nabla_mean = -J_batch.transpose(3, 2) @ invR_batch @ residual_batch
                nabla_cov = -0.5 * (J_batch.transpose(3, 2) @ invR_batch @ J_batch)

                factor_mean = factor_mean.reshape(B, N_batch, dima, 1)
                factor_cov = factor_cov.reshape(B, N_batch, dima, dima)

                gradient_batch[:, :, :dima] = nabla_mean - 2 * nabla_cov @ factor_mean
                gradient_batch[:, :, dima:] = nabla_cov.reshape(B, N_batch, -1, 1)

                for var in factor.adj_var.values():
                    if not isinstance(var.belief, Gaussian):
                        self.A[sensor_name] = J_batch @ factor_cov @ J_batch.transpose(
                            3, 2
                        ) + residual_batch @ residual_batch.transpose(3, 2)
                        break

            if self.nonlinear_approx_mode == "NN" and not factor.factor_param.if_linear and factor.NN_model != None:
                mean_sampling = factor_mean.repeat_interleave(self.sampling_num, dim=0)

                features_cov = cov_features(factor_cov)
                features_cov = features_cov.repeat_interleave(self.sampling_num, dim=0)

                invR_flat = invR_batch.reshape(B * N_batch, dimz, dimz)
                features_invR = cov_features(invR_flat)
                features_invR = features_invR.repeat_interleave(self.sampling_num, dim=0)

                if isinstance(self.meas_batch[sensor_name], State):
                    ori = self.meas_batch[sensor_name].orientation.repeat_interleave(self.sampling_num, dim=0)
                    pos = self.meas_batch[sensor_name].position.repeat_interleave(self.sampling_num, dim=0)
                    meas_cond = torch.cat([rotm2rotvec(ori), pos], dim=1).squeeze(-1)
                    meas_sampling = State(ori, pos)
                else:
                    meas_flat = self.meas_batch[sensor_name].reshape(B * N_batch, -1)
                    meas_cond = meas_flat.repeat_interleave(self.sampling_num, dim=0)
                    meas_sampling = meas_cond.unsqueeze(-1)

                ext_param_sampling = defaultdict(lambda: defaultdict(list))
                if self.ext_param_batch[sensor_name] != None:
                    for idx in self.ext_param_batch[sensor_name]:
                        for Rt in self.ext_param_batch[sensor_name][idx]:
                            value = self.ext_param_batch[sensor_name][idx][Rt]
                            ext_param_sampling[idx][Rt] = value.repeat_interleave(self.sampling_num, dim=0)

                if self.state_space == "R3" and sensor_name == "uwb":
                    diff_sampling = mean_sampling[:, 0:3] - mean_sampling[:, 3:6]
                    diff_sampling = diff_sampling.squeeze(-1)

                    cond = torch.cat([diff_sampling, meas_cond, features_cov, features_invR], dim=1)

                if self.state_space == "Lie_Group" and sensor_name == "uwb":
                    tangent_point = []
                    var_num = len(self.factor_tangent_point[sensor_name])

                    for var_id in range(var_num):
                        state_vec = mean_sampling[:, var_id*6:(var_id+1)*6]
                        ori = self.factor_tangent_point[sensor_name][var_id].orientation.repeat_interleave(self.sampling_num, dim=0)
                        pos = self.factor_tangent_point[sensor_name][var_id].position.repeat_interleave(self.sampling_num, dim=0)
                        tangent_point.append(composite_add(State(ori, pos), state_vec))

                    diff_sampling = composite_minus(
                        tangent_point[0], tangent_point[1]
                    ).squeeze(-1)

                    cond = torch.cat([diff_sampling, meas_cond, features_cov, features_invR], dim=1)

                z = torch.randn(B * N_batch * self.sampling_num, dima, device=device).detach()
                sampling_point, logw = factor.NN_model[self.iteration_NN[sensor_name]](z, cond)
                w = torch.exp(logw.clamp(min=-10, max=10))

                chol_cov = torch.linalg.cholesky(factor_cov)
                chol_cov_sampling = chol_cov.reshape(B * N_batch, dima, dima).repeat_interleave(self.sampling_num, dim=0)
                samples = mean_sampling + chol_cov_sampling @ sampling_point.unsqueeze(-1)

                if self.state_space == "R3":
                    residual_sampling = factor.factor_param.model_fcn(samples, meas_sampling, ext_param_sampling)
                    J_sampling = factor.factor_param.Jaco_fcn(samples, meas_sampling, ext_param_sampling)

                if self.state_space == "Lie_Group":
                    tangent_point = []
                    for var_id in range(var_num):
                        state_vec = samples[:, var_id*6:(var_id+1)*6]
                        ori = self.factor_tangent_point[sensor_name][var_id].orientation.repeat_interleave(self.sampling_num, dim=0)
                        pos = self.factor_tangent_point[sensor_name][var_id].position.repeat_interleave(self.sampling_num, dim=0)
                        tangent_point.append(composite_add(State(ori, pos), state_vec))

                    residual_sampling = factor.factor_param.model_fcn(tangent_point, meas_sampling, ext_param_sampling)
                    J_sampling = factor.factor_param.Jaco_fcn(tangent_point, meas_sampling, ext_param_sampling)
                    J_sampling = J_sampling @ composite_rightJaco(samples)

                invR_sampling = invR_flat.repeat_interleave(self.sampling_num, dim=0)

                nabla_mean_sampling = -J_sampling.transpose(1, 2) @ invR_sampling @ residual_sampling
                nabla_cov_sampling = -0.5 * J_sampling.transpose(1, 2) @ invR_sampling @ J_sampling

                nabla_mean_sampling = nabla_mean_sampling * w.reshape(-1, 1, 1)
                nabla_cov_sampling = nabla_cov_sampling * w.reshape(-1, 1, 1)

                nabla_mean_sampling = nabla_mean_sampling.reshape(B * N_batch, self.sampling_num, dima, 1)
                nabla_cov_sampling = nabla_cov_sampling.reshape(B * N_batch, self.sampling_num, dima, dima)

                nabla_mean = torch.mean(nabla_mean_sampling, dim=1)
                nabla_cov = torch.mean(nabla_cov_sampling, dim=1)

                gradient_batch[:, :, :dima] = (nabla_mean - 2 * (nabla_cov @ factor_mean)).reshape(B, N_batch, dima, 1)
                gradient_batch[:, :, dima:] = nabla_cov.reshape(B, N_batch, -1, 1)

                for var in factor.adj_var.values():
                    if not isinstance(var.belief, Gaussian):
                        A_sampling = residual_sampling @ residual_sampling.transpose(1, 2)
                        A_sampling = A_sampling * w.reshape(-1, 1, 1)
                        A_sampling = A_sampling.reshape(B*N_batch, self.sampling_num, dimz, dimz)
                        self.A[sensor_name] = torch.mean(A_sampling, dim=1)
                        self.A[sensor_name] = self.A[sensor_name].reshape(B, N_batch, dimz, dimz)
                        break

                self.iteration_NN[sensor_name] += 1

            flat = gradient_batch.reshape(B, N_batch, -1)

            for i, (factor_name, _, _, _) in enumerate(items):
                results[factor_name] = flat[:, i, :]

        ordered = [results[factor_name] for factor_name in self.factor_offsets.keys()]
        self.gradient = torch.cat(ordered, dim=1).reshape(B, -1, 1)

    def compute_all_factor_messages_BP(self):
        """Compute internal factor-to-BP variable messages."""

        B = self.factor_natural_param.shape[0]
        mean_cov = batch_natural_param2mean_cov(self.factor_natural_param, self.factor_offsets, self.factor_groups)
        msg = batch_mean_cov2_natural_param(self.L_1_factor_msg.to_dense().unsqueeze(0).expand(B, -1, -1) @ mean_cov, self.factor_msg_offsets_BP, self.factor_msg_groups_BP)
        self.factor_message_BP = msg- self.L_2_factor_msg.to_dense().unsqueeze(0).expand(B, -1, -1) @ self.var_msg

    def compute_all_factor_messages_MF(self):
        """Compute closed-form MF messages for all conjugate variables."""

        self.compute_gradient_A()

        B = self.factor_natural_param.shape[0]
        device = self.factor_natural_param.device
        sensor_name_keys = self.A.keys()
        count_A = {key: 0 for key in sensor_name_keys}
        self.factor_message_MF = torch.zeros((B, self.dim["factor_msg_MF"], 1), dtype=torch.float32, device=device)
        count = 0

        for factor in self.fg.factor_node.values():
            if isinstance(factor.factor_param, MeasurementFactor):
                sensor_name = factor.factor_param.sensor_name
                if sensor_name in sensor_name_keys:
                    A_use = self.A[sensor_name][:, count_A[sensor_name]]
                    count_A[sensor_name] += 1
                    inv_cov_init = factor.factor_param.inv_cov_init

                    for var in factor.adj_var.values():
                        if isinstance(var.belief, invWishart):
                            self.factor_message_MF[:, count : count + 1] = -0.5
                            count += 1

                            dim = A_use.shape[1] ** 2
                            self.factor_message_MF[:, count : count + dim] = -0.5 * A_use.reshape(B, -1, 1)
                            count += dim

                        elif isinstance(var.belief, Bernoulli):
                            var_Gamma = next((v for v in factor.adj_var.values() if isinstance(v.belief, Gamma)),None)
                            if var_Gamma:
                                self.factor_message_MF[:, count : count + 1] = 0.5 * (A_use * factor.factor_param.inv_cov_init0 * var_Gamma.belief.expectation() - var_Gamma.belief.ln_expectation() - A_use * inv_cov_init - torch.log(factor.factor_param.inv_cov_init0) + torch.log(inv_cov_init))
                                count += 1

                        elif isinstance(var.belief, Gamma):
                            var_Bernoulli = next((v for v in factor.adj_var.values() if isinstance(v.belief, Bernoulli)), None)
                            if var_Bernoulli:
                                self.factor_message_MF[:, count : count + 1] = 0.5 * (1 - var_Bernoulli.belief.expectation())
                                count += 1

                                self.factor_message_MF[:, count : count + 1] = -0.5 * (1 - var_Bernoulli.belief.expectation()) * A_use * factor.factor_param.inv_cov_init0
                                count += 1

            elif isinstance(factor.factor_param, EF_PriorFactor):
                dim = factor.factor_param.prior_natural_param.shape[1]
                self.factor_message_MF[:, count : count + dim] = factor.factor_param.prior_natural_param
                count += dim

            elif isinstance(factor.factor_param, EF_ConnectFactor):
                vars_list = list(factor.adj_var.values())
                for i, var in enumerate(vars_list):
                    another_var = vars_list[1 - i]
                    if isinstance(var.belief, Beta):
                        self.factor_message_MF[:, count : count + 1] = another_var.belief.expectation()
                        count += 1

                        self.factor_message_MF[:, count : count + 1] = 1-another_var.belief.expectation()
                        count += 1

                    elif isinstance(var.belief, Bernoulli):
                        self.factor_message_MF[:, count : count + 1] = another_var.belief.ln_expectation()-another_var.belief.ln_1_expectation()
                        count += 1

    def update_all_var_belief(self):
        """Combine incoming BP and MF messages into variable beliefs."""

        B = self.factor_message_BP.shape[0]
        self.var_natural_param = self.L_var_BP.to_dense().unsqueeze(0).expand(B, -1, -1)@self.factor_message_BP + self.L_var_MF.to_dense().unsqueeze(0).expand(B, -1, -1)@self.factor_message_MF

    def set_all_MF_var_belief(self):
        """Write updated natural parameters to local MF variable objects."""

        count = 0
        for var in self.fg.var_node.values():
            dim = var.belief.natural_param.shape[1]
            if var.BP_MF == "MF":
                natural_param = self.var_natural_param[:, count : count + dim]
                var.belief.set_by_natural_param(natural_param)
            count += dim

    def set_all_BP_var_belief(self):
        """Write updated natural parameters to local BP variable objects."""

        count = 0
        for var in self.fg.var_node.values():
            dim = var.belief.natural_param.shape[1]
            if var.BP_MF == "BP":
                natural_param = self.var_natural_param[:, count:count+dim]
                var.belief.set_by_natural_param(natural_param)
            count += dim

    def retract_all_BP_var_belief(self):
        """Retract BP means onto the manifold and transport covariances."""

        for var in self.fg.var_node.values():
            if var.BP_MF == "BP":
                var.tangent_point = composite_add(var.tangent_point, var.belief.mean)

                mean = torch.zeros_like(var.belief.mean)
                J = composite_rightJaco(var.belief.mean)
                cov = J @ var.belief.cov @ J.transpose(1, 2)
                var.belief.set_by_Mean_Cov(mean, cov)

    def set_all_BP_factor_belief(self):
        """Reinitialize BP factor tangent points after state retraction."""

        for factor in self.fg.factor_node.values():
            if factor.contain_BP_var:
                if self.state_space == "Lie_Group":
                    orientation_list = []
                    position_list = []
                    for var in factor.adj_var.values():
                        if var.BP_MF == "BP":
                            orientation_list.append(var.tangent_point.orientation)
                            position_list.append(var.tangent_point.position)
                    orientation = torch.stack(orientation_list, dim=0)
                    position = torch.stack(position_list, dim=0)
                    factor.tangent_point = State(orientation, position)

                mean = torch.zeros_like(factor.belief.mean)
                cov = torch.zeros_like(factor.belief.cov)
                count = 0
                for var in factor.adj_var.values():
                    if var.BP_MF == "BP":
                        dimi = var.belief.mean.shape[1]
                        mean[:,count:count+dimi] = var.belief.mean
                        cov[:,count:count+dimi,count:count+dimi] = var.belief.cov
                        count += dimi

                factor.belief.set_by_Mean_Cov(mean, cov)

def batch_natural_param2mean_cov(factor_natural_param, factor_offsets, groups):
    """Convert grouped Gaussian natural parameters to stacked mean and cov."""

    B = factor_natural_param.shape[0]
    results = {}
    for (_, dim_natural), items in groups.items():
        d = int(((-1 + (1 + 4 * dim_natural) ** 0.5) / 2))

        rows = [torch.arange(row_idx, row_idx + dim_natural) for _, row_idx, _, _ in items]
        rows = torch.cat(rows)

        N_batch = len(items)
        factor_natural_param_batch = factor_natural_param[:, rows].reshape(B, N_batch, dim_natural, 1)

        eta = factor_natural_param_batch[:, :, :d]
        Lambda = -2 * factor_natural_param_batch[:, :, d:].reshape(B, N_batch, d, d)

        cov = torch.linalg.inv(Lambda)
        mean = cov @ eta

        flat = torch.cat([mean.reshape(B, N_batch, -1), cov.reshape(B, N_batch, -1)], dim=-1)
        for i, (factor_name, _, _, _) in enumerate(items):
            results[factor_name] = flat[:, i, :]

    ordered = [results[factor_name] for factor_name in factor_offsets.keys()]
    mean_cov = torch.cat(ordered, dim=1)
    return mean_cov.reshape(B, -1, 1)


def batch_mean_cov2_natural_param(mean_cov, msg_offsets, groups):
    """Convert grouped Gaussian moments to stacked natural parameters."""

    B = mean_cov.shape[0]
    results = {}
    for dim_natural, items in groups.items():
        d = int(((-1 + (1 + 4 * dim_natural) ** 0.5) / 2))

        rows = [torch.arange(row_idx, row_idx + dim_natural) for _, _, row_idx in items]
        rows = torch.cat(rows)

        N_batch = len(items)
        mean_cov_batch = mean_cov[:, rows].reshape(B, N_batch, dim_natural, 1)

        mean = mean_cov_batch[:, :, :d]
        cov = mean_cov_batch[:, :, d:].reshape(B, N_batch, d, d)

        Lambda = torch.linalg.inv(cov)
        eta = Lambda @ mean

        flat = torch.cat([eta.reshape(B, N_batch, -1), -0.5 * Lambda.reshape(B, N_batch, -1)], dim=-1)
        for i, (factor_name, var_name, _) in enumerate(items):
            results[factor_name, var_name] = flat[:, i, :]

    ordered = [
        results[factor_name, var_name] for (factor_name, var_name) in msg_offsets.keys()
    ]
    natural_param = torch.cat(ordered, dim=1)
    return natural_param.reshape(B, -1, 1)


def cov_features(cov):
    """Encode covariance using log standard deviations and Fisher correlations."""

    sigma = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    log_sigma = torch.log(sigma + 1e-8)

    denom = sigma.unsqueeze(-1) * sigma.unsqueeze(-2)
    R = cov / (denom + 1e-8)

    idx = torch.triu_indices(R.size(-2), R.size(-1), offset=1)
    r_vals = R[:, idx[0], idx[1]]

    fisher_z = torch.atanh(r_vals)

    features = torch.cat([log_sigma, fisher_z], dim=-1)

    return features
