"""Batched BP-MF solver."""

import copy
from collections import defaultdict, OrderedDict
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix

from Distribution import Gaussian, invWishart, Gamma, Beta, Bernoulli
from Factor import MeasurementFactor, EF_PriorFactor, EF_ConnectFactor
from Transform import State, rotm2rotvec, composite_add, composite_minus, composite_rightJaco

_MATRIX_NAMES = ('L_factor', 'L_1_factor_msg', 'L_2_factor_msg',
                 'L_var_BP', 'L_var_MF', 'L_var_msg')
_CACHE = OrderedDict()
_STATS = {'hits': 0, 'misses': 0}


@lru_cache(maxsize=256)
def _projection_indices(factor_mean_dim, start, var_mean_dim):
    """
    Return the nonzero (row, col) indices of
        L_ai = diag(Gamma_ai, kron(Gamma_ai, Gamma_ai)),
    """
    # var_expectation_param = L_ai @ factor_expectation_param
    sub = np.arange(start, start + var_mean_dim, dtype=np.intp)
    cols = np.concatenate((sub, factor_mean_dim +
                           (sub[:, None] * factor_mean_dim + sub[None, :]).ravel()))
    return np.arange(len(cols), dtype=np.intp), cols


def _csr(blocks, shape):
    if blocks:
        rows = np.concatenate([r for r, c in blocks])
        cols = np.concatenate([c for r, c in blocks])
        return coo_matrix((np.ones(rows.size), (rows, cols)), shape=shape).tocsr()
    return coo_matrix(shape).tocsr()


class fg_solver:
    """Solve one robot's local graph while exposing cross-robot messages.

    Internally, natural parameters and messages are concatenated and updated
    with sparse mapping matrices so factors of the same sensor type can be
    evaluated in batches.
    """
    optimization = "cached"
    max_cache_entries = 64

    @staticmethod
    def clear_cache():
        _CACHE.clear()
        _STATS.update(hits=0, misses=0)
        _projection_indices.cache_clear()

    @staticmethod
    def cache_info():
        return dict(_STATS, entries=len(_CACHE))


    def __init__(self, fg, nonlinear_approx_mode, state_space, sampling_num):
        """Initialize solver."""

        self.iteration = 0

        self.fg = fg
        fg.solver = self

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

        # Mapping:
        # factor_name -> (start index, natural parameter dimension of factor)
        self.factor_offsets = {}

        self.dim["factor"] = 0
        factor_natural_param_list = []

        # Mapping:
        # (sensor_name, natural parameter dimension) -> list of factor information
        # Each list item is:
        # (factor_name, start index, measurement, extrinsic parameters)
        self.factor_groups = defaultdict(list)

        # Mapping:
        # sensor_name -> list of ordinary NumPy-array measurements
        # The list is stacked later to form one measurement batch.
        meas_dict = defaultdict(list)

        # Nested mapping:
        # sensor_name -> "orientation"/"position" -> list of arrays
        # It temporarily stores State-type measurements before batching.
        meas_state_dict = defaultdict(lambda: defaultdict(list))

        # Nested mapping:
        # sensor_name -> adjacent state index -> "orientation"/"position" -> list
        # It collects the tangent points of corresponding states across the same sensor-type factors.
        factor_tangent_point_dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        # Nested mapping:
        # sensor_name -> extrinsic parameter index -> parameter name -> list of values
        # For example, ext_param_dict["uwb"][0]["t"] stores the first UWB lever arm
        # from every UWB factor before these values are stacked into a batch.
        ext_param_dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for factor in self.fg.factor_node.values():
            if not factor.contain_BP_var:
                continue

            sensor_name = factor.factor_param.sensor_name
            dim = len(factor.belief.natural_param)
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

            # Collect the tangent point of each adjacent state for batched Lie group operations.
            if self.state_space == "Lie_Group":
                for d in range(factor.tangent_point.orientation.shape[0]):
                    factor_tangent_point_dict[sensor_name][d]["orientation"].append(
                        factor.tangent_point.orientation[d]
                    )
                    factor_tangent_point_dict[sensor_name][d]["position"].append(
                        factor.tangent_point.position[d]
                    )

            if isinstance(factor.factor_param.meas, State):
                meas_state_dict[sensor_name]["orientation"].append(
                    factor.factor_param.meas.orientation
                )
                meas_state_dict[sensor_name]["position"].append(
                    factor.factor_param.meas.position
                )
            else:
                meas_dict[sensor_name].append(factor.factor_param.meas)

            if factor.factor_param.ext_param is not None:
                for idx, ext_param in enumerate(factor.factor_param.ext_param):
                    for Rt, value in ext_param.items():
                        ext_param_dict[sensor_name][idx][Rt].append(value)

        self.factor_natural_param = np.vstack(factor_natural_param_list)
        self._stack_sensor_batches(
            factor_tangent_point_dict, meas_state_dict, meas_dict, ext_param_dict
        )

    def _stack_sensor_batches(
        self, factor_tangent_point_dict, meas_state_dict, meas_dict, ext_param_dict
    ):
        """Stack the temporary per-factor lists collected for each sensor."""

        # Nested mapping:
        # sensor_name -> adjacent state index -> State
        # Each State stores batched orientations (N, 3, 3) and positions (N, 3, 1).
        self.factor_tangent_point = defaultdict(lambda: defaultdict(dict))

        # Mapping:
        # sensor_name -> batched measurement
        # The value is either a State or a NumPy array with the first dimension as batch size.
        self.meas_batch = {}

        # Nested mapping:
        # sensor_name -> parameter index -> parameter name -> batched parameter array
        self.ext_param_batch = defaultdict(lambda: defaultdict(dict))

        # Mapping:
        # sensor_name -> current normalizing flow iteration index
        self.iteration_NN = {}

        for sensor_name, _ in self.factor_groups:
            if self.state_space == "Lie_Group":
                for idx in factor_tangent_point_dict[sensor_name]:
                    orientation = np.stack(
                        factor_tangent_point_dict[sensor_name][idx]["orientation"],
                        axis=0,
                    )
                    position = np.stack(
                        factor_tangent_point_dict[sensor_name][idx]["position"],
                        axis=0,
                    )
                    self.factor_tangent_point[sensor_name][idx] = State(
                        orientation, position
                    )

            if sensor_name in meas_state_dict:
                orientation = np.stack(
                    meas_state_dict[sensor_name]["orientation"], axis=0
                )
                position = np.stack(meas_state_dict[sensor_name]["position"], axis=0)
                self.meas_batch[sensor_name] = State(orientation, position)

            if sensor_name in meas_dict:
                self.meas_batch[sensor_name] = np.stack(
                    meas_dict[sensor_name], axis=0
                )

            if sensor_name in ext_param_dict:
                for idx in ext_param_dict[sensor_name]:
                    for Rt in ext_param_dict[sensor_name][idx]:
                        self.ext_param_batch[sensor_name][idx][Rt] = np.stack(
                            ext_param_dict[sensor_name][idx][Rt], axis=0
                        )

            self.iteration_NN[sensor_name] = 0


    def _initialize_variable_beliefs(self):
        """Pack local variable beliefs and return their vector offsets."""

        # Mapping:
        # variable_name -> (start index, natural parameter dimension of variable)
        # It locates each variable belief in the stacked variable natural parameter vector.
        var_offsets = {}
        self.dim["var"] = 0
        var_natural_param_list = []

        for var in self.fg.var_node.values():
            dim = len(var.belief.natural_param)
            var_offsets[var.var_name] = (self.dim["var"], dim)
            self.dim["var"] += dim
            var_natural_param_list.append(var.belief.natural_param)

        self.var_natural_param = np.vstack(var_natural_param_list)
        return var_offsets

    def _initialize_factor_message_offsets(self):
        """Allocate factor-to-variable blocks for BP and MF messages."""

        # Mapping:
        # (factor_name, variable_name) -> (start index, dimension of factor-to-variable message)
        # It locates each factor-to-BP variable message in the stacked BP message vector.
        self.factor_msg_offsets_BP = {}

        self.dim["factor_msg_BP"] = 0

        # Mapping:
        # message dimension -> list of local BP factor-to-variable messages
        # Each list item is (factor_name, variable_name, start index).
        # Only messages computed by the current solver are included.
        # Externally received messages are excluded.
        self.intern_factor_msg_groups_BP = defaultdict(list)

        for factor in self.fg.factor_node.values():
            if not factor.contain_BP_var:
                continue
            for var_name in factor.adj_var:
                if factor.adj_var[var_name].BP_MF == "BP":
                    dim = len(factor.adj_var[var_name].belief.natural_param)
                    offset = self.dim["factor_msg_BP"]
                    self.factor_msg_offsets_BP[(factor.factor_name, var_name)] = (
                        offset,
                        dim,
                    )
                    self.intern_factor_msg_groups_BP[dim].append(
                        (factor.factor_name, var_name, offset)
                    )
                    self.dim["factor_msg_BP"] += dim

        # It locates each local factor-to-BP variable message.
        self.intern_factor_msg_offsets_BP = copy.deepcopy(
            self.factor_msg_offsets_BP
        )
        self.dim["intern_factor_msg_BP"] = self.dim["factor_msg_BP"]

        # Messages from factors owned by other robots.
        for factor in self.fg.extern_factor_node.values():
            if not factor.contain_BP_var:
                continue
            for var_name in factor.adj_var:
                if (
                    factor.adj_var[var_name].BP_MF == "BP"
                    and var_name in self.fg.var_node
                ):
                    dim = len(factor.adj_var[var_name].belief.natural_param)
                    self.factor_msg_offsets_BP[(factor.factor_name, var_name)] = (
                        self.dim["factor_msg_BP"],
                        dim,
                    )
                    self.dim["factor_msg_BP"] += dim

        # Mapping:
        # (factor_name, variable_name) -> (start index, dimension)
        # It locates each factor-to-MF variable message in the stacked MF message vector.
        self.factor_msg_offsets_MF = {}
        self.dim["factor_msg_MF"] = 0

        # Allocate MF message blocks produced by locally owned factors.
        for factor in self.fg.factor_node.values():
            self._append_mf_factor_message_offsets(factor, local_var_only=False)

        # Allocate MF messages from external factors.
        for factor in self.fg.extern_factor_node.values():
            self._append_mf_factor_message_offsets(factor, local_var_only=True)

    def _append_mf_factor_message_offsets(self, factor, local_var_only):
        """Append MF message blocks for one local or externally owned factor."""

        for var_name in factor.adj_var:
            var_is_local = var_name in self.fg.var_node
            if factor.adj_var[var_name].BP_MF == "MF" and (
                not local_var_only or var_is_local
            ):
                dim = len(factor.adj_var[var_name].belief.natural_param)
                self.factor_msg_offsets_MF[(factor.factor_name, var_name)] = (
                    self.dim["factor_msg_MF"],
                    dim,
                )
                self.dim["factor_msg_MF"] += dim

    def _initialize_variable_message_offsets(self):
        """Allocate and initialize BP variable-to-factor message blocks."""

        # Mapping:
        # (variable_name, factor_name) -> (start index, dimension)
        # It locates each BP variable-to-factor message in the stacked variable message vector.
        self.var_msg_offsets = {}
        self.dim["var_msg"] = 0
        var_msg_list = []

        for var in self.fg.var_node.values():
            if var.BP_MF == "BP":
                self._append_variable_message_offsets(
                    var, var.adj_factor, var_msg_list
                )

        for var in self.fg.extern_var_node.values():
            if var.BP_MF == "BP":
                local_factor_names = (
                    factor_name
                    for factor_name in var.adj_factor
                    if factor_name in self.fg.factor_node
                )
                self._append_variable_message_offsets(
                    var, local_factor_names, var_msg_list
                )

        # Stack all initialized BP variable-to-factor messages into one column vector.
        # Its shape is (total variable message dimension, 1).
        self.var_msg = np.vstack(var_msg_list)

    def _append_variable_message_offsets(self, var, factor_names, var_msg_list):
        """Append initialized variable-to-factor messages for one BP variable."""

        dim = len(var.belief.natural_param)
        for factor_name in factor_names:
            self.var_msg_offsets[(var.var_name, factor_name)] = (
                self.dim["var_msg"],
                dim,
            )
            self.dim["var_msg"] += dim
            var_msg_list.append(var.belief.natural_param)

    def _build_reverse_message_indices(self):
        """Index packed messages by destination variable or factor name."""

        # Reverse mapping for factor-to-BP variable messages:
        # variable_name -> list of (factor_name, start index, dimension)
        # It provides all incoming BP factor messages associated with each variable.
        back_factor_msgs_BP = defaultdict(list)
        for (factor_name, var_name), (col, dim) in self.factor_msg_offsets_BP.items():
            back_factor_msgs_BP[var_name].append(
                (factor_name, col, dim)
            )

        # Reverse mapping for factor-to-MF variable messages:
        # variable_name -> list of (factor_name, start index, dimension)
        # It provides all incoming factor messages associated with each variable.
        back_factor_msgs_MF = defaultdict(list)
        for (factor_name, var_name), (col, dim) in self.factor_msg_offsets_MF.items():
            back_factor_msgs_MF[var_name].append(
                (factor_name, col, dim)
            )

        # Reverse mapping for BP variable-to-factor messages:
        # factor_name -> list of (variable_name, start index, dimension)
        # It provides all incoming variable messages associated with each factor.
        back_var_msgs = defaultdict(list)
        for (var_name, factor_name), (col, dim) in self.var_msg_offsets.items():
            back_var_msgs[factor_name].append(
                (var_name, col, dim)
            )

        return back_factor_msgs_BP, back_factor_msgs_MF, back_var_msgs

    def _build_mapping_matrices(self, back_var_msgs, var_offsets,
                                   back_factor_msgs_BP, back_factor_msgs_MF):

            shape = ((self.dim['factor'], self.dim['var_msg']),
                     (self.dim['intern_factor_msg_BP'], self.dim['factor']),
                     (self.dim['intern_factor_msg_BP'], self.dim['var_msg']),
                     (self.dim['var'], self.dim['factor_msg_BP']),
                     (self.dim['var'], self.dim['factor_msg_MF']),
                     (self.dim['var_msg'], self.dim['factor_msg_BP']))

            L_factor_list, L_1_factor_msg_list, L_2_factor_msg_list, L_var_BP_list, L_var_MF_list, L_var_msg_list = [], [], [], [], [], []

            # factor_natural_param = gradient + L_factor @ var_msg
            # 
            # internal factor-to-BP message = 
            #   g(L_1_factor_msg @ f (factor_natural_param))
            #       -L_2_factor_msg @ var_msg
            # g: expectation parameter to natural parameter
            # f: natural parameter to expectation parameter

            for factor in self.fg.factor_node.values():
                if not factor.contain_BP_var:
                    continue

                mean_offsets = {}
                cursor = 0
                for var_name, var in factor.adj_var.items():
                    if var.BP_MF == 'BP':
                        mean_offsets[var_name] = cursor
                        cursor += len(var.belief.mean)
                        
                factor_param_offset, _ = self.factor_offsets[factor.factor_name]
                factor_mean_dim = len(factor.belief.mean)

                for var_name, var_msg_offset, _ in back_var_msgs.get(factor.factor_name, []):
                    var_mean_dim = len(factor.adj_var[var_name].belief.mean)
                    L_factor_list.append((factor_param_offset, var_msg_offset, factor_mean_dim, mean_offsets[var_name], var_mean_dim))

                for var_name in factor.message:
                    var = factor.adj_var[var_name]
                    if var.BP_MF != 'BP':
                        continue

                    var_mean_dim = len(var.belief.mean)
                    factor_msg_offset, var_param_dim = self.factor_msg_offsets_BP[factor.factor_name, var_name]
                    L_1_factor_msg_list.append((factor_msg_offset, factor_param_offset, factor_mean_dim, mean_offsets[var_name], var_mean_dim))

                    var_msg_offset, _ = self.var_msg_offsets[var_name, factor.factor_name]
                    L_2_factor_msg_list.append((factor_msg_offset, var_msg_offset, var_param_dim))

            # var_natural_param =
            #     L_var_BP @ factor_msg_BP + L_var_MF @ factor_msg_MF

            # var_msg = L_var_msg @ factor_msg_BP

            for var in self.fg.var_node.values():
                var_param_offset, _ = var_offsets[var.var_name]
                incoming_BP_msgs = back_factor_msgs_BP.get(var.var_name, [])
                incoming_MF_msgs = back_factor_msgs_MF.get(var.var_name, [])

                L_var_BP_list.extend((var_param_offset, factor_msg_offset, msg_dim) for _, factor_msg_offset, msg_dim in incoming_BP_msgs)
                L_var_MF_list.extend((var_param_offset, factor_msg_offset, msg_dim) for _, factor_msg_offset, msg_dim in incoming_MF_msgs)

                if var.BP_MF == 'BP':
                    for target_factor_name in var.message:
                        var_msg_offset, _ = self.var_msg_offsets[var.var_name, target_factor_name]
                        L_var_msg_list.extend(
                            (var_msg_offset, factor_msg_offset, msg_dim) 
                            for (source_factor_name, factor_msg_offset, msg_dim) in incoming_BP_msgs 
                            if target_factor_name != source_factor_name)

            key = (shape, tuple(L_factor_list), tuple(L_1_factor_msg_list), tuple(L_2_factor_msg_list), tuple(L_var_BP_list), tuple(L_var_MF_list), tuple(L_var_msg_list))

            matrices = _CACHE.get(key)

            if matrices is not None:
                _STATS['hits'] += 1
                _CACHE.move_to_end(key)
            else:
                _STATS['misses'] += 1
                block_lists = [[] for _ in shape]
                # block_lists[0] -> L_factor
                # block_lists[1] -> L_1_factor_msg
                # block_lists[2] -> L_2_factor_msg
                # block_lists[3] -> L_var_BP
                # block_lists[4] -> L_var_MF
                # block_lists[5] -> L_var_msg

                for factor_param_offset, var_msg_offset, factor_mean_dim, start, var_mean_dim in L_factor_list:
                    small, large = _projection_indices(factor_mean_dim, start, var_mean_dim)
                    block_lists[0].append((factor_param_offset + large, var_msg_offset + small))

                for factor_msg_offset, factor_param_offset, factor_mean_dim, start, var_mean_dim in L_1_factor_msg_list:
                    small, large = _projection_indices(factor_mean_dim, start, var_mean_dim)
                    block_lists[1].append((factor_msg_offset + small, factor_param_offset + large))

                for blocks, specs in zip(block_lists[2:], (L_2_factor_msg_list, L_var_BP_list, L_var_MF_list, L_var_msg_list)):
                    for row, col, dim in specs:
                        idx = np.arange(dim)
                        blocks.append((row + idx, col + idx))

                matrices = tuple(_csr(blocks, dims) for blocks, dims in zip(block_lists, shape))
                _CACHE[key] = matrices
                while len(_CACHE) > self.max_cache_entries:
                    _CACHE.popitem(last=False)

            # Give solver independent mutable matrices, preserving customization.
            for name, matrix in zip(_MATRIX_NAMES, matrices):
                setattr(self, name, matrix.copy())




    def compute_all_BP_var_messages(self):
        """Update internal BP variable-to-factor message."""

        self.var_msg = self.L_var_msg @ self.factor_msg_BP

    def update_all_factor_belief(self):
        """Update concatenated BP factor beliefs in natural coordinates."""

        self.factor_natural_param = self.gradient + self.L_factor @ self.var_msg

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

        results = {}

        # Process together all factors that share the same sensor model.
        for (sensor_name, dim_natural), items in self.factor_groups.items():
            d = int(((-1 + (1 + 4 * dim_natural) ** 0.5) / 2))

            rows = [np.arange(row_idx, row_idx + dim_natural) for _, row_idx, _, _ in items]
            rows = np.concatenate(rows)

            factor_num = len(items)
            factor_natural_param_batch = self.factor_natural_param[rows].reshape(factor_num, -1, 1)

            factor_eta = factor_natural_param_batch[:, :d]
            factor_Lambda = -2 * factor_natural_param_batch[:, d:].reshape(factor_num, d, d)

            factor_cov = np.linalg.inv(factor_Lambda)
            factor_mean = factor_cov @ factor_eta

            factor_name = items[0][0]

            factor = self.fg.factor_node[factor_name]
            dima = len(factor.belief.mean)
            if isinstance(self.meas_batch[sensor_name], State):
                dimz = 6
            else:
                dimz = self.meas_batch[sensor_name].shape[1]

            invR_batch = np.zeros((factor_num, dimz, dimz))
            count = 0
            for factor_name, _, _, _ in items:
                invR_batch[count, :, :] = self.fg.factor_node[factor_name].factor_param.inv_cov
                count += 1

            gradient_batch = np.zeros((factor_num, dima+dima**2, 1))
            
            if (
                self.nonlinear_approx_mode == "Taylor_Expansion"
                or factor.factor_param.if_Taylor_Expansion
                or (
                    factor.NN_model is None
                    and self.nonlinear_approx_mode == "NN"
                )
            ):
                if self.state_space =="R3" or (sensor_name =="prior" and self.state_space =="Lie_Group"):
                    residual_batch = factor.factor_param.model_fcn(
                        factor_mean, self.meas_batch[sensor_name], self.ext_param_batch[sensor_name]
                    )
                    J_batch = factor.factor_param.Jaco_fcn(
                        factor_mean, self.meas_batch[sensor_name], self.ext_param_batch[sensor_name]
                    )
                if self.state_space =="Lie_Group" and sensor_name !="prior":
                    tangent_point = []
                    var_num = len(self.factor_tangent_point[sensor_name])
                    
                    for var_id in range(var_num):
                        state_vec = factor_mean[:,var_id*6:(var_id+1)*6]
                        tangent_point_ori = self.factor_tangent_point[sensor_name][var_id].orientation
                        tangent_point_pos = self.factor_tangent_point[sensor_name][var_id].position
                        tangent_point.append(composite_add(State(tangent_point_ori, tangent_point_pos), state_vec))

                    residual_batch = factor.factor_param.model_fcn(tangent_point, self.meas_batch[sensor_name], self.ext_param_batch[sensor_name])
                    J_batch = factor.factor_param.Jaco_fcn(tangent_point, self.meas_batch[sensor_name], self.ext_param_batch[sensor_name])
                    # J_batch returned by the sensor model is the residual Jacobian with respect
                    # to a local right perturbation applied at the current retracted state.
                    # Apply the block right Jacobian to convert it, through the chain rule, into
                    # the Jacobian with respect to the factor's Gaussian error-state coordinates.
                    J_batch = J_batch @ composite_rightJaco(factor_mean)
                
                nabla_mean = -J_batch.transpose(0, 2, 1) @ invR_batch @ residual_batch
                nabla_cov = -0.5 * (J_batch.transpose(0, 2, 1) @ invR_batch @ J_batch)

                gradient_batch[:, :dima] = nabla_mean - 2 * nabla_cov @ factor_mean
                gradient_batch[:, dima:] = nabla_cov.reshape(nabla_mean.shape[0], -1, 1)

                for var in factor.adj_var.values():
                    if not isinstance(var.belief, Gaussian):
                        self.A[sensor_name] = J_batch @ factor_cov @ J_batch.transpose(0, 2, 1) + \
                            residual_batch @ residual_batch.transpose(0, 2, 1)
                        break

            if self.nonlinear_approx_mode == "Sampling" and not factor.factor_param.if_Taylor_Expansion:
                MC_mode = True
            else:
                MC_mode = False
            if self.nonlinear_approx_mode == "NN" and not factor.factor_param.if_Taylor_Expansion and factor.NN_model is not None:
                NN_mode = True
            else:
                NN_mode = False

            if MC_mode or NN_mode:

                var_num = len(self.factor_tangent_point[sensor_name])
                
                mean_sampling = np.repeat(factor_mean, self.sampling_num, axis=0)

                if isinstance(self.meas_batch[sensor_name], State):
                    ori = np.repeat(self.meas_batch[sensor_name].orientation, self.sampling_num, axis=0)
                    pos = np.repeat(self.meas_batch[sensor_name].position, self.sampling_num, axis=0)
                    meas_sampling = State(ori,pos)

                    if NN_mode:
                        meas_cond = np.concatenate([rotm2rotvec(ori), pos], axis=1).squeeze(-1)
                else:
                    meas_flat = self.meas_batch[sensor_name].squeeze(-1)
                    meas_sampling = np.repeat(meas_flat, self.sampling_num, axis=0)[..., None]

                    if NN_mode:
                        meas_cond = meas_sampling.squeeze(-1)

                ext_param_sampling = defaultdict(lambda: defaultdict(list))
                if self.ext_param_batch[sensor_name] is not None:
                    for idx in self.ext_param_batch[sensor_name]:
                        for Rt in self.ext_param_batch[sensor_name][idx]:
                            value = self.ext_param_batch[sensor_name][idx][Rt]
                            ext_param_sampling[idx][Rt] = np.repeat(value,self.sampling_num, axis=0)

                gaussian_point = np.random.randn(factor_num*self.sampling_num, dima, 1)

                if NN_mode:
                    features_cov = np.repeat(
                        cov_features(factor_cov), self.sampling_num, axis=0
                    )
                    features_invR = np.repeat(
                        cov_features(invR_batch), self.sampling_num, axis=0
                    )

                    if self.state_space == "R3" and sensor_name == "uwb":
                        diff_sampling = (mean_sampling[:, 0:3] - mean_sampling[:, 3:6]).squeeze(-1)
                        parts = [diff_sampling, meas_cond, features_cov, features_invR]

                    if self.state_space == "Lie_Group" and sensor_name == "uwb":
                        tangent_point = []
                        var_num = len(self.factor_tangent_point[sensor_name])

                        for var_id in range(var_num):
                            state_vec = mean_sampling[:, var_id*6:(var_id+1)*6]
                            ori = np.repeat(self.factor_tangent_point[sensor_name][var_id].orientation, self.sampling_num, axis=0)
                            pos = np.repeat(self.factor_tangent_point[sensor_name][var_id].position, self.sampling_num, axis=0)
                            tangent_point.append(
                                composite_add(State(ori, pos), state_vec)
                            )

                        diff_sampling = composite_minus(
                            tangent_point[0], tangent_point[1]
                        ).squeeze(-1)
                        parts = [
                            diff_sampling,
                            meas_cond,
                            features_cov,
                            features_invR,
                        ]

                    cond = np.concatenate(parts, axis=1)

                    session, input_names, output_names = factor.NN_model[
                        self.iteration_NN[sensor_name]
                    ]
                    outputs = session.run(
                        output_names,
                        {
                            input_names[0]: gaussian_point.squeeze(-1).astype(np.float32),
                            input_names[1]: cond.astype(np.float32),
                        },
                    )

                    sampling_point = outputs[0][..., None]
                    logw = outputs[1]
                    w = np.exp(logw)
                else:
                    sampling_point = gaussian_point
                    w = np.ones(factor_num*self.sampling_num)

                chol_cov = np.linalg.cholesky(factor_cov)
                chol_cov_sampling = np.repeat(chol_cov, self.sampling_num, axis=0)
                samples = mean_sampling + chol_cov_sampling @ sampling_point

                if self.state_space =="R3":
                    residual_sampling = factor.factor_param.model_fcn(samples, meas_sampling, ext_param_sampling)
                    J_sampling = factor.factor_param.Jaco_fcn(samples, meas_sampling, ext_param_sampling)

                if self.state_space =="Lie_Group":
                    tangent_point = []
                    for var_id in range(var_num):
                        state_vec = samples[:,var_id*6:(var_id+1)*6]
                        ori = np.repeat(self.factor_tangent_point[sensor_name][var_id].orientation, self.sampling_num, axis=0)
                        pos = np.repeat(self.factor_tangent_point[sensor_name][var_id].position, self.sampling_num, axis=0)
                        tangent_point.append(composite_add(State(ori, pos), state_vec))
                
                    residual_sampling = factor.factor_param.model_fcn(
                        tangent_point, meas_sampling, ext_param_sampling
                    )
                    J_sampling = factor.factor_param.Jaco_fcn(
                        tangent_point, meas_sampling, ext_param_sampling
                    )
                    # J_sampling returned by the sensor model is the residual Jacobian with respect
                    # to a local right perturbation applied at the current retracted state.
                    # Apply the block right Jacobian to convert it, through the chain rule, into
                    # the Jacobian with respect to the factor's Gaussian error-state coordinates.
                    J_sampling = J_sampling @ composite_rightJaco(samples)

                invR_sampling = invR_batch.repeat(self.sampling_num, axis=0)
                
                nabla_mean_sampling = -J_sampling.transpose(0, 2, 1) @ invR_sampling @ residual_sampling
                nabla_cov_sampling  = -0.5 * (J_sampling.transpose(0, 2, 1) @ invR_sampling @ J_sampling)

                nabla_mean_sampling = nabla_mean_sampling * w.reshape(-1, 1, 1)
                nabla_cov_sampling = nabla_cov_sampling * w.reshape(-1, 1, 1)

                nabla_mean_sampling = nabla_mean_sampling.reshape(factor_num, self.sampling_num, dima, 1)
                nabla_cov_sampling  = nabla_cov_sampling.reshape(factor_num, self.sampling_num, dima, dima)

                nabla_mean = np.mean(nabla_mean_sampling, axis=1)
                nabla_cov  = np.mean(nabla_cov_sampling, axis=1)

                gradient_batch[:, :dima] = nabla_mean - 2 * nabla_cov @ factor_mean
                gradient_batch[:, dima:] = nabla_cov.reshape(factor_num, -1, 1)

                for var in factor.adj_var.values():
                    if not isinstance(var.belief, Gaussian):
                        A_sampling = residual_sampling @ residual_sampling.transpose(0, 2, 1)
                        A_sampling = A_sampling * w.reshape(-1, 1, 1)
                        A_sampling = A_sampling.reshape(factor_num, self.sampling_num, dimz, dimz)
                        self.A[sensor_name] = np.mean(A_sampling, axis=1)
                        break

                if NN_mode:
                    self.iteration_NN[sensor_name] += 1

            flat = gradient_batch.reshape(factor_num, -1)

            for i, (factor_name, _, _, _) in enumerate(items):
                results[factor_name] = flat[i]
        
        ordered = [results[factor_name] for factor_name in self.factor_offsets.keys()]
        self.gradient = np.concatenate(ordered, axis=0).reshape(-1, 1)

    def compute_all_factor_messages_BP(self):
        """Compute internal factor-to-BP variable messages."""

        # internal factor-to-BP message = 
        #   g(L_1_factor_msg @ f (factor_natural_param))
        #       -L_2_factor_msg @ var_msg
        # g: expectation parameter to natural parameter (batch_mean_cov2_natural_param, mean and cov replace expectation parameter)
        # f: natural parameter to expectation parameter (batch_natural_param2mean_cov, mean and cov replace expectation parameter)

        mean_cov = batch_natural_param2mean_cov(self.factor_natural_param, self.factor_offsets, self.factor_groups)
        msg = batch_mean_cov2_natural_param(self.L_1_factor_msg @ mean_cov, self.intern_factor_msg_offsets_BP, self.intern_factor_msg_groups_BP)
        self.factor_msg_BP = np.zeros((self.dim["factor_msg_BP"],1))
        self.factor_msg_BP[0:self.dim["intern_factor_msg_BP"]] = msg - self.L_2_factor_msg @ self.var_msg

    def compute_all_factor_messages_MF(self):
        """Compute closed-form MF messages for all conjugate variables."""

        self.compute_gradient_A()

        sensor_name_keys = self.A.keys()
        count_A = {key: 0 for key in sensor_name_keys}
        self.factor_msg_MF = np.zeros((self.dim["factor_msg_MF"],1))
        count = 0
        for factor in self.fg.factor_node.values():
            if isinstance(factor.factor_param, MeasurementFactor):
                sensor_name = factor.factor_param.sensor_name
                if sensor_name in sensor_name_keys:
                    A_use = self.A[sensor_name][count_A[sensor_name]]
                    count_A[sensor_name] += 1
                    inv_cov_init = factor.factor_param.inv_cov_init
                    for var_name in factor.adj_var:
                        var = factor.adj_var[var_name]
                        if isinstance(var.belief, invWishart):
                            self.factor_msg_MF[count:count+1] = -0.5
                            count += 1

                            dim = A_use.shape[0]**2
                            self.factor_msg_MF[count:count+dim] = -0.5 * A_use.reshape(-1, 1)
                            count += dim

                        elif isinstance(var.belief, Bernoulli):
                            for var_Gamma in factor.adj_var.values():
                                if isinstance(var_Gamma.belief, Gamma):
                                    break
                            self.factor_msg_MF[count:count+1] = 0.5 * (A_use*factor.factor_param.inv_cov_init0*var_Gamma.belief.expectation()-var_Gamma.belief.ln_expectation()-A_use*inv_cov_init-np.log(factor.factor_param.inv_cov_init0)+np.log(inv_cov_init))
                            count += 1

                        elif isinstance(var.belief, Gamma):
                            for var_Bernoulli in factor.adj_var.values():
                                if isinstance(var_Bernoulli.belief, Bernoulli):
                                    break
                            self.factor_msg_MF[count:count+1] = 0.5 * (1-var_Bernoulli.belief.expectation())
                            count += 1

                            self.factor_msg_MF[count:count+1] = -0.5 *(1-var_Bernoulli.belief.expectation())*A_use*factor.factor_param.inv_cov_init0
                            count += 1

            elif isinstance(factor.factor_param, EF_PriorFactor):
                for var_name in factor.adj_var:
                    dim = len(factor.factor_param.prior_natural_param)
                    self.factor_msg_MF[count:count+dim] = factor.factor_param.prior_natural_param
                    count += dim

            elif isinstance(factor.factor_param, EF_ConnectFactor):
                vars_list = list(factor.adj_var.values())

                for i, var in enumerate(vars_list):
                    another_var = vars_list[1-i]
                    if isinstance(var.belief, Beta):
                        self.factor_msg_MF[count:count+1] = another_var.belief.expectation()
                        count += 1

                        self.factor_msg_MF[count:count+1] = 1-another_var.belief.expectation()
                        count += 1

                    elif isinstance(var.belief, Bernoulli):
                        self.factor_msg_MF[count:count+1] = another_var.belief.ln_expectation()-another_var.belief.ln_1_expectation()
                        count += 1

    def update_all_var_belief(self):
        """Combine incoming BP and MF messages into variable beliefs."""

        self.var_natural_param = self.L_var_BP@self.factor_msg_BP + self.L_var_MF@self.factor_msg_MF

    def set_all_MF_var_belief(self):
        """Write updated natural parameters to local MF variable objects."""

        count = 0
        for var in self.fg.var_node.values():
            dim = len(var.belief.natural_param)
            if var.BP_MF == "MF":
                natural_param = self.var_natural_param[count : count+dim]
                var.belief.set_by_natural_param(natural_param)
            count += dim

    def set_all_BP_var_belief(self):
        """Write updated natural parameters to local BP variable objects."""

        count = 0
        for var in self.fg.var_node.values():
            dim = len(var.belief.natural_param)
            if var.BP_MF == "BP":
                natural_param = self.var_natural_param[count : count+dim]
                var.belief.set_by_natural_param(natural_param)
            count += dim
    
    def retract_all_BP_var_belief(self):
        """Retract BP means onto the manifold and transport covariances."""

        for var in self.fg.var_node.values():
            if var.BP_MF == "BP":
                var.tangent_point = composite_add(var.tangent_point,var.belief.mean)

                mean = np.zeros_like(var.belief.mean)
                J = composite_rightJaco(var.belief.mean)
                cov = J @ var.belief.cov @ J.T
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
                    orientation = np.stack(orientation_list, axis=0)
                    position = np.stack(position_list, axis=0)
                    factor.tangent_point = State(orientation, position)

                mean = np.zeros_like(factor.belief.mean)
                cov = np.zeros_like(factor.belief.cov)
                count = 0
                for var in factor.adj_var.values():
                    if var.BP_MF == "BP":
                        dimi = len(var.belief.mean)
                        mean[count:count+dimi] = var.belief.mean
                        cov[count:count+dimi, count:count+dimi] = var.belief.cov
                        count += dimi
                
                factor.belief.set_by_Mean_Cov(mean, cov)

def batch_natural_param2mean_cov(factor_natural_param, factor_offsets, groups):
    """Convert grouped Gaussian natural parameters to stacked mean and cov."""

    results = {}
    for (_, dim_natural), items in groups.items():
        d = int(((-1 + (1 + 4 * dim_natural) ** 0.5) / 2))

        rows = [np.arange(row_idx, row_idx + dim_natural) for _, row_idx, _, _ in items]
        rows = np.concatenate(rows)

        factor_num = len(items)
        factor_natural_param_batch = factor_natural_param[rows].reshape(factor_num, -1, 1)

        eta = factor_natural_param_batch[:, :d]
        Lambda = -2 * factor_natural_param_batch[:, d:].reshape(factor_num, d, d)

        cov = np.linalg.inv(Lambda)
        mean = cov @ eta

        flat = np.concatenate([mean.reshape(factor_num, -1), cov.reshape(factor_num, -1)], axis=1)
        for i, (factor_name, _, _, _) in enumerate(items):
            results[factor_name] = flat[i]

    ordered = [results[factor_name] for factor_name in factor_offsets.keys()]
    mean_cov = np.concatenate(ordered, axis=0)
    return mean_cov.reshape(-1, 1)

def batch_mean_cov2_natural_param(mean_cov, msg_offsets, groups):
    """Convert grouped Gaussian moments to stacked natural parameters."""

    results = {}
    for dim_natural, items in groups.items():
        d = int(((-1 + (1 + 4 * dim_natural) ** 0.5) / 2))

        rows = [np.arange(row_idx, row_idx + dim_natural) for _, _, row_idx in items]
        rows = np.concatenate(rows)

        factor_num = len(items)
        mean_cov_batch = mean_cov[rows].reshape(factor_num, -1, 1)

        mean = mean_cov_batch[:, :d]
        cov  = mean_cov_batch[:, d:].reshape(factor_num, d, d)

        Lambda = np.linalg.inv(cov)
        eta = Lambda @ mean

        flat = np.concatenate([eta.reshape(factor_num, -1),
                               -0.5 * Lambda.reshape(factor_num, -1)], axis=1)
        for i, (factor_name, var_name, _) in enumerate(items):
            results[factor_name, var_name] = flat[i]

    ordered = [results[factor_name, var_name] for (factor_name, var_name) in msg_offsets.keys()]
    natural_param = np.concatenate(ordered, axis=0)
    return natural_param.reshape(-1, 1)


def cov_features(cov):
    """Encode covariance scale and correlation for an NF condition vector."""

    sigma = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    log_sigma = np.log(sigma + 1e-8)

    denom = sigma[:, :, None] * sigma[:, None, :]
    R = cov / (denom + 1e-8)

    idx = np.triu_indices(R.shape[-2], k=1)
    r_vals = R[:, idx[0], idx[1]]
    fisher_z = np.atanh(r_vals)

    return np.concatenate([log_sigma, fisher_z], axis=-1)


def transfer_var_msg(fg_solver_list):
    """Copy cross-robot variable messages to the solver owning each factor."""
    # Cross-robot variables are all BP variables in the considered application.

    for fg_solver in fg_solver_list:
        fg = fg_solver.fg
        for var in fg.extern_var_node.values():
            if var.BP_MF == "BP":
                for factor_name in var.adj_factor:
                    if factor_name in fg.factor_node:
                        extern_fg_solver = fg.extern_var_node_fg[(factor_name,var.var_name)].solver
                        row, dim = extern_fg_solver.var_msg_offsets[(var.var_name, factor_name)]
                        msg = extern_fg_solver.var_msg[row:row+dim]
                        row, dim = fg_solver.var_msg_offsets[(var.var_name, factor_name)]
                        fg_solver.var_msg[row:row+dim] = msg

def transfer_factor_msg(fg_solver_list):
    """Copy cross-robot factor messages to the solver owning each variable."""
    # Cross-robot variables are all BP variables in the considered application.

    for fg_solver in fg_solver_list:
        fg = fg_solver.fg
        for factor in fg.extern_factor_node.values():
            if factor.contain_BP_var:
                for var_name in factor.adj_var:
                    if factor.adj_var[var_name].BP_MF == "BP" and var_name in fg.var_node:
                        extern_fg_solver = fg.extern_factor_node_fg[(var_name,factor.factor_name)].solver
                        row, dim = extern_fg_solver.factor_msg_offsets_BP[(factor.factor_name, var_name)]
                        msg = extern_fg_solver.factor_msg_BP[row:row+dim]
                        row, dim = fg_solver.factor_msg_offsets_BP[(factor.factor_name, var_name)]
                        fg_solver.factor_msg_BP[row:row+dim] = msg
                    
