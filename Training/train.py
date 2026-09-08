import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

from FactorGraph import FactorGraph
from Distribution import Gaussian, invWishart, Gamma, Beta, Bernoulli
from Factor import MeasurementFactor, EF_PriorFactor, EF_ConnectFactor
from MeasurementModel import prior, odo, gnss, uwb
from FactorSolver import fg_solver
from Transform import (
    State,
    rotvec2rotm,
    tform,
    tform2rotm,
    tform2trans,
    composite_add,
    composite_minus,
    composite_rightJaco,
    inv_rightJaco,
)
from normflow import RealNVP

PROJECT_ROOT = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Loss(nn.Module):
    """Weighted trajectory-estimation loss over all solver iterations."""

    def forward(
        self,
        savedata,
        max_iteration,
        batch_size,
        train_interval,
        n_robot,
        dim_state,
        device,
    ):
        diff = torch.zeros(
            (max_iteration, batch_size, train_interval, n_robot, dim_state, 1),
            dtype=torch.float32,
            device=device,
        )
        for times in range(max_iteration):
            for k in range(train_interval):
                for n in range(n_robot):
                    diff[times, :, k, n] = composite_minus(
                        savedata["x"][times][k][n], savedata["x_esti"][times][k][n]
                    )

        squared_norm = torch.norm(diff.squeeze(-1), dim=4) ** 2
        error_2_per_batch_per_iter = torch.mean(squared_norm, dim=(2, 3))
        error_per_iter = torch.sqrt(
            torch.mean(error_2_per_batch_per_iter, dim=1)
        )

        iters = (
            torch.arange(max_iteration, device=device, dtype=torch.float32)
            .unsqueeze(-1)
            .expand(-1, batch_size)
        )
        weights_iter = torch.exp(
            -0.05 * (max_iteration - 1 - iters)
        )  # weights for each iteration

        loss_per_batch_per_iter = weights_iter * error_2_per_batch_per_iter
        loss_per_batch = torch.sum(
            loss_per_batch_per_iter, dim=0
        )  # for evaluating max and min

        loss = torch.mean(loss_per_batch)

        formatted = [round(v, 4) for v in error_per_iter.tolist()]
        print(
            f"Estimation error: {formatted}"
        )

        return loss, loss_per_batch.detach().cpu()


def obtain_state_estimate(
    batch_size,
    state_start,
    state_end,
    n_robot,
    d,
    fg,
    var_natural_param
):
    """Recover retracted state estimates and covariances from packed beliefs."""

    x_esti = [[None for _ in range(n_robot)] for _ in range(state_end - state_start + 1)]
    cov_esti = torch.zeros(batch_size, state_end-state_start+1, n_robot, d, d, dtype=torch.float32, device=device,)

    count = 0

    for var in fg.var_node.values():
        dim = var.belief.natural_param.shape[1]
        if str(var.var_name).startswith("x") and var.var_name[1].isdigit() \
            and var.time_step >= state_start and var.time_step <= state_end:

            if var_natural_param != None:
                natural_param = var_natural_param[:, count : count + dim]

                gaussian = Gaussian()
                gaussian.set_by_natural_param(natural_param)

                mean = gaussian.mean
                cov = gaussian.cov
            else:
                mean = var.belief.mean
                cov = var.belief.cov

            timestep = var.time_step
            n = var.robot_ID

            tangent_point = var.tangent_point

            x_esti[timestep-state_start][n-1] = composite_add(tangent_point, mean)

            J = composite_rightJaco(mean)
            cov_esti[:, timestep-state_start, n-1] = J @ cov @ J.transpose(1, 2)
        count += dim

    return x_esti, cov_esti


def main():
    """Generate training data, unroll inference, and optimize the NF."""

    save_path = str(PROJECT_ROOT)
    writer = SummaryWriter(log_dir=save_path)

    tb = program.TensorBoard()
    tb.configure(argv=[None, f"--logdir={save_path}", "--port=6006"])
    url = tb.launch()
    print(f"TensorBoard started at {url}")

    # Training configuration. These values are unchanged from the source code.
    dim_state = 6
    n_state = 100
    window_size = 3
    batch_size = 64
    data_size = 10 * batch_size
    iteration_num = 20
    sample_num = 16
    nonlinear_approx_mode = "NN"
    state_space = "Lie_Group"
    max_iteration = 5
    train_interval = 5

    with torch.no_grad():
        cov_single = {}
        cov_single["P_odo"] = torch.diag(
            torch.tensor([1e-6] * 3 + [1e-2] * 3, dtype=torch.float32, device=device)
        )
        cov_single["P_prior"] = torch.diag(
            torch.tensor([1e-4] * 3 + [1e-1] * 3, dtype=torch.float32, device=device)
        )
        cov_single["P_gnss"] = 1.0 * torch.eye(3, dtype=torch.float32, device=device)
        cov_single["P_uwb"] = 1e-2 * torch.eye(1, dtype=torch.float32, device=device)

        data_cov = {}
        data_cov["P_odo"] = cov_single["P_odo"].expand(data_size, -1, -1)
        data_cov["P_prior"] = cov_single["P_prior"].expand(data_size, -1, -1)
        data_cov["P_gnss"] = cov_single["P_gnss"].expand(data_size, -1, -1)
        data_cov["P_uwb"] = cov_single["P_uwb"].expand(data_size, -1, -1)

        cov = {}
        cov["P_odo"] = cov_single["P_odo"].expand(batch_size, -1, -1)
        cov["P_prior"] = cov_single["P_prior"].expand(batch_size, -1, -1)
        cov["P_gnss"] = cov_single["P_gnss"].expand(batch_size, -1, -1)
        cov["P_uwb"] = cov_single["P_uwb"].expand(batch_size, -1, -1)

        L = {}
        L["P_odo"] = torch.linalg.cholesky(data_cov["P_odo"])
        L["P_prior"] = torch.linalg.cholesky(data_cov["P_prior"])
        L["P_gnss"] = torch.linalg.cholesky(data_cov["P_gnss"])
        L["P_uwb"] = torch.linalg.cholesky(data_cov["P_uwb"])

        noise_param = {}
        noise_param["P_odo_t"] = torch.tensor([8], dtype=torch.float32, device=device)
        noise_param["P_odo_T"] = noise_param["P_odo_t"] * torch.diag(torch.tensor([1e-6]*3 + [1e-2]*3, dtype=torch.float32, device=device))
        noise_param["P_odo_t"] = noise_param["P_odo_t"].expand(batch_size, 1, 1)
        noise_param["P_odo_T"] = noise_param["P_odo_T"].expand(batch_size, -1, -1)

        noise_param["pi_uwb"] = torch.tensor([0.8], dtype=torch.float32, device=device)
        noise_param["y_uwb"] = torch.tensor([1], dtype=torch.float32, device=device)
        noise_param["xi_uwb"] = torch.tensor([7], dtype=torch.float32, device=device)
        noise_param["pi_uwb"] = noise_param["pi_uwb"].expand(batch_size, 1, 1)
        noise_param["y_uwb"] = noise_param["y_uwb"].expand(batch_size, 1, 1)
        noise_param["xi_uwb"] = noise_param["xi_uwb"].expand(batch_size, 1, 1)
        noise_param["P_uwb_0"] = 1*cov["P_uwb"]

        rho = 0.99

        ext_param = defaultdict(dict)
        ext_param["gnss"]["t"] = torch.tensor(
            [[0.6], [0], [-0.2]], dtype=torch.float32, device=device
        )
        ext_param["uwb"]["t"] = torch.tensor(
            [[0], [0], [-0.4]], dtype=torch.float32, device=device
        )

    # Build one iteration-specific UWB proposal for each solver iteration.
    torch.manual_seed(0)

    NN = {}
    NN["uwb"] = []
    dim_x = 6

    for _ in range(max_iteration):
        dim_meas = 1
        dim_a = 2 * dim_x
        dim_cond = dim_x + dim_meas + dim_a + (dim_a-1)*dim_a//2 + dim_meas + (dim_meas-1)*dim_meas//2
        NN["uwb"].append(RealNVP(dim_a, dim_cond, 4, device).to(device))

    all_params = []
    for iter in range(max_iteration):
        all_params += list(NN["uwb"][iter].parameters())

    optimizer = optim.Adam(all_params, lr=1.0 * 10 ** (-3), weight_decay=1e-4)
    train_iteration_last = 0

    loss = Loss()

    loss_history = []
    error_history = []

    # Generate the complete synthetic dataset once, before optimization.
    with torch.no_grad():
        n_robot = torch.tensor(4)

        data_x = [[None for _ in range(n_robot)] for _ in range(n_state)]
        data_prior = [None for _ in range(n_robot)]
        data_meas = {}
        data_meas["odo"] = [[None for _ in range(n_robot)] for _ in range(n_state)]
        data_meas["gnss"] = torch.zeros(data_size, n_state, n_robot, 3, 1, device=device)
        data_meas["uwb"] = torch.full((data_size, n_state, n_robot, n_robot, 1, 1), float("nan"), device=device)

        for n in range(n_robot):
            ori = torch.pi * (torch.rand((data_size, 3, 1), dtype=torch.float32, device=device) * 2-1)
            pos = 10 * (torch.rand((data_size, 3, 1), dtype=torch.float32, device=device) * 2-1)
            data_x[0][n] = State(rotvec2rotm(ori), pos)
            T = tform(data_x[0][n].orientation, data_x[0][n].position)

            noise = torch.randn((data_size, 6, 1), dtype=torch.float32, device=device)
            data_prior[n] = composite_add(data_x[0][n], L["P_prior"] @ noise)

            for k in range(1, n_state):
                ori = 0.01* torch.pi * (torch.rand((data_size, 3, 1), dtype=torch.float32, device=device) * 2-1)
                pos = torch.rand((data_size, 3, 1), dtype=torch.float32, device=device) * 2-1
                delta_T = tform(rotvec2rotm(ori), pos)
                T = T @ delta_T
                data_x[k][n] = State(tform2rotm(T), tform2trans(T))
            
                ww = 10**(2*torch.rand(data_size, 1, 1, dtype=torch.float32, device=device).expand(-1, dim_state, -1)-3)
                ww = torch.sqrt(ww)
                noise = torch.sqrt((1+0.1*torch.sin(torch.tensor(k)/20))) * ww *L["P_odo"] @ torch.randn(data_size, dim_state, 1, dtype=torch.float32, device=device)
                data_meas["odo"][k][n] = composite_add(State(rotvec2rotm(ori),pos),noise)

                meas_gt = -gnss.model_fcn([data_x[k][n]], torch.zeros((data_size,3,1), dtype = torch.float32, device=device), [ext_param["gnss"]])
                data_meas["gnss"][:, k, n] = meas_gt + L["P_gnss"] @ torch.randn(data_size, 3, 1, dtype=torch.float32, device=device)

        for k in range(1, n_state):
            for n in range(n_robot):
                for m in range(n_robot):
                    if torch.rand(1, dtype=torch.float32) >= 0 and n != m:
                        mask = (torch.rand(data_size, 1, 1, device=device) < 0.05).float()

                        base_noise = L["P_uwb"] @ torch.randn(data_size, 1, 1, dtype=torch.float32, device=device)

                        noise_uwb = base_noise * (1 - mask) + (10 * base_noise) * mask

                        meas_gt = -uwb.model_fcn([data_x[k][n],data_x[k][m]], torch.zeros((data_size,1,1), dtype = torch.float32, device=device), [ext_param["uwb"],ext_param["uwb"]])
                        data_meas["uwb"][:, k, n, m] = meas_gt + noise_uwb

    perm = torch.randperm(data_size, device=device)[:batch_size]
    global_step = 0
    for _ in range(iteration_num):
        # Retain 95% of the previous mini-batch and replace the remainder.
        with torch.no_grad():
            keep_size = int(batch_size * 0.95)
            replace_size = batch_size - keep_size

            rand_idx = torch.randperm(batch_size, device=device)
            keep_idx = perm[rand_idx[:keep_size]]

            available_mask = torch.ones(data_size, dtype=torch.bool, device=device)
            available_mask[perm] = False
            available_idx = torch.arange(data_size, device=device)[available_mask]

            new_idx = available_idx[
                torch.randperm(available_idx.numel(), device=device)[:replace_size]
            ]

            perm = torch.cat([keep_idx, new_idx], dim=0)
            perm = perm[torch.randperm(batch_size, device=device)]

            x = [[None for _ in range(n_robot)] for _ in range(n_state)]
            x_prior = [[None for _ in range(n_robot)] for _ in range(n_state)]
            for n in range(n_robot):
                for k in range(n_state):
                    ori = data_x[k][n].orientation[perm]
                    pos = data_x[k][n].position[perm]
                    x[k][n] = State(ori, pos)

                ori = data_prior[n].orientation[perm]
                pos = data_prior[n].position[perm]
                x_prior[0][n] = State(ori, pos)

            cov_prior = torch.zeros(batch_size, n_state, n_robot, dim_state, dim_state, device=device)
            cov_prior[:, 0, :, :] = cov_single["P_prior"].expand(batch_size, n_robot, -1, -1)
            meas = {}
            meas["odo"] = [[None for _ in range(n_robot)] for _ in range(n_state)]
            for n in range(n_robot):
                for k in range(1,n_state):
                    ori = data_meas["odo"][k][n].orientation[perm]
                    pos = data_meas["odo"][k][n].position[perm]
                    meas["odo"][k][n] = State(ori, pos)
            meas["gnss"] = data_meas["gnss"][perm]
            meas["uwb"] = data_meas["uwb"][perm]

        # Construct the initial factor graph for this training epoch.
        fg = FactorGraph()

        # Add the initial robot states and their prior factors.
        for n in range(n_robot):
            var_name = f"x0,{n+1}"
            belief = Gaussian()
            belief.set_by_Mean_Cov(torch.zeros((batch_size, 6, 1),dtype=torch.float32, device=device), cov_prior[:, 0, n])
            fg.add_var_node(var_name, belief, x_prior[0][n], 0, n+1, "BP")

            factor_name = f"prior0,{n+1}"
            factor_param = MeasurementFactor("prior", torch.zeros((batch_size, 6, 1),dtype=torch.float32, device=device), prior, cov_prior[:, 0, n], None, True)

            fg.add_factor_node(factor_name, [var_name], factor_param, NN)

        x_esti, cov_esti = obtain_state_estimate(batch_size, 0, 0, n_robot, dim_state, fg, None)

        for k in range(1, n_state):
            for nets in NN.values():
                for net in nets:
                    net.train()

            if k == 1 or k % train_interval == 0:
                # Start a new truncated backpropagation interval.
                savedata = {
                "x": [[[None for _ in range(n_robot)] for _ in range(train_interval)] for _ in range(max_iteration)],
                "x_esti": [[[None for _ in range(n_robot)] for _ in range(train_interval)] for _ in range(max_iteration)],
                }

            if k>1 and k<window_size:
                for n in range(n_robot):
                    var_name = f"x0,{n+1}"
                    factor_name = f"prior{0},{n+1}"
                    fg.delete_factor_node(factor_name)

                    mean = composite_minus(x_prior[0][n], fg.var_node[var_name].tangent_point)
                    invJ = torch.eye(6, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
                    invJ[:,0:3,0:3] = inv_rightJaco(mean[:,0:3])
                    cov_new = cov_prior[:, 0, n].clone()
                    cov_new = invJ @ cov_new @ invJ.transpose(1,2)
                    factor_param = MeasurementFactor("prior", mean, prior, cov_new, None, True)
                    fg.add_factor_node(factor_name, [var_name], factor_param, NN)

            if k >= window_size:
                for n in range(n_robot):
                    # Remove variables and factors outside the sliding window.
                    var_name = f"x{k-window_size},{n+1}"
                    fg.delete_var_node(var_name)

                    var_name_P_odo = f"P_odo{k-window_size},{k+1-window_size},{n+1}"
                    fg.delete_var_node(var_name_P_odo)

                    factor_name = f"gnss{k+1-window_size},{n+1}"
                    fg.delete_factor_node(factor_name)

                    # Replace the removed history by a Gaussian prior.
                    var_name = f"x{k+1-window_size},{n+1}"
                    factor_name = f"prior{k+1-window_size},{n+1}"
                    mean = composite_minus(x_prior[k+1-window_size][n], fg.var_node[var_name].tangent_point)
                    invJ = torch.eye(6, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
                    invJ[:,0:3,0:3] = inv_rightJaco(mean[:,0:3])
                    cov_new = cov_prior[:, k+1-window_size, n].clone()
                    cov_new = invJ @ cov_new @ invJ.transpose(1,2)
                    factor_param = MeasurementFactor("prior", mean, prior, cov_new, None, True)
                    fg.add_factor_node(factor_name, [var_name], factor_param, NN)

                for n in range(n_robot):
                    for m in range(n_robot):
                        if not torch.isnan(
                            meas["uwb"][:, k + 1 - window_size, n, m]
                        ).any():
                            var_name_pi = f"pi{k+1-window_size},{n+1},{m+1}"
                            fg.delete_var_node(var_name_pi)

                            var_name_y = f"y{k+1-window_size},{n+1},{m+1}"
                            fg.delete_var_node(var_name_y)

                            var_name_xi = f"xi{k+1-window_size},{n+1},{m+1}"
                            fg.delete_var_node(var_name_xi)

            for n in range(n_robot):
                # Propagate the adaptive odometry covariance prior.
                if k > 1:
                    var_name_P_odo_last = f"P_odo{k-2},{k-1},{n+1}"
                    P_odo_t = rho * fg.var_node[var_name_P_odo_last].belief.t
                    P_odo_T = rho * fg.var_node[var_name_P_odo_last].belief.T
                else:
                    P_odo_t = noise_param["P_odo_t"]
                    P_odo_T = noise_param["P_odo_T"]
                P_odo_set = P_odo_T / P_odo_t

                # Predict and add the next state variable.
                x_esti_T = tform(x_esti[-1][n].orientation, x_esti[-1][n].position)
                odo_T = tform(meas["odo"][k][n].orientation, meas["odo"][k][n].position)
                x_next_T = x_esti_T @ odo_T
                x_next = State(tform2rotm(x_next_T), tform2trans(x_next_T))

                J = odo.Jaco_fcn([x_esti[-1][n],x_next], meas["odo"][k][n], None)
                G = J[:,0:6,0:6]
                invF = torch.linalg.inv(J[:,0:6,6:12])
                pred_cov = invF @ (G @ cov_esti[:, -1, n] @ G.transpose(1,2)+P_odo_set) @ invF.transpose(1,2)

                var_name1 = f"x{k-1},{n+1}"
                var_name2 = f"x{k},{n+1}"
                belief = Gaussian()
                belief.set_by_Mean_Cov(torch.zeros((batch_size, 6, 1),dtype=torch.float32, device=device), pred_cov)
                fg.add_var_node(var_name2, belief, x_next, k, n + 1, "BP")

                # Add the odometry covariance variable, prior, and factor.
                var_name_P_odo = f"P_odo{k-1},{k},{n+1}"
                belief = invWishart()
                belief.set(P_odo_t, P_odo_T)
                fg.add_var_node(var_name_P_odo, belief, None, k, n + 1, "MF")

                factor_param = EF_PriorFactor(belief.natural_param)
                factor_name = f"P_odo_prior{k-1},{k},{n+1}"
                fg.add_factor_node(factor_name, [var_name_P_odo], factor_param, NN)

                factor_name = f"odo{k-1},{k},{n+1}"
                factor_param = MeasurementFactor("odo", meas["odo"][k][n], odo, P_odo_set, None, False)
                fg.add_factor_node(factor_name, [var_name1, var_name2, var_name_P_odo], factor_param, NN)

                # Add the GNSS factor for the new state.
                var_name = f"x{k},{n+1}"
                factor_name = f"gnss{k},{n+1}"
                factor_param = MeasurementFactor("gnss", meas["gnss"][:, k, n], gnss, cov["P_gnss"], [ext_param["gnss"]], False)
                fg.add_factor_node(factor_name, [var_name], factor_param, NN)

            # Add robust directed UWB factors and their MF noise variables.
            for n in range(n_robot):
                for m in range(n_robot):
                    if not torch.isnan(meas["uwb"][:, k, n, m]).any():
                        var_name_pi = f"pi{k},{n+1},{m+1}"
                        belief = Beta()
                        belief.set(noise_param["pi_uwb"], 1-noise_param["pi_uwb"])
                        fg.add_var_node(var_name_pi, belief, None, k, n+1, "MF")

                        factor_param = EF_PriorFactor(belief.natural_param)
                        factor_name = f"pi_uwb_prior{k},{n+1},{m+1}"
                        fg.add_factor_node(factor_name, [var_name_pi], factor_param, NN)

                        var_name_y = f"y{k},{n+1},{m+1}"
                        belief = Bernoulli()
                        belief.set(noise_param["y_uwb"])
                        fg.add_var_node(var_name_y, belief, None, k, n+1, "MF")

                        factor_param = EF_ConnectFactor()
                        factor_name = f"y_uwb_prior{k},{n+1},{m+1}"
                        fg.add_factor_node(factor_name, [var_name_pi, var_name_y], factor_param, NN)

                        var_name_xi = f"xi{k},{n+1},{m+1}"
                        belief = Gamma()
                        belief.set(noise_param["xi_uwb"]/2, noise_param["xi_uwb"]/2)
                        fg.add_var_node(var_name_xi, belief, None, k, n+1, "MF")

                        factor_param = EF_PriorFactor(belief.natural_param)
                        factor_name = f"xi_uwb_prior{k},{n+1},{m+1}"
                        fg.add_factor_node(factor_name, [var_name_xi], factor_param, NN)

                        var_name1 = f"x{k},{n+1}"
                        var_name2 = f"x{k},{m+1}"

                        factor_name = f"uwb{k},{n+1},{m+1}"
                        factor_param = MeasurementFactor("uwb", meas["uwb"][:, k, n, m], uwb, cov["P_uwb"], [ext_param["uwb"],ext_param["uwb"]], False)
                        factor_param.inv_cov_init0 = torch.linalg.inv(noise_param["P_uwb_0"])
                        fg.add_factor_node(factor_name, [var_name1, var_name2, var_name_y, var_name_xi], factor_param, NN)

            # Unroll all BP-MF iterations and retain estimates for the loss.
            MP_solver = fg_solver(fg, nonlinear_approx_mode, state_space, sample_num)

            for iteration in range(max_iteration):
                MP_solver.solve_one_step()

                k_start = max(k//train_interval*train_interval,k-window_size+1)
                x_esti, cov_esti = obtain_state_estimate(batch_size, k_start, k, n_robot, dim_state, fg, MP_solver.var_natural_param)

                count_j = 0
                for j in range(k//train_interval*train_interval, k+1):
                    if j>=k_start:
                        savedata["x"][iteration][j-k//train_interval*train_interval] = x[j]
                        for m in range(n_robot):
                            ori = x_esti[count_j][m].orientation.clone()
                            pos = x_esti[count_j][m].position.clone()
                            savedata["x_esti"][iteration][j-k//train_interval*train_interval][m] = State(ori,pos)
                        count_j += 1

            MP_solver.set_all_BP_var_belief()
            MP_solver.retract_all_BP_var_belief()
            MP_solver.set_all_BP_factor_belief()
            fg.detach_all_belief()

            for j in range(len(x_esti)):
                for m in range(n_robot):
                    ori = x_esti[j][m].orientation.detach().clone()
                    pos = x_esti[j][m].position.detach().clone()
                    x_esti[j][m] = State(ori, pos)
            cov_esti = cov_esti.detach()
            x_prior[k] = x_esti[-1]
            cov_prior[:, k] = cov_esti[:, -1]

            # Backpropagate once at the end of each truncated interval.
            if (k + 1) % train_interval == 0:
                optimizer.zero_grad()

                loss_val, error_vec = loss(
                    savedata,
                    max_iteration,
                    batch_size,
                    train_interval,
                    n_robot,
                    dim_state,
                    device,
                )
                loss_val.backward()

                optimizer.step()

                writer.add_scalar("Loss/min", min(error_vec), global_step)
                writer.add_scalar("Loss/max", max(error_vec), global_step)
                writer.add_scalar("Loss/mean", loss_val.item(), global_step)
                global_step += 1

                loss_history.append(loss_val.item())
                error_history.append(error_vec.numpy())

                print(
                    f"Iteration {global_step+train_iteration_last}, Loss={loss_val.item():.4f}"
                )

                # Save a resumable checkpoint and periodically export ONNX.
                checkpoint = {
                    "iteration": global_step + train_iteration_last,
                    "models": {
                        "uwb": [net.state_dict() for net in NN["uwb"]],
                    },
                    "optimizer": optimizer.state_dict(),
                    "loss_history": loss_history,
                    "error_history": error_history,
                }
                torch.save(checkpoint, os.path.join(save_path, "checkpoint.pth"))

                if global_step % 5 == 0:
                    dummy_sample = torch.randn(1, dim_a, device=device)
                    dummy_cond = torch.randn(1, dim_cond, device=device)

                    for i, net in enumerate(NN["uwb"]):
                        onnx_path = os.path.join(save_path, f"uwb_iter{i+1}.onnx")

                        net.eval()

                        torch.onnx.export(
                            net,
                            (dummy_sample, dummy_cond),
                            onnx_path,
                            input_names=["z", "cond"],
                            output_names=["eps", "logw"],
                            opset_version=17,
                            dynamic_axes={
                                "z": {0: "batch"},
                                "cond": {0: "batch"},
                                "eps": {0: "batch"},
                                "logw": {0: "batch"},
                            },
                        )

    writer.close()


if __name__ == "__main__":
    main()
