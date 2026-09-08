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
from MeasurementModel import prior_R3, odo_R3, gnss_R3, uwb_R3
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
    def forward(self, savedata, max_iteration):
        B = savedata["x"].shape[1] # batch size

        diff = savedata["x"] - savedata["x_esti"]

        squared_norm = torch.norm(diff.squeeze(-1), dim=4) ** 2
        error_2_per_batch_per_iter = torch.mean(squared_norm, dim=(2, 3))
        error_per_iter = torch.sqrt(torch.mean(error_2_per_batch_per_iter, dim=1))
        
        iters = torch.arange(max_iteration, device=device, dtype=torch.float32).unsqueeze(-1).expand(-1, B)
        weights_iter = torch.exp(-0.05 * (max_iteration - 1 - iters)) # weights for each iteration

        loss_per_batch_per_iter = weights_iter * error_2_per_batch_per_iter
        loss_per_batch = torch.sum(loss_per_batch_per_iter, dim=0) # for evaluating max and min
        loss = torch.mean(loss_per_batch)
        
        formatted = [round(v, 4) for v in error_per_iter.tolist()]
        print(f"Estimation error: {formatted}")

        return loss, loss_per_batch.detach().cpu()


def obtain_R_state_estimate(batch_size, state_start, state_end, n_robot, d, fg):
    x_esti = torch.zeros(batch_size, state_end-state_start+1, n_robot, d, 1, dtype=torch.float32, device=device)
    cov_esti = torch.zeros(batch_size, state_end-state_start+1, n_robot, d, d, dtype=torch.float32, device=device)

    for var in fg.var_node.values():

        if str(var.var_name).startswith("x") and var.var_name[1].isdigit() \
            and var.time_step>=state_start and var.time_step<=state_end:

            timestep = var.time_step
            n = var.robot_ID

            x_esti[:, timestep-state_start, n-1] = var.belief.mean
            cov_esti[:, timestep-state_start, n-1] = var.belief.cov

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
    dim_state = 3
    n_state = 100
    window_size = 3
    batch_size = 64
    data_size = 10 * batch_size
    iteration_num = 20
    sample_num = 16
    nonlinear_approx_mode = "NN"
    state_space = "R3"
    max_iteration = 5
    train_interval = 10

    with torch.no_grad():
        cov_single = {}
        cov_single["P_odo"] = 1e-2 * torch.eye(3, dtype=torch.float32, device=device)
        cov_single["P_prior"] = 1e-1 * torch.eye(3, dtype=torch.float32, device=device)
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

        rho = 0.99

        ext_param = defaultdict(dict)
        ext_param["gnss"]["t"] = torch.tensor([[0],[0],[0]], dtype=torch.float32, device=device)
        ext_param["uwb"]["t"]  = torch.tensor([[0],[0],[0]], dtype=torch.float32, device=device)

    # Build one iteration-specific UWB proposal for each solver iteration.
    torch.manual_seed(0)

    NN = {}
    NN["uwb"] = []
    dim_x = 3

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
        
        data_x = torch.zeros(data_size, n_state, n_robot, dim_state, 1, device=device)
        data_prior = torch.zeros(data_size, n_robot, dim_state, 1, device=device)
        data_meas = {}
        data_meas["odo"] = torch.zeros(data_size, n_state, n_robot, 3, 1, device=device)
        data_meas["gnss"] = torch.zeros(data_size, n_state, n_robot, 3, 1, device=device)
        data_meas["uwb"] = torch.full((data_size, n_state, n_robot, n_robot, 1, 1), float("nan"), device=device)
    
        for n in range(n_robot):
            data_x[:, 0, n] = 10 * (torch.rand((data_size, dim_state, 1), dtype=torch.float32, device=device) * 2-1)
            data_prior[:, n] = data_x[:, 0, n] + L["P_prior"] @ torch.randn(data_size, dim_state, 1, dtype=torch.float32, device=device)
    
            for k in range(1, n_state):
                data_x[:, k, n] = data_x[:, k-1, n] + 1* (torch.rand((data_size, dim_state, 1), dtype=torch.float32, device=device) * 2-1)
                ww = 10**(2*torch.rand(data_size, 1, 1, dtype=torch.float32, device=device).expand(-1, dim_state, -1)-3)
                ww = torch.sqrt(ww)
                data_meas["odo"][:, k, n] = data_x[:, k, n] - data_x[:, k-1, n] + torch.sqrt((1+0.1*torch.sin(torch.tensor(k)/20))) * ww *(L["P_odo"] @ torch.randn(data_size, dim_state, 1, dtype=torch.float32, device=device))
    
                meas_gt = -gnss_R3.model_fcn(data_x[:, k, n], torch.zeros((data_size,3,1), dtype = torch.float32, device=device), [ext_param["gnss"]])
                data_meas["gnss"][:, k, n] = meas_gt + L["P_gnss"] @ torch.randn(data_size, 3, 1, dtype=torch.float32, device=device)
    
        for k in range(1, n_state):
            for n in range(n_robot):
                for m in range(n_robot):
                
                    if torch.rand(1, dtype=torch.float32) >= 0 and n != m:
                        mask = (torch.rand(data_size, 1, 1, device=device) < 0.05).float()
    
                        base_noise = L["P_uwb"] @ torch.randn(data_size, 1, 1, dtype=torch.float32, device=device)
    
                        noise_uwb = base_noise * (1 - mask) + (10 * base_noise) * mask
    
                        meas_gt = -uwb_R3.model_fcn(torch.cat([data_x[:, k, n],data_x[:, k, m]],dim=1), torch.zeros((data_size,1,1), dtype = torch.float32, device=device), [ext_param["uwb"],ext_param["uwb"]])
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

            x = data_x[perm]
            x_prior = torch.zeros(batch_size, n_state, n_robot, dim_state, 1, device=device)
            x_prior[:, 0] = data_prior[perm]
            cov_prior = torch.zeros(batch_size, n_state, n_robot, dim_state, dim_state, device=device)
            cov_prior[:, 0, :, :] = cov_single["P_prior"].expand(batch_size, n_robot, -1, -1)
            meas = {}
            meas["odo"] = data_meas["odo"][perm]
            meas["gnss"] = data_meas["gnss"][perm]
            meas["uwb"] = data_meas["uwb"][perm]

        # Construct the initial factor graph for this training epoch.
        fg = FactorGraph()

        # Add the initial robot states and their prior factors.
        for n in range(n_robot):
            var_name = f"x0,{n+1}"
            belief = Gaussian()
            belief.set_by_Mean_Cov(x_prior[:, 0, n], cov_prior[:, 0, n])
            fg.add_var_node(var_name, belief, None, 0, n+1, "BP")
    
            factor_name = f"prior0,{n+1}"
            factor_param = MeasurementFactor("prior", x_prior[:, 0, n], prior_R3, cov_prior[:, 0, n], None, True)
    
            fg.add_factor_node(factor_name, [var_name], factor_param, NN)

        x_esti, cov_esti = obtain_R_state_estimate(batch_size, 0, 0, n_robot, dim_state, fg)

        for k in range(1, n_state):
            for nets in NN.values():
                for net in nets:
                    net.train()

            if k == 1 or k % train_interval == 0:
                # Start a new truncated backpropagation interval.
                savedata = {
                "x": torch.zeros(max_iteration, batch_size, train_interval, n_robot, dim_state, 1, dtype=torch.float32, device=device),
                "x_esti": torch.zeros(max_iteration, batch_size, train_interval, n_robot, dim_state, 1, dtype=torch.float32, device=device),
                }

            if k >= window_size:
                for n in range(n_robot):
                    # Remove variables and factors outside the sliding window.
                    var_name = f"x{k-window_size},{n+1}"
                    fg.delete_var_node(var_name)

                    factor_name = f"gnss{k+1-window_size},{n+1}"
                    fg.delete_factor_node(factor_name)

                    # Replace the removed history by a Gaussian prior.
                    var_name = f"x{k+1-window_size},{n+1}"
                    factor_name = f"prior{k+1-window_size},{n+1}"
                    factor_param = MeasurementFactor("prior", x_prior[:, k+1-window_size, n], prior_R3, cov_prior[:, k+1-window_size, n], None, True)
                    fg.add_factor_node(factor_name, [var_name], factor_param, NN)

                for n in range(n_robot):
                    for m in range(n_robot):
                        if not torch.isnan(
                            meas["uwb"][:, k + 1 - window_size, n, m]
                        ).any():
                            factor_name = f"uwb{k+1-window_size},{n+1},{m+1}"
                            fg.delete_factor_node(factor_name)

            for n in range(n_robot):
                # Propagate the adaptive odometry covariance prior.
                P_odo_set = cov["P_odo"]

                # Predict and add the next state variable.
                var_name1 = f"x{k-1},{n+1}"
                var_name2 = f"x{k},{n+1}"
                belief = Gaussian()
                belief.set_by_Mean_Cov(x_esti[:, -1, n]+meas["odo"][:, k, n], cov_esti[:, -1, n]+P_odo_set)
                fg.add_var_node(var_name2, belief, None, k, n+1, "BP")

                # Add the odometry covariance variable, prior, and factor.
                factor_name = f"odo{k-1},{k},{n+1}"
                factor_param = MeasurementFactor("odo", meas["odo"][:, k, n], odo_R3, P_odo_set, None, True)
                fg.add_factor_node(factor_name, [var_name1, var_name2], factor_param, NN)

                # Add the GNSS factor for the new state.
                var_name = f"x{k},{n+1}"
                factor_name = f"gnss{k},{n+1}"
                factor_param = MeasurementFactor("gnss", meas["gnss"][:, k, n], gnss_R3, cov["P_gnss"], [ext_param["gnss"]], True)
                fg.add_factor_node(factor_name, [var_name], factor_param, NN)

            # Add robust directed UWB factors and their MF noise variables.
            for n in range(n_robot):
                for m in range(n_robot):
                    if not torch.isnan(meas["uwb"][:, k, n, m]).any():
                        var_name1 = f"x{k},{n+1}"
                        var_name2 = f"x{k},{m+1}"
    
                        factor_name = f"uwb{k},{n+1},{m+1}"
                        factor_param = MeasurementFactor("uwb", meas["uwb"][:, k, n, m], uwb_R3, cov["P_uwb"], [ext_param["uwb"],ext_param["uwb"]], False)
                        fg.add_factor_node(factor_name, [var_name1, var_name2], factor_param, NN)

            # Unroll all BP iterations and retain estimates for the loss.
            MP_solver = fg_solver(fg, nonlinear_approx_mode, state_space, sample_num)

            for iteration in range(max_iteration):
                MP_solver.solve_one_step()
                MP_solver.set_all_BP_var_belief()

                k_start = max(k//train_interval*train_interval,k-window_size+1)
                x_esti, cov_esti = obtain_R_state_estimate(batch_size, k_start, k, n_robot, dim_state, fg)

                count_j = 0
                for j in range(k//train_interval*train_interval, k+1):
                    if j>=k_start:
                        savedata["x"][iteration][:,j-k//train_interval*train_interval] = x[:,j]
                        savedata["x_esti"][iteration][:, j-k//train_interval*train_interval] = x_esti[:, count_j]
                        count_j += 1

            MP_solver.set_all_BP_factor_belief()

            fg.detach_all_belief()

            x_esti = x_esti.detach()
            cov_esti = cov_esti.detach()
            x_prior[:, k] = x_esti[:, -1].clone()
            cov_prior[:, k] = cov_esti[:, -1].clone()

            # Backpropagate once at the end of each truncated interval.
            if (k + 1) % train_interval == 0:
                optimizer.zero_grad()

                loss_val, error_vec = loss(savedata, max_iteration)
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
