import time as pytime
from collections import defaultdict
import torch
import onnxruntime as ort

import matplotlib.pyplot as plt
import numpy as np

from FactorGraph import FactorGraph
from Distribution import Gaussian, invWishart, Gamma, Beta, Bernoulli
from Factor import MeasurementFactor, EF_PriorFactor, EF_ConnectFactor
from MeasurementModel import prior_R3, odo_R3, gnss_R3, uwb_R3
from FactorSolver import fg_solver, transfer_var_msg, transfer_factor_msg
from Transform import State, rotvec2rotm, tform, tform2rotm, tform2trans, composite_add, composite_minus, inv_rightJaco

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "Training" / "MF-UWB-R3"

def obtain_R_state_estimate(state_start, state_end, n_robot, d, fg_list):
    x_esti = np.zeros((state_end-state_start+1, n_robot, d, 1))
    cov_esti = np.zeros((state_end-state_start+1, n_robot, d, d))

    for n in range(n_robot):
        for var in fg_list[n].var_node.values():
            if str(var.var_name).startswith("x") and var.var_name[1].isdigit() \
                and var.time_step>=state_start and var.time_step<=state_end:

                timestep = var.time_step

                x_esti[timestep-state_start, n] = var.belief.mean
                cov_esti[timestep-state_start, n] = var.belief.cov

    return x_esti, cov_esti

def main():
    # Simulation configuration.
    dim_state = 3
    n_robot = 4
    n_state = 101
    window_size = 3
    Times = 50
    # nonlinear_approx_mode = "Taylor_Expansion"
    # nonlinear_approx_mode = "Sampling"
    nonlinear_approx_mode = "NN"
    state_space = "R3"
    # state_space = "Lie_Group"
    sample_num = 4
    max_iteration = 5

    # Nominal measurement and initial-state covariances.
    cov = {}
    cov["P_odo"] = 1e-2 * np.eye(3)
    cov["P_prior"] = 1e-1 * np.eye(3)
    cov["P_gnss"] = 1.0 * np.eye(3)
    cov["P_uwb"] = 1e-2 * np.eye(1)

    L = {}
    L["P_odo"] = np.linalg.cholesky(cov["P_odo"])
    L["P_prior"] = np.linalg.cholesky(cov["P_prior"])
    L["P_gnss"] = np.linalg.cholesky(cov["P_gnss"])
    L["P_uwb"] = np.linalg.cholesky(cov["P_uwb"])

    # Priors for the adaptive odometry and robust UWB noise models.
    noise_param = {}
    noise_param["P_odo_t"] = np.float64(5)
    noise_param["P_odo_T"] = 1*10**(-2) * noise_param["P_odo_t"] * np.eye(3)
    noise_param["pi_uwb"] = np.float64(0.8)
    noise_param["y_uwb"] = np.float64(1)
    noise_param["xi_uwb"] = np.float64(7)
    noise_param["P_uwb_0"] = 4*cov["P_uwb"]
    rho = 0.99

    # Sensor lever arms expressed in each robot's body frame.
    ext_param = defaultdict(dict)
    ext_param["gnss"]["t"] = [[0],[0],[0]]
    ext_param["uwb"]["t"] = [[0],[0],[0]]

    # NN
    NN = {}
    NN["uwb"] = []

    if nonlinear_approx_mode == "NN":
        for i in range(max_iteration):
            session = ort.InferenceSession(f"{MODEL_DIR}/uwb_iter{i+1}.onnx", providers=["CPUExecutionProvider"])

            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]

            NN["uwb"].append((session, input_names, output_names))

    # Preallocate Monte Carlo simulation outputs.
    plotdata = {
    "x": np.zeros((Times, n_state, n_robot, dim_state, 1)),
    "x_esti": np.zeros((Times, n_state, n_robot, dim_state, 1)),
    "time": np.zeros(1)
    }

    for times in range(Times):
        np.random.seed(times)

        # Generate Monte Carlo evaluation sequence.
        x = np.zeros((n_state, n_robot, dim_state, 1))
        x_prior = np.zeros((n_state, n_robot, dim_state, 1))
        cov_prior = np.zeros((n_state, n_robot, dim_state, dim_state))
        meas = {}
        meas["odo"] = np.zeros((n_state, n_robot, 3, 1))
        meas["gnss"] = np.zeros((n_state, n_robot, 3, 1))
        meas["uwb"] = np.full((n_state, n_robot, n_robot, 1, 1), float("nan"))
    
        for n in range(n_robot):
            x[0, n] = 10 * (np.random.rand(dim_state, 1) * 2-1)
            x_prior[0, n] = x[0, n] + L["P_prior"] @ np.random.randn(dim_state,1)
            cov_prior[0, n] = cov["P_prior"]
    
            for k in range(1, n_state):
                x[k, n] = x[k-1, n] + np.random.rand(dim_state, 1) * 2
                meas["odo"][k, n] = x[k, n] - x[k-1, n] + np.sqrt((1+0.1*np.sin(k/20)))* L["P_odo"] @ np.random.randn(dim_state, 1)
    
                meas_gt = -gnss_R3.model_fcn(x[k, n], np.zeros((3,1)), [ext_param["gnss"]])
                meas["gnss"][k, n] = meas_gt + L["P_gnss"] @ np.random.randn(3, 1)
    
        for k in range(1, n_state):
            for n in range(n_robot):
                for m in range(n_robot):
                    if np.random.rand(1) >= 0 and n != m:
                        if np.random.rand(1) < 0.05 and k>=(n_state-1)/5+1 and k<(n_state-1)/5*2+1:
                            noise_uwb = 10*L["P_uwb"] @ np.random.randn(1, 1)
                        elif np.random.rand(1) < 0.05 and k>=(n_state-1)/5*3+1 and k<(n_state-1)/5*4+1:
                            noise_uwb = 20*L["P_uwb"] @ np.random.randn(1, 1)
                        else:
                            noise_uwb = L["P_uwb"] @ np.random.randn(1, 1)
                        meas_gt = -uwb_R3.model_fcn(np.concatenate([x[k, n], x[k, m]],axis=0), np.zeros((1,1)), [ext_param["uwb"],ext_param["uwb"]])
                        meas["uwb"][k, n, m] = meas_gt + noise_uwb

        # Construct factor graph
        fg_list = [FactorGraph() for n in range(n_robot)]

        # Add state 0 node
        for n in range(n_robot):
            var_name = f"x0,{n+1}"
            belief = Gaussian()
            belief.set_by_Mean_Cov(x_prior[0, n], cov_prior[0, n])
            fg_list[n].add_var_node(var_name, belief, None, 0, n+1, "BP")

            factor_name = f"prior0,{n+1}"
            factor_param = MeasurementFactor("prior", x_prior[0, n], prior_R3, cov_prior[0, n], None, True)

            fg_list[n].add_factor_node(factor_name, [var_name], factor_param, NN)

        x_esti, cov_esti = obtain_R_state_estimate(0, 0, n_robot, dim_state, fg_list)

        for k in range(1, n_state):
            if k>=window_size:
                for n in range(n_robot):
                    # Delete states that have left the sliding window.
                    var_name = f"x{k-window_size},{n+1}"
                    fg_list[n].delete_var_node(var_name)

                    # Delete the oldest Odometry noise parameter factor.
                    var_name_P_odo = f"P_odo{k-window_size},{k+1-window_size},{n+1}"
                    fg_list[n].delete_var_node(var_name_P_odo)

                    # Delete the oldest GNSS factor.
                    factor_name = f"gnss{k+1-window_size},{n+1}"
                    fg_list[n].delete_factor_node(factor_name)

                    # Add prior factor.
                    var_name = f"x{k+1-window_size},{n+1}"
                    factor_name = f"prior{k+1-window_size},{n+1}"
                    factor_param = MeasurementFactor("prior", x_prior[k+1-window_size, n], prior_R3, cov_prior[k+1-window_size, n], None, True)
                    fg_list[n].add_factor_node(factor_name, [var_name], factor_param, NN)
                
                # Delete UWB noise parameters and factors that have left the window.
                for n in range(n_robot):
                    for m in range(n_robot):
                        if not np.isnan(meas["uwb"][k+1-window_size, n, m]):
                            var_name_pi = f"pi{k+1-window_size},{n+1},{m+1}"
                            fg_list[n].delete_var_node(var_name_pi)

                            var_name_y = f"y{k+1-window_size},{n+1},{m+1}"
                            fg_list[n].delete_var_node(var_name_y)

                            var_name_xi = f"xi{k+1-window_size},{n+1},{m+1}"
                            fg_list[n].delete_var_node(var_name_xi)
                    
            for n in range(n_robot):
                # Obtain the estimates of odometry covariances.
                if k > 1:
                    var_name_P_odo_last = f"P_odo{k-2},{k-1},{n+1}"
                    P_odo_t = rho*fg_list[n].var_node[var_name_P_odo_last].belief.t
                    P_odo_T = rho*fg_list[n].var_node[var_name_P_odo_last].belief.T
                else:
                    P_odo_t = noise_param["P_odo_t"]
                    P_odo_T = noise_param["P_odo_T"]
                P_odo_set = P_odo_T/P_odo_t
            
                # Predict and add the current robot state.
                var_name1 = f"x{k-1},{n+1}"
                var_name2 = f"x{k},{n+1}"
                belief = Gaussian()
                belief.set_by_Mean_Cov(x_esti[-1, n]+meas["odo"][k, n], cov_esti[-1, n]+P_odo_set)
                fg_list[n].add_var_node(var_name2, belief, None, k, n+1, "BP")

                # Add odometry factor and its adaptive covariance variable and noise factor.
                var_name_P_odo = f"P_odo{k-1},{k},{n+1}"
                belief = invWishart()
                belief.set(P_odo_t, P_odo_T)
                fg_list[n].add_var_node(var_name_P_odo, belief, None, k, n+1, "MF")

                factor_param = EF_PriorFactor(belief.natural_param)
                factor_name = f"P_odo_prior{k-1},{k},{n+1}"
                fg_list[n].add_factor_node(factor_name, [var_name_P_odo], factor_param, NN)
                
                factor_name = f"odo{k-1},{k},{n+1}"
                factor_param = MeasurementFactor("odo", meas["odo"][k, n], odo_R3, P_odo_set, None, True)
                fg_list[n].add_factor_node(factor_name, [var_name1, var_name2, var_name_P_odo], factor_param, NN)

                # Add the GNSS position factor.
                var_name = f"x{k},{n+1}"
                factor_name = f"gnss{k},{n+1}"
                factor_param = MeasurementFactor("gnss", meas["gnss"][k, n], gnss_R3, cov["P_gnss"], [ext_param["gnss"]], True)
                fg_list[n].add_factor_node(factor_name, [var_name], factor_param, NN)

            # Add robust inter-robot UWB factors and their GSTM variables.
            for n in range(n_robot):
                for m in range(n_robot):
                    if not np.isnan(meas["uwb"][k, n, m]):
                        var_name_pi = f"pi{k},{n+1},{m+1}"
                        belief = Beta()
                        belief.set(noise_param["pi_uwb"], 1-noise_param["pi_uwb"])
                        fg_list[n].add_var_node(var_name_pi, belief, None, k, n+1, "MF")

                        factor_param = EF_PriorFactor(belief.natural_param)
                        factor_name = f"pi_uwb_prior{k},{n+1},{m+1}"
                        fg_list[n].add_factor_node(factor_name, [var_name_pi], factor_param, NN)

                        var_name_y = f"y{k},{n+1},{m+1}"
                        belief = Bernoulli()
                        belief.set(noise_param["y_uwb"])
                        fg_list[n].add_var_node(var_name_y, belief, None, k, n+1, "MF")

                        factor_param = EF_ConnectFactor()
                        factor_name = f"y_uwb_prior{k},{n+1},{m+1}"
                        fg_list[n].add_factor_node(factor_name, [var_name_pi, var_name_y], factor_param, NN)

                        var_name_xi = f"xi{k},{n+1},{m+1}"
                        belief = Gamma()
                        belief.set(noise_param["xi_uwb"]/2, noise_param["xi_uwb"]/2)
                        fg_list[n].add_var_node(var_name_xi, belief, None, k, n+1, "MF")

                        factor_param = EF_PriorFactor(belief.natural_param)
                        factor_name = f"xi_uwb_prior{k},{n+1},{m+1}"
                        fg_list[n].add_factor_node(factor_name, [var_name_xi], factor_param, NN)

                        var_name1 = f"x{k},{n+1}"
                        var_name2 = f"x{k},{m+1}"

                        factor_name = f"uwb{k},{n+1},{m+1}"
                        factor_param = MeasurementFactor("uwb", meas["uwb"][k, n, m], uwb_R3, cov["P_uwb"], [ext_param["uwb"],ext_param["uwb"]], False)
                        factor_param.inv_cov_init0 = np.linalg.inv(noise_param["P_uwb_0"])
                        fg_list[n].add_cross_factor_node(factor_name, [var_name1, var_name_y, var_name_xi], [var_name2], [fg_list[m]], factor_param, NN)
                
            # Run distributed message passing and record the window estimates.
            start = pytime.time()
            MP_solver_list = [fg_solver(fg_list[n], nonlinear_approx_mode, state_space, sample_num) for n in range(n_robot)]

            for _ in range(max_iteration):
                for n in range(n_robot):
                    MP_solver_list[n].iteration += 1
                    if MP_solver_list[n].iteration > 1:
                        MP_solver_list[n].compute_all_BP_var_messages()

                transfer_var_msg(MP_solver_list)
                for n in range(n_robot):
                    MP_solver_list[n].compute_all_factor_messages_MF()
                    MP_solver_list[n].update_all_factor_belief()
                    MP_solver_list[n].compute_all_factor_messages_BP()

                transfer_factor_msg(MP_solver_list)
                for n in range(n_robot):
                    MP_solver_list[n].update_all_var_belief()
                    MP_solver_list[n].set_all_MF_var_belief()
                
            for n in range(n_robot):
                MP_solver_list[n].set_all_BP_var_belief()
            for n in range(n_robot):
                MP_solver_list[n].set_all_BP_factor_belief()
                
            plotdata["time"] += pytime.time() - start

            x_esti, cov_esti = obtain_R_state_estimate(max(0, k+1-window_size), k, n_robot, dim_state, fg_list)
            plotdata["x"][times] = x.copy()
            plotdata["x_esti"][times, range(max(0, k+1-window_size), k+1)] = x_esti.copy()
        
            x_prior[k] = x_esti[-1]
            cov_prior[k] = cov_esti[-1]

    # Save output in the format used by the original plotting scripts.
    np.savez(f'plotdata-MF-{nonlinear_approx_mode}-R3.npz', **plotdata)

    print(f"Avg. Time cost: {plotdata['time']/max_iteration/n_robot/Times/(n_state-1)*1000}ms")

    # Compute and display the original summary curves.
    diff = np.array(plotdata["x"]) - np.array(plotdata["x_esti"])
    squared_norm = np.linalg.norm(diff.squeeze(-1), axis=3) ** 2
    error_state = squared_norm.mean(axis=(0, 2))
    RMSE_state = np.sqrt(error_state)
    print(f"{nonlinear_approx_mode} ARMSE: {np.sqrt(np.mean(RMSE_state**2))}")

    plt.plot(range(n_state), RMSE_state)
    plt.xlabel("State")
    plt.ylabel("RMSE")
    plt.title("Root Mean Squared Error over Time")
    plt.show()

if __name__ == "__main__":
    main()
