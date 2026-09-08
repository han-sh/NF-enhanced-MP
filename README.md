# NF-enhanced-MP

This repository provides the training and evaluation code for **normalizing flow-enhanced parametric message passing (MP)** for distributed multi-robot cooperative localization. The framework unifies Gaussian belief propagation (GBP) for state estimation with mean-field (MF) inference for adaptive noise estimation, while directly operating in the natural parameter space of beliefs. Nonlinear and non-conjugate factors can be handled using Taylor linearization, Monte Carlo sampling, or a **learnable normalizing flow gradient estimator trained end-to-end with the MP algorithm**. The provided examples fuse odometry, GNSS, and inter-robot UWB measurements, supporting both Euclidean states and `SO(3) x R3` error states.


## Repository structure

```text
.
├── Evaluation/
│   ├── main.py              # BP–MF evaluation
│   ├── main-BP.py           # BP evaluation
│   ├── main-R3.py           # BP–MF evaluation in R3
│   ├── main-BP-R3.py        # BP evaluation in R3
│   ├── FactorGraph.py       # Factor graph data structures
│   ├── FactorSolver.py      # Batched MP solver
│   ├── Factor.py            # Factor parameter containers
│   ├── Distribution.py      # Exponential family distributions
│   ├── MeasurementModel.py  # Prior, odometry, GNSS, and UWB models
│   └── Transform.py         # SO(3) x R3 state operations and Jacobians
└── Training/
    ├── train.py             # Trains proposals for BP–MF
    ├── train-BP.py          # Trains proposals for BP
    ├── train-R3.py          # Trains proposals for BP–MF in R3
    ├── train-BP-R3.py       # Trains proposals for BP in R3
    ├── normflow.py          # Conditional RealNVP implementation
    ├── FactorGraph.py       
    ├── FactorSolver.py      
    ├── Factor.py
    ├── Distribution.py
    ├── MeasurementModel.py
    ├── Transform.py
    ├── MF-UWB/              # BP–MF ONNX proposals
    |── BP-UWB/              # BP ONNX proposals
    ├── MF-UWB-R3/           # BP–MF ONNX proposals in R3
    └── BP-UWB-R3/           # BP ONNX proposals in R3
```

## What the two evaluation entry points do

### `Evaluation/main.py`

`main.py` evaluates the proposed hybrid BP–MF estimator. Robot poses are Gaussian BP variables, while uncertain noise quantities are represented by conjugate MF variables. In the included experiment:

- an inverse-Wishart variable adapts the odometry covariance;
- Beta, Bernoulli, and Gamma variables form the robust UWB noise model;
- each robot owns a local factor graph;
- cross-robot UWB factors exchange messages between local graphs;
- a sliding window removes old states and factors;
- gradient updates associated with nonlinear UWB ranging can use Taylor expansion, Monto Carlo sampling, or normalizing flow;
- the script reports runtime and orientation/position ARMSE, saves an `.npz` result, and plots RMSE curves.

### `Evaluation/main-BP.py`

`main-BP.py` is the BP-only comparison. It uses the same simulated trajectories, sensors, sliding-window structure, distributed message transfers, nonlinear approximation choices, and evaluation metrics, but keeps the nominal measurement covariances fixed and omits the MF noise variables used for adaptive/robust inference. Its output file is named `plotdata-BP-<mode>.npz`, whereas the hybrid script writes `plotdata-MF-<mode>.npz`.

The corresponding training scripts follow the same split: `Training/train.py` trains the hybrid BP–MF, and `Training/train-BP.py` trains the BP. Both unroll factor-graph inference in PyTorch and export one ONNX proposal per solver iteration.

## Requirements

The code has been tested with:

- Python: 3.13.7
- NumPy: 2.3.3
- SciPy: 1.16.2
- Matplotlib: 3.10.6
- PyTorch: 2.8.0+cu129
- CUDA used by PyTorch: 12.9
- TensorBoard: 2.20.0
- ONNX Runtime: 1.24.2

## Quick start: evaluation

Run the scripts from the `Evaluation` directory:

```bash
cd Evaluation
python main.py
python main-BP.py
python main-R3.py
python main-BP-R3.py
```

The default configuration uses four robots, 101 states, a three-state sliding window, five message-passing iterations, and 30 Monte Carlo trials.

The main experiment settings are defined near the beginning of each entry point:

```python
dim_state = 6
n_robot = 4
n_state = 101
window_size = 3
Times = 30
nonlinear_approx_mode = "NN"   # "Taylor_Expansion", "Sampling", or "NN"
state_space = "Lie_Group"      # or "R3" with matching models and dimensions
sample_num = 4
max_iteration = 5
```

## Evaluation workflow

At each time step, the example performs the following operations:

1. Predict each robot state using odometry and add the new state variable.
2. Add odometry and GNSS factors to the robot's local graph.
3. Add directed inter-robot UWB factors and register their external variables.
4. For the hybrid method, add and update the conjugate variables associated with adaptive odometry and robust UWB noise.
5. Construct one `fg_solver` per robot for the current sliding-window topology.
6. Alternate variable-to-factor, MF factor-to-variable, BP factor-to-variable, and cross-robot message updates.
7. Write the BP beliefs back to the graph, retract pose increments onto the manifold, and record estimates as priors.

## Customizing the factor graph

### 1. Define a vectorized residual and Jacobian

A measurement model is a class exposing two static methods:

```python
class MySensor:
    @staticmethod
    def model_fcn(x, z, ext_param):
        """Return the residual."""
        ...

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        """Return the residual Jacobian with respect to adjacent BP states."""
        ...
```

For `Lie_Group`, `x` is a list of adjacent `State` objects. In evaluation, an unbatched state stores an orientation of shape `(3, 3)` and a position of shape `(3, 1)`; a batched state stores `(N, 3, 3)` and `(N, 3, 1)`. A factor with residual dimension `m` connected to `q` six-dimensional pose variables must return a batched residual of shape `(N, m, 1)` and a batched Jacobian of shape `(N, m, 6*q)`. The functions should also accept the corresponding unbatched inputs.

When using Lie-group states, `Jaco_fcn` should return the Jacobian with respect to the local right perturbation at the retracted state. The solver applies `composite_rightJaco(...)` to map it to the Gaussian error-state coordinates used internally.

### 2. Add variables

`FactorGraph.add_var_node(...)` registers a local variable:

```python
belief = Gaussian()
belief.set_by_Mean_Cov(np.zeros((6, 1)), initial_covariance)

fg.add_var_node(
    var_name="x0,1",
    belief=belief,
    tangent_point=initial_state,
    time_step=0,
    robot_ID=1,
    BP_MF="BP",
)
```

Use `BP_MF="BP"` for Gaussian state variables updated by belief propagation. Use `BP_MF="MF"` only for a supported conjugate variable such as `invWishart`, `Gamma`, `Beta`, or `Bernoulli`. For Euclidean states, use a Gaussian belief with `tangent_point=None` and make the dimensions consistent with the selected `R3` model.

### 3. Add a local measurement factor

Wrap the measurement model in `MeasurementFactor`, then connect it to existing variables:

```python
factor_param = MeasurementFactor(
    sensor_name="position_sensor",
    meas=z_position,
    model=position_sensor,
    cov=R_position,
    ext_param=None,
    if_Taylor_Expansion=True,
)

fg.add_factor_node(
    factor_name="position0,1",
    adj_var_name=["x0,1"],
    factor_param=factor_param,
    NN=NN,
)
```

`sensor_name` is also the batching key, so factors grouped under the same name must use compatible residuals, state arity, dimensions, measurements, and extrinsic-parameter structures. Setting `if_Taylor_Expansion=True` forces this factor to use the Taylor branch even when the solver is globally configured for sampling or a learned proposal.

### 4. Add a cross-robot factor

For a factor owned by one graph but connected to a variable in another robot's graph, use `add_cross_factor_node(...)`:

```python
factor_param = MeasurementFactor(
    "uwb",
    z_uwb,
    uwb,
    R_uwb,
    [uwb_extrinsic_robot_1, uwb_extrinsic_robot_2],
    False,
)

fg_list[0].add_cross_factor_node(
    factor_name="uwb1,1,2",
    self_adj_var_name=["x1,1"],
    extern_adj_var_name=["x1,2"],
    extern_fg=[fg_list[1]],
    factor_param=factor_param,
    NN=NN,
)
```

The order of adjacent variables must match the order expected by `model_fcn` and `Jaco_fcn`. Each name in `extern_adj_var_name` must have a corresponding graph at the same position in `extern_fg`.

### 5. Run distributed message passing

The evaluation example uses the following update order:

```python
solvers = [
    fg_solver(graph, nonlinear_approx_mode, state_space, sample_num)
    for graph in fg_list
]

for _ in range(max_iteration):
    for solver in solvers:
        solver.iteration += 1
        if solver.iteration > 1:
            solver.compute_all_BP_var_messages()

    transfer_var_msg(solvers)

    for solver in solvers:
        solver.compute_all_factor_messages_MF()
        solver.update_all_factor_belief()
        solver.compute_all_factor_messages_BP()

    transfer_factor_msg(solvers)

    for solver in solvers:
        solver.update_all_var_belief()
        solver.set_all_MF_var_belief()

for solver in solvers:
    solver.set_all_BP_var_belief()
    solver.retract_all_BP_var_belief()

for solver in solvers:
    solver.set_all_BP_factor_belief()
```

## Key interfaces

| Interface | Purpose |
|---|---|
| `FactorGraph.add_var_node` | Add a local BP or MF variable. |
| `FactorGraph.add_factor_node` | Add a factor whose adjacent variables are local. |
| `FactorGraph.add_cross_factor_node` | Add a locally owned factor that also references variables in other robot graphs. |
| `FactorGraph.delete_var_node` | Remove a variable and all adjacent factors, including cross-graph links. |
| `FactorGraph.delete_factor_node` | Remove a factor and unlink all adjacent variables. |
| `MeasurementFactor` | Bind a sensor name, measurement, residual/Jacobian model, covariance, extrinsics, and approximation policy. |
| `fg_solver.compute_all_BP_var_messages` | Compute all internal BP variable-to-factor messages. |
| `fg_solver.compute_all_factor_messages_MF` | Compute closed-form MF messages for supported conjugate variables. |
| `fg_solver.update_all_factor_belief` | Update factor belief natural parameters using estimated gradients and incoming variable messages. |
| `fg_solver.compute_all_factor_messages_BP` | Marginalize batched factor beliefs into BP factor-to-variable messages. |
| `fg_solver.update_all_var_belief` | Update variable belief natural parameters using incoming factor messages. |
| `transfer_var_msg` / `transfer_factor_msg` | Exchange packed cross-robot messages. |
| `fg_solver.retract_all_BP_var_belief` | Retract Gaussian pose increments and transport their covariance on the manifold. |

## Vectorization mechanism

The solver avoids processing every edge independently during its main numerical updates:

- Natural parameters and BP/MF messages are packed into contiguous column vectors.
- Factors are grouped by `(sensor_name, natural_parameter_dimension)`.
- Measurements, Lie-group tangent points, and extrinsic parameters are stacked by sensor type.
- Sparse mapping matrices gather incoming messages, project factor means and covariances, and support belief updates.
- Residuals, Jacobians, matrix inversions, Cholesky decompositions, and Monte Carlo samples are evaluated in batches with NumPy during evaluation.
- The sampling path flattens `number_of_factors x number_of_samples` into one leading dimension, evaluates the sensor model once per batch, and then reshapes the results for averaging.

## Cache mechanism

The evaluation solver caches the six sparse mapping matrices that encode a graph's packed message layout:

```text
L_factor, L_1_factor_msg, L_2_factor_msg,
L_var_BP, L_var_MF, L_var_msg
```

The cache key is derived from matrix shapes and structural block specifications rather than belief values or measurements. Therefore, newly constructed solvers with the same packed topology can reuse the mapping structure while retaining fresh numerical beliefs and sensor data. This is especially useful when the sliding window repeatedly creates the same graph pattern.

The cache is process-local and in-memory. It uses an LRU policy with at most 64 graph-layout entries. Projection indices used while building the matrices have a separate `lru_cache` with at most 256 entries. Each solver receives a copy of the cached sparse matrices, so instance-level customization does not mutate the stored template.

```python
print(fg_solver.cache_info())
# {'hits': ..., 'misses': ..., 'entries': ...}

fg_solver.clear_cache()
```

Use `clear_cache()` when intentionally releasing cached layouts. The cache does not persist to disk and does not cache factor beliefs, measurements, learned-model outputs, or inferred messages.

## Training

Run training from the `Training` directory:

```bash
cd Training
python train.py
python train-BP.py
python train-R3.py
python train-BP-R3.py
```

The scripts generate synthetic data, start TensorBoard on port 6006, unroll inference, optimize iteration-specific conditional RealNVP models, save `checkpoint.pth`, and export `uwb_iter1.onnx` through `uwb_iter5.onnx`.

Training can be computationally and memory intensive. Review `batch_size`, `data_size`, `n_state`, `iteration_num`, `sample_num`, and `train_interval` before running. Also verify that port 6006 is available and that the installed PyTorch/PyTorch3D builds match the local CPU or CUDA environment.

## Solver assumptions that users should review

`FactorSolver.py` is a research solver and contains application-specific assumptions. Before using it in a new problem, verify at least the following:

- state ordering and Jacobian conventions, especially the right-perturbation convention on `SO(3) x R3`;
- the learned-proposal condition vector, which is currently specialized to the UWB factor and to the state dimensions used by the example;
- closed-form MF message rules, which are implemented for the specific inverse-Wishart/Beta/Bernoulli/Gamma structures used here;
- cross-robot ownership and the update order used by `transfer_var_msg` and `transfer_factor_msg`;

Adding a new measurement factor normally requires only a compatible residual/Jacobian and graph construction. Adding a learned proposal for a new sensor, a new MF distribution, a different state manifold, or a different message schedule will generally require changes inside the solver.

To better emulate distributed multi-robot operation, the evaluation code constructs a separate factor graph and solver for each robot and explicitly implements inter-robot message exchange. When the factor graph is solved on a single compute unit, this structure can be simplified to a single factor graph and solver. The training code adopts this implementation.

## Outputs

Evaluation creates:

- `plotdata-MF-<mode>.npz` for BP–MF in SO(3) x R3 test;
- `plotdata-BP-<mode>.npz` for BP in SO(3) x R3 test;
- `plotdata-MF-<mode>-R3.npz` for BP–MF in R3 test;
- `plotdata-BP-<mode>-R3.npz` for BP in R3 test;
- printed average runtime and orientation/position ARMSE;
- Matplotlib RMSE figures.

Training creates or overwrites checkpoints, ONNX models, and TensorBoard event files in its output directory.

## Experimental video

[https://github.com/han-sh/NF-enhanced-MP/issues/1#issue-5388960236](https://github.com/user-attachments/assets/48148f01-cbac-4bc3-9c3e-c8101ee3391c)

## Citation

If this code is useful in your research, please cite:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {IEEE Transactions on Robotics},
  year    = {TODO}
}
```

## License

No license is declared in the current snapshot.
