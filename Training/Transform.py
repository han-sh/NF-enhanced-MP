"""SO(3) x R3 state operations."""

import torch
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix


class State:
    """State containing a rotation matrix and a position vector."""

    def __init__(self, orientation, position):
        self.orientation = orientation.clone()
        self.position = position.clone()


def rotm2rotvec(R):
    """Convert one rotation matrix or a batch of matrices to rotation vectors."""

    return matrix_to_axis_angle(R).unsqueeze(-1)


def rotvec2rotm(r):
    """Convert one rotation vector or a batch of vectors to rotation matrices."""

    return axis_angle_to_matrix(r.squeeze(-1))


def tform(R, t):
    """Assemble transforms."""

    single = False
    if R.ndim == 2:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
        single = True

    b = R.shape[0]
    T = torch.eye(4, dtype=R.dtype, device=R.device).repeat(b, 1, 1)
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = t.squeeze(-1)
    return T[0] if single else T


def tform2rotm(T):
    """Extract rotations from transforms."""

    if T.ndim == 2:
        return T[0:3, 0:3]
    else:
        return T[:, 0:3, 0:3]


def tform2trans(T):
    """Extract translation column vectors from transforms."""

    if T.ndim == 2:
        return T[0:3, 3].reshape(3, 1)
    else:
        return T[:, 0:3, 3].reshape(-1, 3, 1)


def composite_add(state1, state2_vec):
    """Apply state1 boxplus state2_vec."""

    state = State(state1.orientation, state1.position)
    if state1.orientation.ndim == 2:
        state.orientation = state.orientation @ rotvec2rotm(state2_vec[0:3])
        state.position = state.position + state2_vec[3:6]
    else:
        state.orientation = state.orientation @ rotvec2rotm(state2_vec[:, 0:3])
        state.position = state.position + state2_vec[:, 3:6]
    return state


def composite_minus(state1, state2):
    """Compute state1 boxminus state2."""

    if state1.orientation.ndim == 2:
        state_vec = torch.zeros(
            (6, 1), dtype=state1.orientation.dtype, device=state1.orientation.device
        )
        R_rel = state2.orientation.T @ state1.orientation
        state_vec[0:3] = rotm2rotvec(R_rel)
        state_vec[3:6] = state1.position - state2.position
    else:
        B = state1.orientation.shape[0]
        state_vec = torch.zeros(
            (B, 6, 1), dtype=state1.orientation.dtype, device=state1.orientation.device
        )
        R_rel = state2.orientation.transpose(1, 2) @ state1.orientation
        state_vec[:, 0:3] = rotm2rotvec(R_rel)
        state_vec[:, 3:6] = state1.position - state2.position
    return state_vec


def rightJaco(x, eps=0.1):
    single = x.ndim == 2
    if single:
        x = x.unsqueeze(0)

    theta2 = x.squeeze(-1).square().sum(dim=-1)
    small = theta2 < eps**2

    safe_theta2 = torch.where(
        small, torch.ones_like(theta2), theta2
    )
    safe_theta = torch.sqrt(safe_theta2)

    a_series = 0.5 + theta2 * (
        -1 / 24 + theta2 * (1 / 720 - theta2 / 40320)
    )
    b_series = 1 / 6 + theta2 * (
        -1 / 120 + theta2 * (1 / 5040 - theta2 / 362880)
    )

    a_regular = (1 - torch.cos(safe_theta)) / safe_theta2
    b_regular = (
        (safe_theta - torch.sin(safe_theta))
        / (safe_theta2 * safe_theta)
    )

    a = torch.where(small, a_series, a_regular)[:, None, None]
    b = torch.where(small, b_series, b_regular)[:, None, None]

    S = skewsym(x)
    eye = torch.eye(
        3, dtype=x.dtype, device=x.device
    ).unsqueeze(0)

    J = eye - a * S + b * (S @ S)
    return J[0] if single else J


def inv_rightJaco(x, eps=0.1):
    single = x.ndim == 2
    if single:
        x = x.unsqueeze(0)

    theta2 = x.squeeze(-1).square().sum(dim=-1)
    small = theta2 < eps**2

    safe_theta2 = torch.where(
        small, torch.ones_like(theta2), theta2
    )
    half_theta = 0.5 * torch.sqrt(safe_theta2)

    c_series = 1 / 12 + theta2 * (
        1 / 720 + theta2 * (1 / 30240 + theta2 / 1209600)
    )
    c_regular = (
        1 - half_theta / torch.tan(half_theta)
    ) / safe_theta2

    c = torch.where(small, c_series, c_regular)[:, None, None]

    S = skewsym(x)
    eye = torch.eye(
        3, dtype=x.dtype, device=x.device
    ).unsqueeze(0)

    Jinv = eye + 0.5 * S + c * (S @ S)
    return Jinv[0] if single else Jinv


def composite_rightJaco(x):
    """Build the block right Jacobian for multi SO(3) x R3."""

    single = False
    if x.ndim == 2:
        x = x.unsqueeze(0)
        single = True

    dim = 6
    dima = x.shape[1]
    b = x.shape[0]
    var_num = dima // dim
    J = (
        torch.eye(dima, dtype=x.dtype, device=x.device)
        .reshape(1, dima, dima)
        .repeat(b, 1, 1)
    )
    for var_id in range(var_num):
        r = x[:, var_id * dim : var_id * dim + 3]
        J[:, var_id * dim : var_id * dim + 3, var_id * dim : var_id * dim + 3] = (
            rightJaco(r)
        )

    return J[0] if single else J


def skewsym(x):
    """Return skew-symmetric matrices for one or many three-vectors."""

    if x.ndim == 2:
        x = x.reshape(1, 3)
        dim2 = True
    else:
        x = x.squeeze(-1)
        dim2 = False

    B = x.shape[0]
    K = torch.zeros((B, 3, 3), dtype=x.dtype, device=x.device)

    K[:, 0, 1] = -x[:, 2]
    K[:, 0, 2] = x[:, 1]
    K[:, 1, 0] = x[:, 2]
    K[:, 1, 2] = -x[:, 0]
    K[:, 2, 0] = -x[:, 1]
    K[:, 2, 1] = x[:, 0]

    return K[0] if dim2 else K
