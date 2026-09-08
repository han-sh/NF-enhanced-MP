"""SO(3) x R3 state operations."""

import numpy as np
from scipy.spatial.transform import Rotation


class State:
    """State containing a rotation matrix and a position vector."""

    def __init__(self, orientation, position):
        self.orientation = orientation.copy()
        self.position = position.copy()

def rotm2rotvec(R):
    """Convert one rotation matrix or a batch of matrices to rotation vectors."""

    R = Rotation.from_matrix(R)
    return R.as_rotvec()[..., None]

def rotvec2rotm(r):
    """Convert one rotation vector or a batch of vectors to rotation matrices."""

    r = Rotation.from_rotvec(r.squeeze(-1))
    return r.as_matrix()

def tform(R, t):
    """Assemble transforms."""

    single = False
    if R.ndim == 2:
        R = R[None, :, :]
        t = t[None, :, :]
        single = True

    b = R.shape[0]
    T = np.tile(np.eye(4), (b,1,1))
    T[:,0:3,0:3] = R
    T[:,0:3,3]   = t.squeeze(-1)
    return T[0] if single else T

def tform2rotm(T):
    """Extract rotation matrices from transforms."""

    if T.ndim == 2:
        return T[0:3,0:3]
    else:
        return T[:,0:3,0:3]

def tform2trans(T):
    """Extract column-vector translations from transforms."""

    if T.ndim == 2:
        return T[0:3,3].reshape(3,1)
    else:
        return T[:,0:3,3].reshape(-1,3,1)

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
        state_vec = np.zeros((6,1))
        R_rel = state2.orientation.T @ state1.orientation
        state_vec[0:3] = rotm2rotvec(R_rel)
        state_vec[3:6] = state1.position - state2.position
    else:
        b = state1.orientation.shape[0]
        state_vec = np.zeros((b,6,1))
        R_rel = state2.orientation.transpose(0, 2, 1) @ state1.orientation
        state_vec[:, 0:3] = rotm2rotvec(R_rel)
        state_vec[:, 3:6] = state1.position - state2.position
    return state_vec

def rightJaco(x, eps=1e-8):
    """Evaluate the SO(3) right Jacobian given rotation vectors."""

    single = False
    if x.ndim == 2:
        x = x[None, :, :]
        single = True

    normx = np.linalg.norm(x.squeeze(-1), axis=1)
    S = skewsym(x)

    coef_a = np.empty_like(normx)[..., None, None]
    coef_b = np.empty_like(normx)[..., None, None]

    small_mask = normx < eps
    large_mask = ~small_mask

    if np.any(large_mask):
        nx = normx[large_mask]
        coef_a[large_mask] = ((1 - np.cos(nx)) / (nx**2))[:, None, None]
        coef_b[large_mask] = ((nx - np.sin(nx)) / (nx**3))[:, None, None]

    eye = np.eye(3)[None, :, :]
    J = eye - coef_a * S + coef_b * (S @ S)
    if np.any(small_mask):
        J[small_mask] = eye

    return J[0] if single else J
    
def inv_rightJaco(x, eps=1e-8):
    """Evaluate the inverse SO(3) right Jacobian."""

    single = False
    if x.ndim == 2:
        x = x[None, :, :]
        single = True

    normx = np.linalg.norm(x.squeeze(-1), axis=1)
    S = skewsym(x)

    coeff = np.empty_like(normx)[..., None, None]

    small_mask = normx < eps
    large_mask = ~small_mask

    if np.any(large_mask):
        nx = normx[large_mask]
        coeff[large_mask] = (
            (1.0 / (nx**2)) - (1.0 + np.cos(nx)) / (2.0 * nx * np.sin(nx))
        )[:, None, None]

    eye = np.eye(3)[None, :, :]
    invJ = eye + 0.5 * S + coeff * (S @ S)

    if np.any(small_mask):
        invJ[small_mask] = eye

    return invJ[0] if single else invJ

def composite_rightJaco(x):
    """Build the block right Jacobian for multi SO(3) x R3."""

    single = False
    if x.ndim == 2:
        x = x[None, :, :]
        single = True
        
    dim = 6
    dima = x.shape[1]
    b = x.shape[0]
    var_num = dima//dim
    J = np.repeat(np.eye(dima, dima).reshape(1, dima, dima), b, axis=0)
    for var_id in range(var_num):
        r = x[:, var_id*dim:var_id*dim+3]
        J[:, var_id*dim:var_id*dim+3, var_id*dim:var_id*dim+3] = rightJaco(r)

    return J[0] if single else J

def skewsym(x):
    """Return skew-symmetric matrices for one or many three-vectors."""

    if x.ndim == 2:
        x = x.reshape(1,3)
        single = True
    else:
        x = x.squeeze(-1)
        single = False

    b = x.shape[0]
    K = np.zeros((b,3,3), dtype=float)

    K[:,0,1] = -x[:,2]
    K[:,0,2] =  x[:,1]
    K[:,1,0] =  x[:,2]
    K[:,1,2] = -x[:,0]
    K[:,2,0] = -x[:,1]
    K[:,2,1] =  x[:,0]

    return K[0] if single else K
