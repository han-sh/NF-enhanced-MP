"""Residual and Jacobian models."""

import numpy as np

from Transform import rotm2rotvec, inv_rightJaco, skewsym


class prior:
    """Prior factor in SO(3) x R3."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        """Return the prior residual ``z - x`` in error-state coordinates."""

        return z - x

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        """Return the prior residual Jacobian."""

        J = -np.eye(6)
        if x.ndim == 3:
            b = x.shape[0]
            J = np.repeat(J[None, :, :], b, axis=0)

        return J
    

class odo:
    """Relative-pose odometry residual."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        """Evaluate rotational and translational residuals."""

        Rk_1 = x[0].orientation
        pk_1 = x[0].position
        Rk   = x[1].orientation
        pk   = x[1].position
        Rz   = z.orientation
        pz   = z.position

        if Rk_1.ndim == 3:
            b = Rk_1.shape[0]
            error = np.zeros((b,6,1))
            error[:,0:3] = rotm2rotvec(Rk.transpose(0,2,1) @ Rk_1 @ Rz)
            error[:,3:6] = pz - Rk_1.transpose(0,2,1) @ (pk - pk_1)
        else:
            error = np.zeros((6,1))
            error[0:3] = rotm2rotvec(Rk.T @ Rk_1 @ Rz)
            error[3:6] = pz - Rk_1.T @ (pk - pk_1)
        return error
    
    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        """Evaluate the odometry residual Jacobian."""

        Rk_1 = x[0].orientation
        pk_1 = x[0].position
        Rk   = x[1].orientation
        pk   = x[1].position
        Rz   = z.orientation

        if Rk_1.ndim == 3:
            b = Rk_1.shape[0]
            J = np.zeros((b,6,12))

            inv_rightJ = inv_rightJaco(rotm2rotvec(Rk.transpose(0,2,1) @ Rk_1 @ Rz))

            J[:,0:3,0:3]   = inv_rightJ @ Rz.transpose(0,2,1)
            J[:,0:3,6:9]   = -inv_rightJ @ Rz.transpose(0,2,1) @ Rk_1.transpose(0,2,1) @ Rk

            J[:,3:6,0:3]   = -skewsym(Rk_1.transpose(0,2,1) @ (pk - pk_1))
            J[:,3:6,3:6]   = Rk_1.transpose(0,2,1)
            J[:,3:6,9:12]  = -Rk_1.transpose(0,2,1)
        else:
            J = np.zeros((6,12))

            inv_rightJ = inv_rightJaco(rotm2rotvec(Rk.T @ Rk_1 @ Rz))

            J[0:3,0:3]   = inv_rightJ @ Rz.T
            J[0:3,6:9]   = -inv_rightJ @ Rz.T @ Rk_1.T @ Rk

            J[3:6,0:3]   = -skewsym(Rk_1.T @ (pk - pk_1))
            J[3:6,3:6]   = Rk_1.T
            J[3:6,9:12]  = -Rk_1.T

        return J

    
class gnss:
    """GNSS antenna-position residual."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        """Evaluate ``z - R t_G - p``."""

        tG = ext_param[0]["t"]
        Rk = x[0].orientation
        pk = x[0].position

        error = z - (Rk @ tG) - pk
        return error
    
    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        """Evaluate the GNSS residual Jacobian."""

        tG = ext_param[0]["t"]
        Rk = x[0].orientation

        if Rk.ndim == 3:
            b = Rk.shape[0]
            J = np.concatenate(
            (Rk @ skewsym(tG), -np.eye(3)[None, ...].repeat(b, axis=0)),
            axis=-1
        )
        else:
            J = np.hstack((Rk @ skewsym(tG), -np.eye(3)))

        return J
    

class uwb:
    """Inter-robot UWB range residual."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        """Evaluate measured range minus antenna-to-antenna distance."""

        tU1 = ext_param[0]["t"]
        tU2 = ext_param[1]["t"]

        R1 = x[0].orientation
        R2 = x[1].orientation
        p1 = x[0].position
        p2 = x[1].position

        deltat = R1 @ tU1 + p1 - (R2 @ tU2 + p2)
        if R1.ndim == 3:
            error = z - np.linalg.norm(deltat.squeeze(-1), axis=-1).reshape(-1,1,1)
        else:
            error = z - np.linalg.norm(deltat.squeeze(-1), axis=-1).reshape(-1,1)
        return error

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        """Evaluate the UWB residual Jacobian."""

        tU1 = ext_param[0]["t"]
        tU2 = ext_param[1]["t"]

        R1 = x[0].orientation
        R2 = x[1].orientation
        p1 = x[0].position
        p2 = x[1].position

        deltat = R1 @ tU1 + p1 - (R2 @ tU2 + p2)
        norms = np.linalg.norm(deltat.squeeze(-1), axis=-1)

        if R1.ndim == 3:
            b = R1.shape[0]
            mask = np.isclose(norms, 0.0)

            J = np.zeros((b, 1, 12))

            if np.any(~mask):
                d = (deltat[~mask] / norms[~mask, None, None]).transpose(0,2,1)

                part1 = d @ R1[~mask] @ skewsym(tU1[~mask])
                part2 = -d
                part3 = -d @ R2[~mask] @ skewsym(tU2[~mask])
                part4 = d

                J[~mask] = np.concatenate((part1, part2, part3, part4), axis=-1)
        else:
            if np.isclose(norms, 0.0):
                J = np.zeros((1,12))
            else:
                d = (deltat / norms).T
                J = np.hstack((
                    d @ R1 @ skewsym(tU1),
                    -d,
                    -d @ R2 @ skewsym(tU2),
                    d
                ))
        return J

class prior_R3:
    """Prior factor in R3."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        return z - x

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        J = -np.eye(3)
        if x.ndim == 3:
            b = x.shape[0]
            J = np.repeat(J[None, :, :], b, axis=0)

        return J


class odo_R3:
    @staticmethod
    def model_fcn(x, z, ext_param):
        if x.ndim == 3:
            x1 = x[:, 0:3, :]   # (b, 3, 1)
            x2 = x[:, 3:6, :]   # (b, 3, 1)
            error = z - (x2 - x1)   # (b, 3, 1)
        else:
            x1 = x[0:3, :]   # (3, 1)
            x2 = x[3:6, :]   # (3, 1)
            error = z - (x2 - x1)   # (3, 1)
        return error

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        I3 = np.eye(3)
        J = np.hstack([I3, -I3])  # (3,6)
        if x.ndim == 3:
            b = x.shape[0]
            J = np.repeat(J[None, :, :], b, axis=0) # (b, 3, 6)

        return J

class gnss_R3:
    @staticmethod
    def model_fcn(x, z, ext_param):
        tG = ext_param[0]["t"]
        return z - x- tG

    def Jaco_fcn(x, meas, ext_param):
        J = -np.eye(3)
        if x.ndim == 3:
            b = x.shape[0]
            J = np.repeat(J[None, :, :], b, axis=0)

        return J

class uwb_R3:
    @staticmethod
    def model_fcn(x, z, ext_param):
        tU1 = ext_param[0]["t"]
        tU2 = ext_param[1]["t"]

        if x.ndim == 3:
            x1 = x[:, 0:3, :]   # (b, 3, 1)
            x2 = x[:, 3:6, :]   # (b, 3, 1)

            dist = np.linalg.norm(x1+tU1 - x2-tU2, axis=1, keepdims=True)  # (b, 1, 1)
            error = z - dist
        else:
            x1 = x[0:3, :]   # (3, 1)
            x2 = x[3:6, :]   # (3, 1)

            dist = np.linalg.norm(x1+tU1 - x2-tU2, keepdims=True)  # (1, 1)
            error = z - dist
        return error

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        tU1 = ext_param[0]["t"]
        tU2 = ext_param[1]["t"]
        if x.ndim == 3:
            b = x.shape[0]

            J = np.zeros((b, 1, 6))

            x1 = x[:, 0:3, :]   # (b, 3, N)
            x2 = x[:, 3:6, :]   # (b, 3, N)

            diff = x1+tU1 - x2-tU2      # (b, 3, N)
            dis = np.linalg.norm(diff, axis=1, keepdims=True)  # (b, 1, 1)
            dis = np.repeat(dis, 3, axis=1)  # (b, 3, 1)

            J[:, :, 0:3] = -(diff / dis).transpose(0, 2, 1)

            J[:, :, 3:6] = (diff / dis).transpose(0, 2, 1)

        else:
            J = np.zeros((1, 6))

            x1 = x[0:3, :]   # (3, N)
            x2 = x[3:6, :]   # (3, N)

            diff = x1+tU1 - x2-tU2      # (3, N)
            dis = np.linalg.norm(diff)  # (1, 1)
            dis = np.tile(dis, (3, 1))  # (3, 1)

            J[:, 0:3] = -(diff / dis).T
            J[:, 3:6] = (diff / dis).T

        return J