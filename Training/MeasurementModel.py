"""Batched residual and Jacobian models for the training factor graph."""

import torch

from Transform import rotm2rotvec, inv_rightJaco, skewsym


class prior:
    """Prior factor in SO(3) x R3."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        return z - x

    @staticmethod
    def Jaco_fcn(x, meas, ext_param):
        B = x.shape[0]
        I6 = torch.eye(6, dtype=x.dtype, device=x.device)
        J = -I6.reshape(1, 6, 6).expand(B, -1, -1).clone()

        return J


class odo:
    """Relative-pose odometry residual."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        Rk_1 = x[0].orientation
        pk_1 = x[0].position
        Rk = x[1].orientation
        pk = x[1].position
        Rz = z.orientation
        pz = z.position

        dtype = Rk_1.dtype
        device = Rk_1.device

        B = Rk_1.shape[0]
        error = torch.zeros((B, 6, 1), dtype=dtype, device=device)
        error[:, 0:3] = rotm2rotvec(Rk.transpose(2, 1) @ Rk_1 @ Rz)
        error[:, 3:6] = pz - Rk_1.transpose(2, 1) @ (pk - pk_1)
        return error

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        Rk_1 = x[0].orientation
        pk_1 = x[0].position
        Rk = x[1].orientation
        pk = x[1].position
        Rz = z.orientation

        dtype = Rk_1.dtype
        device = Rk_1.device

        B = Rk_1.shape[0]
        J = torch.zeros((B, 6, 12), dtype=dtype, device=device)

        inv_rightJ = inv_rightJaco(rotm2rotvec(Rk.transpose(2, 1) @ Rk_1 @ Rz))

        J[:, 0:3, 0:3] = inv_rightJ @ Rz.transpose(2, 1)
        J[:, 0:3, 6:9] = -inv_rightJ @ Rz.transpose(2, 1) @ Rk_1.transpose(2, 1) @ Rk

        J[:, 3:6, 0:3] = -skewsym(Rk_1.transpose(2, 1) @ (pk - pk_1))
        J[:, 3:6, 3:6] = Rk_1.transpose(2, 1)
        J[:, 3:6, 9:12] = -Rk_1.transpose(2, 1)

        return J


class gnss:
    """GNSS antenna-position residual."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        tG = ext_param[0]["t"]
        Rk = x[0].orientation
        pk = x[0].position

        error = z - (Rk @ tG) - pk
        return error

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        tG = ext_param[0]["t"]
        Rk = x[0].orientation

        B = Rk.shape[0]
        J = torch.cat(
            (
                (Rk @ skewsym(tG)),
                -torch.eye(3, dtype=Rk.dtype, device=Rk.device)
                .unsqueeze(0)
                .repeat(B, 1, 1),
            ),
            dim=-1,
        )

        return J


class uwb:
    """Inter-robot UWB range residual."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        tU1 = ext_param[0]["t"]
        tU2 = ext_param[1]["t"]

        R1 = x[0].orientation
        R2 = x[1].orientation
        p1 = x[0].position
        p2 = x[1].position

        deltat = R1 @ tU1 + p1 - (R2 @ tU2 + p2)
        error = z - torch.linalg.norm(deltat.squeeze(-1), dim=-1).reshape(-1, 1, 1)
        return error

    @staticmethod
    def Jaco_fcn(x, z, ext_param):
        tU1 = ext_param[0]["t"]
        tU2 = ext_param[1]["t"]

        R1 = x[0].orientation
        R2 = x[1].orientation
        p1 = x[0].position
        p2 = x[1].position

        dtype = R1.dtype
        device = R1.device

        deltat = R1 @ tU1 + p1 - (R2 @ tU2 + p2)
        norms = torch.linalg.norm(deltat.squeeze(-1), dim=-1)

        B = R1.shape[0]
        mask = torch.isclose(norms, torch.zeros_like(norms))

        J = torch.zeros((B, 1, 12), dtype=dtype, device=device)

        if (~mask).any():
            d = (deltat[~mask] / norms[~mask].reshape(-1, 1, 1)).transpose(1, 2)
            part1 = d @ R1[~mask] @ skewsym(tU1[~mask])
            part2 = -d
            part3 = -d @ R2[~mask] @ skewsym(tU2[~mask])
            part4 = d

            J[~mask] = torch.cat((part1, part2, part3, part4), dim=-1)

        return J


class prior_R3:
    """Prior factor in R3."""

    @staticmethod
    def model_fcn(x, z, ext_param):
        return z - x

    @staticmethod
    def Jaco_fcn(x, meas, ext_param):
        B = x.shape[0]
        I3 = torch.eye(3, dtype=x.dtype, device=x.device)
        J = -I3.reshape(1, 3, 3).expand(B, -1, -1).clone()

        return J


class odo_R3:
    @staticmethod
    def model_fcn(x, z, ext_param):
        x1 = x[:, 0:3, :]
        x2 = x[:, 3:6, :]
        error = z - (x2 - x1)
        return error

    @staticmethod
    def Jaco_fcn(x, meas, ext_param):
        B = x.shape[0]
        I3 = torch.eye(3, dtype=x.dtype, device=x.device)

        J_single = torch.cat([I3, -I3], dim=1)

        J = J_single.reshape(1, 3, 6).expand(B, -1, -1).clone()

        return J


class gnss_R3:
    @staticmethod
    def model_fcn(x, z, ext_param):
        tG = ext_param[0]["t"]
        return z - x - tG

    @staticmethod
    def Jaco_fcn(x, meas, ext_param):
        B = x.shape[0]
        I3 = torch.eye(3, dtype=x.dtype, device=x.device)
        J = -I3.reshape(1, 3, 3).expand(B, -1, -1).clone()

        return J


class uwb_R3:
    @staticmethod
    def model_fcn(x, z, ext_param):
        tU1 = ext_param[0]["t"]
        tU2 = ext_param[1]["t"]

        x1 = x[:, 0:3, :]  # (B, 3, N)
        x2 = x[:, 3:6, :]  # (B, 3, N)
        dist = torch.norm(x1 + tU1 - x2 - tU2, dim=1, keepdim=True)  # (B, 1, 1)
        error = z - dist
        return error

    @staticmethod
    def Jaco_fcn(x, meas, ext_param):
        tU1 = ext_param[0]["t"]
        tU2 = ext_param[1]["t"]

        B = x.shape[0]

        J = torch.zeros((B, 1, 6), dtype=x.dtype, device=x.device)

        x1 = x[:, 0:3, :]  # (B, 3, N)
        x2 = x[:, 3:6, :]  # (B, 3, N)

        diff = x1 + tU1 - x2 - tU2  # (B, 3, N)
        dis = torch.norm(diff, dim=1, keepdim=True)  # (B, 1, 1)
        dis = dis.repeat(1, 3, 1)  # (B, 3, 1)

        J[:, :, 0:3] = -(diff / dis).transpose(1, 2)
        J[:, :, 3:6] = (diff / dis).transpose(1, 2)

        return J
