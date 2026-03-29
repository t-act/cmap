import numpy as np


class Grid:
    """R-Z グリッドの一元管理。

    Attributes
    ----------
    r, z : np.ndarray
        (101,), (201,) の座標配列
    dr, dz : float
        グリッド間隔 (0.02 m)
    nr, nz : int
        グリッド点数 (101, 201)
    ir_min, ir_max : int
        R 方向インデックス範囲 (0, 100)
    iz_min, iz_max : int
        Z 方向インデックス範囲 (0, 200)
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 2.0,
        dr: float = 0.02,
        z_min: float = -2.0,
        z_max: float = 2.0,
        dz: float = 0.02,
    ) -> None:
        self.dr = dr
        self.dz = dz
        self.dl = 0.02
        self.r_min = r_min
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max

        self.r = np.arange(r_min, r_max + dr, dr)  # (101,)
        self.z = np.arange(z_min, z_max + dz, dz)  # (201,)
        self.nr = len(self.r)          # 101
        self.nz = len(self.z)          # 201
        self.ir_min = 0
        self.ir_max = self.nr - 1      # 100
        self.iz_min = 0
        self.iz_max = self.nz - 1      # 200
