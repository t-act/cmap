"""QUEST トカマクの装置形状・コイル配置・電極データを管理するモジュール。

Parameter.py の全データと mfield_sub.py の elect_posi / cal_sn / get_PF を統合。
"""

import numpy as np


# ---------------------------------------------------------------------------
# 装置形状関数 (旧 mfield_sub.py から移動)
# ---------------------------------------------------------------------------

def elect_posi(r: float, z: float) -> bool:
    """電極位置の判定 (容器壁・電極領域なら True)。"""
    return (
        r < 0.22
        or r > 1.2
        or z > 1.0
        or (r < 0.280 and z < -1.13)
        or (r < 0.389 and z < -1.32)
        or (r >= 0.389 and z < -1.165)
        or (z > -1.0694 * r - 0.14326 and z < 1.2956 * r - 1.84223 and r > 0.74833)
        or (r - 0.78983) ** 2 + (z + 0.91089) ** 2 < 0.053 ** 2
        or z > -1.2956 * r + 1.84223
    )


def get_PF() -> tuple[np.ndarray, np.ndarray]:
    """PFdata.csv から PF コイル電流・TF コイル電流を読み込む。

    Returns
    -------
    PF : np.ndarray  [A]  PF3-1, 3-2, 2, 1, 7, 6, 5-2, 5-1, 4-1, 4-2, 4-3 (shape 11)
    TF : float       [A]
    """
    data = np.genfromtxt("modules/PFdata.csv", delimiter=",", skip_footer=4)
    return data[1:12] * 1e3, data[12] * 1e3


def cal_sn(
    elect0: np.ndarray,
    elect: np.ndarray,
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """電極の法線ベクトルを計算し electrode.csv を保存する。

    Returns
    -------
    sn_r0, sn_z0 : 外側電極 (elect0) の法線ベクトル
    sn_r,  sn_z  : 注入電流電極 (elect) の法線ベクトル
    """
    sn_r0, sn_z0 = np.zeros(19), np.zeros(19)
    sn_r, sn_z = np.zeros(19), np.zeros(19)
    electrode_data = [[elect0[0, 0], elect0[0, 1], 0]]

    for i in range(18):
        if elect0[i, 0] > 0.1 and elect0[i + 1, 0] > 0.1:
            electrode_data.append([elect0[i + 1, 0], elect0[i + 1, 1], 0])
            seg0 = np.sqrt(
                (elect0[i + 1, 0] - elect0[i, 0]) ** 2
                + (elect0[i + 1, 1] - elect0[i, 1]) ** 2
            )
            sn_0 = seg0 * 2 * np.pi * (elect0[i + 1, 0] + elect0[i, 0]) * 0.5
            e_r0 = (elect0[i + 1, 0] - elect0[i, 0]) / seg0
            e_z0 = (elect0[i + 1, 1] - elect0[i, 1]) / seg0
            sn_r0[i] = -e_z0 * sn_0
            sn_z0[i] = e_r0 * sn_0

        if elect[i, 0] > 0.1 and elect[i + 1, 0] > 0.1:
            seg = np.sqrt(
                (elect[i + 1, 0] - elect[i, 0]) ** 2
                + (elect[i + 1, 1] - elect[i, 1]) ** 2
            )
            sn = seg * 2 * np.pi * (elect[i + 1, 0] + elect[i, 0]) * 0.5
            e_r = (elect[i + 1, 0] - elect[i, 0]) / seg
            e_z = (elect[i + 1, 1] - elect[i, 1]) / seg
            sn_r[i] = -e_z * sn
            sn_z[i] = e_r * sn

    electrode_data_np = np.array(electrode_data)
    np.savetxt(f"{path}/electrode.csv", electrode_data_np, delimiter=",", fmt="%.6f")
    return sn_r0, sn_z0, sn_r, sn_z


# ---------------------------------------------------------------------------
# TokamakConfig クラス
# ---------------------------------------------------------------------------

class TokamakConfig:
    """QUEST トカマクの静的装置形状データを管理するクラス。

    Attributes
    ----------
    elect0 : np.ndarray  (20, 3)  外側電極位置
    elect  : np.ndarray  (20, 3)  注入電流電極位置
    flux   : np.ndarray (118, 2)  フラックスループ位置 [R, Z]
    r_c, z_c : np.ndarray  (55,)  PF コイル位置
    ele_lim : list  [R, Z]        外側電極限界位置
    SMmin   : np.ndarray  (3,)    ソルバー収束管理用初期値
    """

    def __init__(self) -> None:
        self.ele_lim = [0.47, -1.165]

        self._build_electrodes()
        self._build_flux_loops()
        self._build_pf_coil_positions()

        self.SMmin = np.array([1.0e30, 1.0e30, 1.0e30])

    # ------------------------------------------------------------------
    def _build_electrodes(self) -> None:
        self.elect0 = np.zeros((20, 3))
        self.elect0[0, :] = [0.389, -1.315, 1.0]
        self.elect0[1, :] = [0.389, -1.165, 1.0]
        self.elect0[2, :] = [0.500, -1.165, 1.0]

        self.elect = np.array([
            [0.50,  -1.165, 1.0],
            [0.389, -1.165, 1.0],
            [0.389, -1.32,  1.0],
            [0.28,  -1.32,  1.0],
            [0.28,  -1.13,  1.0],
            [0.22,  -1.13,  1.0],
            [0.22,  -0.49,  1.0],
            [0.22,   0.11,  1.0],
            [0.22,   0.86,  1.0],
            [0.22,   1.0,   1.0],
            [0.722,  1.0,   0.0],
            [1.2,    0.284, 0.0],
            [1.2,   -0.284, 0.0],
            [0.719, -0.918, 0.0],
            [0.753, -1.165, 0.0],
            [0.50,  -1.165, 0.0],
            [0,      0,     0],
            [0,      0,     0],
            [0,      0,     0],
            [0,      0,     1],
        ])

    # ------------------------------------------------------------------
    def _build_flux_loops(self) -> None:
        flux = np.zeros((118, 2))

        # FLC
        flux[:24, 0] = 0.1985
        flux[:24, 1] = np.linspace(1.15, -1.15, 24)

        # FLT
        flux[24:30, :] = [
            [0.214746, 1.300], [0.263, 1.394], [0.37479, 1.394],
            [0.47479,  1.394], [0.57479, 1.394], [0.699524, 1.394],
        ]

        # FLTS
        flux[30:41, :] = [
            [0.732669, 1.350], [0.770334, 1.300], [0.827584, 1.224],
            [0.894628, 1.135], [0.953385, 1.057], [1.015532, 0.9745],
            [1.071652, 0.900], [1.133799, 0.8175], [1.202726, 0.726],
            [1.271653, 0.6345], [1.326267, 0.562],
        ]

        # FLS
        flux[41:50, :] = [
            [1.374,  0.450], [1.374,  0.327], [1.374,  0.257],
            [1.374,  0.1335], [1.374,  0.0], [1.374, -0.137],
            [1.374, -0.263], [1.374, -0.3325], [1.374, -0.481],
        ]

        flux[50:61, :] = [
            [1.327773, -0.560], [1.270146, -0.6365], [1.200843, -0.7285],
            [1.136813, -0.8135], [1.077679, -0.892], [1.013649, -0.977],
            [0.951878, -1.059], [0.886342, -1.146], [0.834364, -1.215],
            [0.784270, -1.2815], [0.732669, -1.350],
        ]

        flux[61:67, :] = [
            [0.699524, -1.394], [0.574790, -1.394], [0.474790, -1.394],
            [0.374790, -1.394], [0.263, -1.394], [0.214746, -1.300],
        ]

        flux[67:79, 0] = [0.23, 0.21, 0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01]
        flux[67:79, 1] = 0.05

        flux[79, :] = [1.345, 0.159]

        flux[80:99, :] = self.elect[:19, :2]
        flux[99:118, :] = self.elect0[:19, :2]

        self.flux = flux

    # ------------------------------------------------------------------
    def _build_pf_coil_positions(self) -> None:
        r_c = np.zeros(55)
        z_c = np.zeros(55)

        r_c[[0, 1]] = 0.273, 0.8
        z_c[[0, 1]] = 1.615

        r_c[[2, 5]] = 1.2608
        z_c[[2, 5]] = 1.035, -1.035

        r_c[[3, 4]] = 1.5552
        z_c[[3, 4]] = 0.54, -0.54

        r_c[[6, 7]] = 0.800, 0.273
        z_c[[6, 7]] = -1.615

        r_c[8]  = 0.1574; z_c[8]  = 0.740
        r_c[9]  = 0.1632; z_c[9]  = 0.0
        r_c[10] = 0.1574; z_c[10] = -0.740

        k = 10
        for i in range(8, 11):
            for j in range(1, 9):
                k += 1
                r_c[k] = r_c[i]
                z_c[k] = z_c[i] + j * 0.05

                k += 1
                r_c[k] = r_c[i]
                z_c[k] = z_c[i] - j * 0.05

                if (i == 9 or i == 11) and j == 6:
                    break

        self.r_c = r_c
        self.z_c = z_c

    # ------------------------------------------------------------------
    # ショートカットメソッド (module-level 関数への委譲)
    # ------------------------------------------------------------------

    def is_electrode(self, r: float, z: float) -> bool:
        return elect_posi(r, z)

    def compute_electrode_normals(self, path: str):
        return cal_sn(self.elect0, self.elect, path)

    def load_pf_currents(self):
        return get_PF()
