"""グリーン関数行列 A_0 の管理とテンソル集約モジュール。

A_0 (101×201×101×201 ≈ 3.4 GB) の読み込みと、
開/閉磁気面のベクトルポテンシャル寄与を一括計算する。
"""

import numpy as np


class GreenFunction:
    """グリーン関数行列 A_0 の管理クラス。

    Attributes
    ----------
    A_0 : np.ndarray  (nr, nz, nr, nz)
        各ソース点 (ir2, iz2) がグリッド点 (ir, iz) に作る A_phi の行列。
    """

    def __init__(self, bin_path: str = "modules/mfile_py.bin",
                 nr: int = 101, nz: int = 201) -> None:
        self.nr = nr
        self.nz = nz
        self.ir_max = nr - 1  # 100
        self.iz_max = nz - 1  # 200
        self.A_0 = self._load(bin_path)

    # ------------------------------------------------------------------
    def _load(self, bin_path: str) -> np.ndarray:
        return (np.fromfile(bin_path, dtype=np.float64)
                .reshape(self.nr, self.nz, self.nr, self.nz))

    # ------------------------------------------------------------------
    def accumulate_fields(
        self,
        wq_f: np.ndarray,
        elf: np.ndarray,
        lamb: np.ndarray,
        r: np.ndarray,
        dr: float,
        dz: float,
        mu: float,
        I_tf_total: float,
        pi: float,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """開/閉磁気面からのベクトルポテンシャル寄与をベクトル化計算する。

        main.py の二重ループ (L775-799) を numpy テンソル縮約で置換。
        ~100x 高速化。

        Parameters
        ----------
        wq_f   : (ir_max, iz_max, 4) int  磁力線追跡結果
        elf    : (ir_max, iz_max)          電極電流補正係数
        lamb   : (20,)                     λ 係数
        r      : (nr,)                     R グリッド
        dr, dz : float
        mu     : float  真空透磁率
        I_tf_total : float  TF コイル電流 [A]
        pi     : float

        Returns
        -------
        A_phi_close      : (nr, nz)
        A_phi_open       : (nr, nz)
        I_tor_close      : float         閉磁気面電流合計
        I_tor_open_delta : (20,)         開磁気面電流の各電極への寄与
        wq_l             : (ir_max, iz_max) int  最内閉磁気面追跡配列
        """
        ir_max = self.ir_max
        iz_max = self.iz_max

        r_mid = (r[:ir_max] + 0.5 * dr)[:, np.newaxis]  # (ir_max, 1)
        coeff = mu * I_tf_total / (2 * pi) * dr * dz       # scalar

        wq_f3 = wq_f[:ir_max, :iz_max, 3]                # (ir_max, iz_max)
        close_mask = wq_f3 == 20                           # bool
        open_mask  = (wq_f3 != 0) & ~close_mask            # bool

        # ---- 閉磁気面 ------------------------------------------------
        I_close_grid = (coeff / r_mid) * close_mask        # (ir_max, iz_max)
        I_tor_close  = float(np.sum(I_close_grid))

        # wq_l: 閉磁気面セルに ir インデックスを格納
        ir_idx = np.broadcast_to(np.arange(ir_max)[:, np.newaxis],
                                  (ir_max, iz_max))
        wq_l = np.where(close_mask, ir_idx, 1000)

        # ---- 開磁気面 ------------------------------------------------
        # lamb は 20 要素 (インデックス 0-19)、wq_f3 は 1-20 の電極インデックス
        safe_idx  = np.clip(wq_f3 - 1, 0, 19)
        lamb_grid = np.where(open_mask, lamb[safe_idx], 0.0)
        I_open_grid = (coeff / r_mid) * lamb_grid * elf[:ir_max, :iz_max] * open_mask

        # scatter-add: 各電極への電流寄与
        I_tor_open_delta = np.zeros(20)
        if np.any(open_mask):
            np.add.at(I_tor_open_delta, wq_f3[open_mask] - 1, I_open_grid[open_mask])

        # ---- テンソル縮約 (P3 最適化) ---------------------------------
        # A_0[:,:,:ir_max,:iz_max] は shape (nr, nz, ir_max, iz_max)
        # einsum('ijkl,kl->ij', ...) で (ir_max, iz_max) を縮約 → (nr, nz)
        A_0_view = self.A_0[:, :, :ir_max, :iz_max]
        A_phi_close = np.einsum('ijkl,kl->ij', A_0_view, I_close_grid)
        A_phi_open  = np.einsum('ijkl,kl->ij', A_0_view, I_open_grid)

        return A_phi_close, A_phi_open, I_tor_close, I_tor_open_delta, wq_l
