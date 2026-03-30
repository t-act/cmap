"""結果ファイル出力管理モジュール。

ResultsManager が時刻ごとの保存パスと CSV/コピー操作を一元管理する。
main.py に散在していた np.savetxt / copy_file 呼び出しを集約。
"""

import numpy as np
from cmaplib.utils import copy_file


class ResultsManager:
    """1時刻分の出力先パス管理と保存操作。

    Parameters
    ----------
    path  : ショット番号ルートパス (例: "test30_#53034")
    t_ana : 解析時刻 [ms]
    """

    def __init__(self, path: str, t_ana: float) -> None:
        self.path = path
        self.t_ana = t_ana
        self.time_path = f"{path}/{t_ana:.3f}"

    # ------------------------------------------------------------------
    def save_iteration(
        self,
        i: int,
        mf_output: np.ndarray,
        jf_array: np.ndarray,
        mv_flux: np.ndarray,
        z_Bz_data: np.ndarray,
    ) -> None:
        """1 反復分の CSV データを保存する。

        Parameters
        ----------
        i         : 反復インデックス
        mf_output : 磁場フラックス配列 (nr, nz) — 転置して保存
        jf_array  : 電流ベクトル配列 (N, 4)
        mv_flux   : フラックスループ計算結果 (118, 5)
        z_Bz_data : Bz 比較データ (3, N) — 転置して保存
        """
        np.savetxt(f"{self.time_path}/M_field_{i:03}.csv",
                   mf_output.T, delimiter=",", fmt="%12.4e")
        np.savetxt(f"{self.time_path}/J_field_{i:03}.csv",
                   jf_array, delimiter=",", fmt="%12.4e")
        np.savetxt(f"{self.time_path}/MV_field_flux_{i:03}.csv",
                   mv_flux, fmt="%.3e", delimiter=",")
        np.savetxt(f"{self.path}/data/z_Bz_{self.t_ana:.3f}.csv",
                   z_Bz_data.T, fmt="%.3e", delimiter=",")

    # ------------------------------------------------------------------
    def copy_best(self, i_best: int) -> None:
        """最良反復結果を data/ ディレクトリにコピーする。

        Parameters
        ----------
        i_best : 最良反復インデックス
        """
        for stem in ("M_field", "J_field", "MV_field_flux"):
            src = f"{self.time_path}/{stem}_{i_best:03}.csv"
            dst = f"{self.path}/data/{stem}_{self.t_ana:.3f}.csv"
            copy_file(src, dst)
