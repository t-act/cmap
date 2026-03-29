"""磁力線追跡モジュール。

main.py の track_mag_line + 二重グリッドループを
Numba JIT + prange 並列化で置換する。

期待高速化: ~300-600x (20k点×2方向×最大4000ステップ)
"""

import numpy as np
import numba


@numba.njit(cache=True)
def _trace_single(
    ir0: int,
    iz0: int,
    i_dir: int,
    rr10: float,
    flux_mag_r: np.ndarray,
    flux_mag_z: np.ndarray,
    A_phi: np.ndarray,
    wq0: np.ndarray,
    wq: np.ndarray,
    r: np.ndarray,
    nr: int,
    nz: int,
) -> int:
    """1本の磁力線を最大 4000 ステップ追跡し、終点の wq 値を返す。

    Parameters
    ----------
    ir0, iz0 : 開始グリッドインデックス
    i_dir    : 追跡方向 (1=前進, 2=後退)
    rr10     : 参照 A_phi 評価値
    wq0      : 追跡済みマーク配列 (in-place 修正)
    wq       : 容器壁マスク (変更しない)
    r        : R グリッド配列
    nr, nz   : グリッドサイズ

    Returns
    -------
    int : 終点の wq 値 (20=閉磁気面, 1-19=電極インデックス, 0=未確定)
    """
    sign = 1 if i_dir == 1 else -1
    jr, jz = ir0, iz0

    for _ in range(4000):
        wq0[jr, jz] = 20

        # 磁場方向を明示的条件分岐で決定 (Numba互換)
        fmr = flux_mag_r[jr, jz] * sign
        if fmr > 0.0:
            ir_ad = 1
        elif fmr < 0.0:
            ir_ad = -1
        else:
            ir_ad = 0

        fmz = flux_mag_z[jr, jz] * sign
        if fmz > 0.0:
            iz_ad = 1
        elif fmz < 0.0:
            iz_ad = -1
        else:
            iz_ad = 0

        # 候補方向 (j=0: Z方向, j=1: R方向, j=2: 対角方向)
        i_ad = -1
        A_phi_eva = 1e29

        for j in range(3):
            if j == 0:
                ir_off_j = 0
                iz_off_j = iz_ad
            elif j == 1:
                ir_off_j = ir_ad
                iz_off_j = 0
            else:
                ir_off_j = ir_ad
                iz_off_j = iz_ad

            ir_eva = jr + ir_off_j
            iz_eva = jz + iz_off_j

            if ir_eva < 0 or ir_eva >= nr - 1 or iz_eva < 0 or iz_eva >= nz - 1:
                continue

            A_new = abs(
                (A_phi[ir_eva, iz_eva] + A_phi[ir_eva, iz_eva + 1]) * r[ir_eva]
                + (A_phi[ir_eva + 1, iz_eva] + A_phi[ir_eva + 1, iz_eva + 1]) * r[ir_eva + 1]
                - rr10
            )

            if A_phi_eva > A_new:
                A_phi_eva = A_new
                i_ad = j

        if i_ad == -1:
            return 0

        if i_ad == 0:
            ir_off = 0
            iz_off = iz_ad
        elif i_ad == 1:
            ir_off = ir_ad
            iz_off = 0
        else:
            ir_off = ir_ad
            iz_off = iz_ad

        next_r = jr + ir_off
        next_z = jz + iz_off

        # 容器外に出た場合 → 現在位置の電極インデックスを返す
        if wq[next_r, next_z] == -1:
            return int(wq[jr, jz])

        # 既に追跡済みセル (閉磁気面マーク) に到達
        if wq0[next_r, next_z] == 20:
            return 20

        jr = next_r
        jz = next_z

    return 0


@numba.njit(parallel=True, cache=True)
def trace_all_field_lines(
    A_phi: np.ndarray,
    flux_mag_r: np.ndarray,
    flux_mag_z: np.ndarray,
    r: np.ndarray,
    wq: np.ndarray,
) -> np.ndarray:
    """グリッド全点の磁力線を prange で並列追跡する。

    main.py の二重ループ + track_mag_line を置換。
    iz 方向を prange で並列化し、各スレッドが wq0 バッファを 1 個保持。

    Parameters
    ----------
    A_phi       : (nr, nz)     ベクトルポテンシャル
    flux_mag_r  : (ir_max, iz_max)  Br 磁場
    flux_mag_z  : (ir_max, iz_max)  Bz 磁場
    r           : (nr,)        R グリッド
    wq          : (nr, nz) int 容器壁マスク

    Returns
    -------
    wq_f : (ir_max, iz_max, 4) int
        [:, :, 1] = 前進方向追跡結果
        [:, :, 2] = 後退方向追跡結果
        [:, :, 0], [:, :, 3] = 後処理で設定 (この関数では 0)
    """
    nr = wq.shape[0]
    nz = wq.shape[1]
    ir_max = nr - 1   # 100
    iz_max = nz - 1   # 200

    wq_f = np.zeros((ir_max, iz_max, 4), dtype=np.int64)

    for iz in numba.prange(iz_max):
        # スレッドごとに wq0 バッファを 1 個確保
        wq0 = wq.copy()

        for ir in range(ir_max):
            if wq[ir, iz] != -1:
                # この (ir, iz) 用に wq0 をリセット (in-place、メモリ再利用)
                for i in range(nr):
                    for j in range(nz):
                        wq0[i, j] = wq[i, j]

                rr10 = (
                    (A_phi[ir, iz] + A_phi[ir, iz + 1]) * r[ir]
                    + (A_phi[ir + 1, iz] + A_phi[ir + 1, iz + 1]) * r[ir + 1]
                )

                # 前進方向 (i_dir=1) — wq0 を in-place 修正
                wq_f[ir, iz, 1] = _trace_single(
                    ir, iz, 1, rr10,
                    flux_mag_r, flux_mag_z, A_phi, wq0, wq, r, nr, nz
                )
                # 後退方向 (i_dir=2) — 前進で修正された wq0 を共有 (原コードと同じ挙動)
                wq_f[ir, iz, 2] = _trace_single(
                    ir, iz, 2, rr10,
                    flux_mag_r, flux_mag_z, A_phi, wq0, wq, r, nr, nz
                )

    return wq_f
