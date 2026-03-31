"""時刻ステップ前処理モジュール。

prepare_time_step() が 1 時刻分の EquilibriumSolver 入力パラメータを
一括生成する。main.py の時間ループ内前処理ブロックを集約。
"""

import numpy as np

from cmaplib.electromagnetics import cal_vecp_2_grid, elect_posi_grid
from cmaplib.utils import nint
from mfield_sub import cal_vecp_2, fitting_bz


_K_PF = 10  # PF コイル本数 (インデックス 0–10)


def prepare_time_step(
    t_ana: float,
    t_decay: float,
    t_inj_0: float,
    count: int,
    grid,
    config,
    t_ip: np.ndarray,
    ip: np.ndarray,
    t_g: np.ndarray,
    G_array: np.ndarray,
    t_puc: np.ndarray,
    bz_puc: np.ndarray,
    z_puc: np.ndarray,
    path: str,
    time_path: str,
    pi: float,
) -> dict:
    """1 時刻分の前処理を実行し、solve_time_step() 用パラメータ辞書を返す。

    Parameters
    ----------
    t_ana, t_decay, t_inj_0 : 解析・減衰・注入開始時刻 [ms]
    count    : ショット番号 (ログ出力用)
    grid     : Grid インスタンス
    config   : TokamakConfig インスタンス
    t_ip, ip : プラズマ電流時系列
    t_g, G_array : Gun 電流時系列
    t_puc, bz_puc : ピックアップコイル時系列
    z_puc    : ピックアップコイル Z 座標 [m]
    path, time_path : 出力パス
    pi       : 円周率定数

    Returns
    -------
    dict  — EquilibriumSolver.solve_time_step() の引数に対応するキー群
    """
    r, z   = grid.r, grid.z
    dr, dz = grid.dr, grid.dz
    r_min  = grid.r_min
    z_min  = grid.z_min
    r_c    = config.r_c
    z_c    = config.z_c
    flux   = config.flux

    # -- PF / TF コイル電流 --
    I_pf_c, I_tf_total = config.build_pf_coil_currents()

    # -- 電極法線ベクトル + electrode.csv 保存 --
    _, _, sn_r, sn_z = config.compute_electrode_normals(path)

    # -- 真空場計算 --
    _, _Bz, _, A_phi = cal_vecp_2_grid(_K_PF, r_c, z_c, I_pf_c, r, z)
    A_phi_0    = A_phi.copy()
    mv_field   = A_phi_0 * 2*pi * r[:, np.newaxis]
    R_grid, Z_grid = np.meshgrid(r, z, indexing='ij')
    mv_field_l = np.where(elect_posi_grid(R_grid, Z_grid), 1e29,
                           A_phi * 2*pi * r[:, np.newaxis])
    np.savetxt(f"{path}/data/MV_field.csv",   mv_field.T,   delimiter=",", fmt="%12.4e")
    np.savetxt(f"{path}/data/MV_field_l.csv", mv_field_l.T, delimiter=",", fmt="%12.4e")

    vac_flux = []
    for i in range(118):
        if flux[i, 0] > 0:
            sbr, sbz, _, ssbz = cal_vecp_2(_K_PF, r_c, z_c, I_pf_c, flux[i, 0], flux[i, 1])
            vac_flux.append([flux[i,0], flux[i,1], ssbz*2*pi*flux[i,0], sbr, sbz])
    np.savetxt(f"{path}/MV_field_flux_000.csv", np.array(vac_flux),
               delimiter=",", fmt='%.4e', header="R, Z, A_phi, Br, Bz", comments=" ")

    # -- ピックアップコイル Bz 読み込み + フィッティング --
    it = np.where(np.isclose(t_puc*1e3, t_ana, atol=5e-4))[0][0]
    bz_puc_it = bz_puc[it, :]
    z_puc_fit, _bz_fit, weights, fit_params = fitting_bz(z_puc, bz_puc_it)

    iz_puc     = np.array([np.where(np.isclose(z, zp,  atol=1e-2))[0][0] for zp in z_puc])
    iz_puc_fit = np.array([np.where(np.isclose(z, zpf, atol=1e-2))[0][0] for zpf in z_puc_fit])
    ir_puc     = np.where(r == 0.22)[0][0]

    z_re  = np.sort(z_puc)[::-1]
    bz_re = np.concatenate([bz_puc_it[1:6], [bz_puc_it[0]], bz_puc_it[6:]])
    z_re  = np.delete(z_re[1:],  [3, 5])
    bz_re = np.delete(bz_re[1:], [3, 5])

    # -- フィラメント電流位置 --
    if t_decay <= t_ana < t_inj_0:
        ikz0    = iz_puc_fit
        ikz_dir = 1
    elif t_ana >= t_inj_0:
        iz_puc_min = np.abs(z - z_re[np.argmin(bz_re)]).argmin()
        ikz0    = iz_puc_min
        ikz_dir = 1
    else:
        ikz0    = nint((-1.5 - z_min)/dz)
        ikz_dir = 1

    if t_ana >= t_decay:
        t1, r1, t2, r2_val = 18.9, 0.6, 19.7, 0.3
        kr0 = (r2_val - r1)/(t2 - t1) * t_ana + r1 - (r2_val - r1)/(t2 - t1) * t1
    else:
        kr0 = 0.3
    ikr0 = nint((kr0 - r_min)/dr)

    # -- elf0 設定 --
    if t_decay <= t_ana < t_inj_0:
        elf0 = np.linspace(1, 0.3, 8)
    elif t_ana >= t_inj_0:
        a, b, c = np.polyfit([19.2, 19.4, 19.6], [1, 3, 6.5], 2)
        elf0 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])/5 * (a*t_ana**2 + b*t_ana + c)
    else:
        elf0 = np.ones(8)

    # -- 入射電流設定 --
    it_g = np.where(np.isclose(t_g*1e3, t_ana, atol=1e-4))[0][0]
    G = np.array([G_array[it_g,0], np.sum(G_array[it_g,1:4]), G_array[it_g,4]])
    G[-100 < G] = 0
    G_round = np.round(G, -2)
    if t_ana >= t_inj_0 and G_round[1] == 0:
        G_round[1] = -100

    I_inA = np.zeros(21)
    I_inA[2:4] = 1;  I_inA[4:8] = 2;  I_inA[8:10] = 3
    I_inj_total = np.sum(G)
    I_inj = np.zeros(21)
    I_inj[1:4] = G_round[0:3]

    # -- トロイダル電流目標 --
    it_ip     = np.where(np.isclose(t_ip*1e3, t_ana, atol=5e-4))[0][0]
    I_tor_def = -np.round(ip[it_ip]*1e3, -2)

    # -- ログファイル --
    with open(f"{time_path}/_log.csv", "w") as fh:
        fh.write(f"#{count}, t_ana = {t_ana:.3f} ms\n")
        fh.write("PF3-1, 3-2, 2, 1, 7, 6, 5-2, 5-1, 4-1, 4-2, 4-3\n")
        fh.write(", ".join(str(v) for v in I_pf_c[:11]) + "\n")
        fh.write(f"I_tor_def [A], {I_tor_def}\n")
        fh.write(f"G [A], {G_round[0]}, {G_round[1]}, {G_round[2]}\n")
        fh.write(f"elf0 ,{elf0}\n")
        fh.write(f"t_decay [ms],{t_decay:.2f}\n")
        fh.write("z [m], " + ", ".join(f"{zf:.2f}" for zf in z_puc_fit) + "\n")
        fh.write("bz weight, " + ", ".join(str(w) for w in weights))

    # -- ピックアップコイル位置の真空場 Bz --
    Bz_vac_puc = np.zeros((len(bz_puc_it), 2))
    for j, jz in enumerate(iz_puc):
        Bz_vac_puc[j, 0] = z[jz]
        _, Bz_vac_puc[j, 1], _, _ = cal_vecp_2(_K_PF, r_c, z_c, I_pf_c, r[ir_puc], z[jz])

    return {
        "A_phi_0":     A_phi_0,
        "I_tf_total":  I_tf_total,
        "elf0":        elf0,
        "weights":     weights,
        "iz_puc":      iz_puc,
        "ir_puc":      ir_puc,
        "fit_params":  fit_params,
        "Bz_vac_puc":  Bz_vac_puc,
        "I_tor_def":   I_tor_def,
        "I_inj":       I_inj,
        "I_inA":       I_inA,
        "I_inj_total": I_inj_total,
        "ikr0":        ikr0,
        "ikz0":        ikz0,
        "ikz_dir":     ikz_dir,
        "sn_r":        sn_r,
        "sn_z":        sn_z,
        "bz_puc_it":   bz_puc_it,
    }
