#. Last update 2024.10.24


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import get_data as g
from sklearn.metrics import r2_score

from cmaplib.utils import nint, cd_main, copy_file
from cmaplib.tokamak_config import elect_posi, cal_sn, get_PF


def check_con(t_ana, t_inj_0, A, B):
    if t_ana >= t_inj_0:
        return A or B
    else:
        return A and B




def cal_vecp_2(k, r_c, z_c, I_pf_c, r, z):
    """
    Calculate vector potential

    Parameters
    ---
    k       (int)      : Coil number (len(r_c) or len(z_c) or len(I_pf_c))
    r_c     (np.array) : R coil positions
    z_c     (np.array) : Z coil positions
    I_pf_c  (np.array) : Poloidal coil currents
    r       (np.array)
    z       (np.array)

    Returns
    ---
    sbr  (np.array) : Summed magnetic field component in the radial direction.
    sbz  (np.array) : Summed magnetic field component in the vertical direction.
    spsi (np.array) : Summed vector potential.
    ssbz (np.array) : Additional calculated quantity, likely related to the magnetic field in the vertical direction.

    Example
    ---
    >>> _, sbz, _, ssbz = cal_vecp(k, r_c, z_c, I_pf_c, r[ir], z[iz], I_tf_total)
    """
    r_c_sq = r_c**2
    r_sq = r**2
    I_cof = I_pf_c.copy()

    sbr = np.zeros_like(r)
    sbz = np.zeros_like(r)
    spsi = np.zeros_like(r)
    ssbz = np.zeros_like(r)

    for ik in range(k):
        zc = z_c[ik]

        # コイルの位置と入力値が非常に近い場合
        if np.abs(r_c[ik] - r) < 5e-4 and np.abs(z_c[ik] - z) < 5e-4:
            zc += 5e-4
            I_cof[ik] *= 0.5

        z_sq = (z - zc)**2
        rp_sq = (r_c[ik] + r)**2
        rm_sq = (r_c[ik] - r)**2
        rp_z = np.sqrt(rp_sq + z_sq)
        k_sq = 4.0 * r_c[ik] * r / (rp_sq + z_sq)
        e = 1 - k_sq
        lg_e = np.log(1./e, dtype = np.float64)

        # 楕円積分の計算
        KD = (1.38629436112 + 0.09666344259 * e + 0.03590092383 * (e**2) +
              0.03742563713 * (e**3) + 0.01451196212 * (e**4) +
              (0.5 + 0.12498593597 * e + 0.06880248576 * (e**2) +
              0.03328355346 * (e**3) + 0.00441787012 * (e**4)) * lg_e)

        ED = (1.0 + 0.44325141463 * e + 0.0626060122 * (e**2) +
              0.04757383546 * (e**3) + 0.01736506451 * (e**4) +
              (0.2499836831 * e + 0.09200180037 * (e**2) +
              0.04069697526 * (e**3) + 0.00526449639 * (e**4)) * lg_e)

        # 磁場とベクトルポテンシャルの計算
        if np.abs(r) < 1e-6 or np.abs(r_c[ik]) < 1e-6:
            Br = 0
            A_phi = 0
        else:
            Br = 2.0e-7 * ((z - zc) / rp_z) * (-KD + (r_c_sq[ik] + r_sq + z_sq) * ED / (rm_sq + z_sq)) / r
            A_phi = 2.0e-7 * np.sqrt(r_c[ik] / (r * k_sq)) * ((2.0 - k_sq) * KD - 2.0 * ED)

        Bz = 2.0e-7 * (1 / rp_z) * (KD + (r_c_sq[ik] - r_sq - z_sq) * ED / (rm_sq + z_sq))
        psi = 2 * np.pi * r * A_phi

        sbr += Br * I_cof[ik]
        sbz += Bz * I_cof[ik]
        #spsi += mu_0 * I_tf / (2 * np.pi * r)
        spsi = 0
        ssbz += A_phi * I_cof[ik]

    return sbr, sbz, spsi, ssbz



def fitting_bz(z_puc, bz_puc):
    """
    Bz fitting

    Parameter
    ---
    z_puc  (np.array) : [12]
    bz_puc (np.array) : [12]

    Return
    ---
    z_fit   (np.array) : [9] 一番上のコイルを除去 赤道面付近の2個を除去
        Z = 0.537, 0.387, 0.237, 0, -0.213, -0.363, -0.513, -0.663, -0.813 m
    bz_fit  (np.array) : [9] 3次関数でFitting
    weights (np.array) : [9] max に対する重み 1 - 4 (max = 4)
    """
    # コイル高さ降順 z = 0.687 -> -0.813 m
    z_puc_re = np.sort(z_puc)[::-1]
    bz_puc_re = np.concatenate([bz_puc[1:6], [bz_puc[0]], bz_puc[6:]])

    # 一番上のコイル(z = 0.687 m) は信号が怪しいので除外
    z_puc_re = z_puc_re[1:]
    bz_puc_re = bz_puc_re[1:]

    # 3次関数でFitting
    wei = np.ones_like(bz_puc_re)
    wei[0] = 5
    wei[1] = 3
    wei[-1] = 5
    wei[-2] = 3

    coeff = np.polyfit(z_puc_re, bz_puc_re, 3, w = wei)
    fit_func = np.poly1d(coeff)

    # フィッティング曲線をプロットするためのx値
    # z_fit = np.array([
    #     0.537, 0.387, 0.237, 0, -0.213, -0.363, -0.513, -0.663, -0.813
    # ])
    z_fit = np.array([
        0.537, 0.387, 0.237, 0, -0.213, -0.363, -0.513, -0.663, -0.813
    ])
    bz_fit = fit_func(z_fit)

    # weight 計算
    bz_max = np.max(np.abs(bz_fit))
    percentage = np.abs(bz_fit / bz_max) * 100
    # weights = np.where(percentage == 100, 4,  # 100%なら4
    #                np.where(percentage >= 85, 3,  # 95% 以上なら3
    #                         np.where(percentage >= 70, 2,  # 90% 以上80%未満なら2
    #                                  1)))  # 90% 未満なら1
    weights = np.where(percentage >= 85, 3,  # 95% 以上なら3
                            np.where(percentage >= 70, 2,  # 90% 以上80%未満なら2
                                     1))  # 90% 未満なら1

    return z_fit, bz_fit, weights, coeff


def error_bz(Bz_cal_puc, Bz_vac_puc, fit_params, weights):
    """
    Check Bz_cal and bz_puc_fit using OLS

    Parameter
    ---
    Bz_cal_puc (np.array)
    Bz_vac_puc (np.array)
    z_puc_fit  (np.array)
    bz_puc_fit (np.array)

    Return
    ---

    """
    # get data
    z = Bz_cal_puc[:,0]
    bz = Bz_cal_puc[:,1] - Bz_vac_puc[:,1]

    # z, bz を z_puc_fit, bz_puc_fit の形にreshape
    z_re = np.sort(z)[::-1]
    bz_re = np.concatenate([bz[1:6], [bz[0]], bz[6:]])
    z_re = z_re[1:]
    bz_re = bz_re[1:]

    z_re = np.delete(z_re, [3, 5])
    bz_re = np.delete(bz_re, [3, 5])

    fit_func = np.poly1d(fit_params)
    bz_fit = fit_func(z_re)

    resid = (bz_re - bz_fit) * np.sqrt(weights)
    sq_error = np.sum(resid**2)

    return sq_error



def cal_r2(x,y):
    weights = np.ones_like(x)
    weights[0], weights[-1] = 1, 1
    lin_coeffs = np.polyfit(x,y,1,w=weights)
    lin_fit = np.polyval(lin_coeffs, x)
    r2_lin = r2_score(y, lin_fit)

    quad_coeffs = np.polyfit(x,y,2,w=weights)
    quad_fit = np.polyval(quad_coeffs, x)
    r2_quad = r2_score(y, quad_fit)

    return r2_lin, r2_quad,


def get_elf(wq_min, fac):
    mapping = {3: fac[0], 4: fac[1], 5: fac[2], 6: fac[3], 7: fac[4], 8: fac[5], 9: fac[6], 10: fac[7]}
    return mapping.get(wq_min, None)  # Return None if wq[ir, iz] is not in the mapping



# -- plot (cmaplib.plotting への後方互換ラッパー) --
from cmaplib.plotting import (
    plot_magnetic_field, plot_bz_profile,
    plot_psi, plot_t_Bz,
)


def plot_field(m_in, m_out, I_tor, path, time_path, num):
    plot_magnetic_field(
        m_field_path=f"{time_path}/M_field_{num:03}.csv",
        j_field_path=f"{time_path}/J_field_{num:03}.csv",
        electrode_path=f"{path}/electrode.csv",
        save_path=f"{time_path}/img/field_cont_{num:03}.png",
        title=(f"No. = {num:03}, "
               r"$I_\mathrm{tor \text{-} close} = $"
               f"{I_tor*1e-3:.1f} [kA]"),
        m_in=m_in,
        m_out=m_out,
    )


def plot_field2(m_in, t_ana, path):
    plot_magnetic_field(
        m_field_path=f"{path}/data/M_field_{t_ana:.3f}.csv",
        j_field_path=f"{path}/data/J_field_{t_ana:.3f}.csv",
        electrode_path=f"{path}/electrode.csv",
        save_path=f"{path}/img/field_cont_{t_ana:.3f}.png",
        title=f"t = {t_ana:.3f} ms",
        m_in=m_in,
    )


def plot_z_Bz(mv_flux, vac_flux, path, num, z_puc, bz_puc, sq_error):
    plot_bz_profile(
        z=mv_flux[:, 0],
        bz=mv_flux[:, 1],
        bz_vac=vac_flux[:, 1],
        z_puc=z_puc,
        bz_puc=bz_puc,
        save_path=f"{path}/img/Bz-Z_{num}.png",
        title=f"No. {num}, sq_error = {sq_error:.1e}",
    )


def plot_z_Bz2(path, t_ana, z_puc, bz_puc, sq_error):
    data = np.loadtxt(f"{path}/data/z_Bz_{t_ana:.3f}.csv", delimiter=",")
    plot_bz_profile(
        z=data[:, 0],
        bz=data[:, 1],
        bz_vac=data[:, 2],
        z_puc=z_puc,
        bz_puc=bz_puc,
        save_path=f"{path}/img/Bz-Z_{t_ana:.3f}.png",
        title=f"t = {t_ana:.3f}, sq_error = {sq_error:.1e}",
    )



#. test
if __name__ == "__main__":
    # Example usage
    cd_main()
    s = g.get_CHI_Data(53044, True)
    t_puc, bz_puc = s.get_bz()
    z_puc = np.array([0 if i == 0 else 687e-3 - ((i-1) * 150e-3) for i in range(12)])
    print(z_puc)
    z_puc_re = np.concatenate([z_puc[1:6], [z_puc[0]], z_puc[6:]])
    print(z_puc_re)
    z_puc_re_2 = np.sort(z_puc)[::-1]
    print(z_puc_re_2)


    # get index of t_ana
    t_ana_arr = [19.4]
    #t_ana_arr = np.arange(18.9, 19.7, 0.01)

    r2_lin = []
    r2_quad = []
    for t_ana in t_ana_arr:
        print(t_ana)
        t_puc, bz_puc = s.get_bz()
        it = np.where(np.isclose(t_puc*1e3, t_ana, atol = 5e-4))[0][0]
        z = z_puc
        bz = bz_puc[it,:]

        # z_re = np.sort(z)[::-1]
        # bz_re = np.concatenate([bz[1:6], [bz[0]], bz[6:]])
        # z_re = z_re[1:]
        # bz_re = bz_re[1:]

        # z_re = np.delete(z_re, [3, 5])
        # bz_re = np.delete(bz_re, [3, 5])

        plt.scatter(bz, z)
        plt.show()
