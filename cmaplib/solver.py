"""磁平衡収束ループ管理モジュール。

EquilibriumSolver が磁力線追跡→λ計算→A_phi 更新→保存判定の
収束ループを一元管理する。
"""

from copy import deepcopy

import numpy as np
from scipy.ndimage import minimum_position
from tqdm import tqdm

from cmaplib.electromagnetics import compute_B_field
from cmaplib.field_line_tracer import trace_all_field_lines, _trace_single
from cmaplib.plotting import plot_magnetic_field, plot_bz_profile
from cmaplib.utils import nint
from mfield_sub import error_bz, get_elf


class EquilibriumSolver:
    """磁平衡収束ループ管理クラス。

    時刻ループをまたいで不変なパラメータをコンストラクタで受け取り、
    時刻ごとの収束ループを solve_time_step() で実行する。

    Parameters
    ----------
    grid   : Grid インスタンス
    config : TokamakConfig インスタンス
    wq     : グリッド要素割り当て配列 (ir_max, iz_max)
    gf     : GreenFunction インスタンス
    pi, mu : 物理定数
    """

    def __init__(self, grid, config, wq, gf, pi, mu) -> None:
        self.r       = grid.r
        self.z       = grid.z
        self.dr      = grid.dr
        self.dz      = grid.dz
        self.dl      = grid.dl
        self.pi      = pi
        self.mu      = mu
        self.elect   = config.elect
        self.flux    = config.flux
        self.ele_lim = config.ele_lim
        self.wq      = wq
        self.gf      = gf
        self.ir_max  = grid.ir_max
        self.iz_max  = grid.iz_max
        self.r_min   = grid.r_min
        self.z_min   = grid.z_min

    # ------------------------------------------------------------------
    @staticmethod
    def _cal_B(ir, iz, dr, dz, A_phi, r):
        """A_phi から中心差分で Br, Bz を計算する。"""
        Br = -(A_phi[ir, iz+1] - A_phi[ir, iz-1]) / (2*dz)
        Bz = (1/r[ir]) * (A_phi[ir+1, iz]*r[ir+1] - A_phi[ir-1, iz]*r[ir-1]) / (2*dr)
        return Br, Bz

    def _cal_flux(self, ir, iz, A_phi):
        """ψ = 2π r A_phi を返す。"""
        return 2 * self.pi * self.r[ir] * A_phi[ir, iz]

    def _update_flux_val(self, ir, iz, f_max_l, f_min_l, idx, A_phi):
        """フラックス最大/最小値を更新する。"""
        flux_eva = self._cal_flux(ir, iz, A_phi)
        if f_max_l[idx] > flux_eva:
            f_max_l[idx] = flux_eva
        if f_min_l[idx] < flux_eva:
            f_min_l[idx] = flux_eva

    # ------------------------------------------------------------------
    def solve_time_step(
        self,
        t_ana: float,
        t_decay: float,
        t_inj_0: float,
        A_phi_0: np.ndarray,
        I_tf_total: float,
        elf0: np.ndarray,
        weights: np.ndarray,
        iz_puc: np.ndarray,
        ir_puc: int,
        fit_params: np.ndarray,
        Bz_vac_puc: np.ndarray,
        I_tor_def: float,
        I_inj: np.ndarray,
        I_inA: np.ndarray,
        I_inj_total: float,
        ikr0: int,
        ikz0,
        ikz_dir: int,
        sn_r: np.ndarray,
        sn_z: np.ndarray,
        z_puc: np.ndarray,
        bz_puc_it: np.ndarray,
        rm,
        path: str,
        time_path: str,
    ) -> dict:
        """収束ループを実行して最良解を返す。

        Returns
        -------
        dict
            A_phi         : 収束後のベクトルポテンシャル
            m_in          : 最内閉磁気面フラックス値
            I_close_store : 保存した反復インデックスと閉磁気面電流のリスト
            sq_error      : 各反復の Bz 二乗誤差配列
        """
        r       = self.r
        z       = self.z
        dr      = self.dr
        dz      = self.dz
        pi      = self.pi
        mu      = self.mu
        dl      = self.dl
        elect   = self.elect
        flux    = self.flux
        ele_lim = self.ele_lim
        wq      = self.wq
        gf      = self.gf
        ir_max  = self.ir_max
        iz_max  = self.iz_max
        r_min   = self.r_min
        z_min   = self.z_min
        A_0     = gf.A_0

        # -- 収束ループ用変数初期化 --
        A_phi         = A_phi_0.copy()
        SM            = 0
        SMmin         = np.full(3, 1e30)
        RSM           = 1e29
        rh            = 0.0
        I_tor         = np.zeros(20)
        lamb          = np.zeros(20)
        J_inj_p0      = np.zeros(20)
        J_inj_p       = np.zeros(20)
        J_inj_p2      = np.zeros(20)
        J_inj_e       = np.zeros(20)
        J_inj_e0      = np.zeros(20)
        flux_mag_r    = np.zeros((ir_max, iz_max))
        flux_mag_z    = np.zeros((ir_max, iz_max))
        ik            = 0
        ikr, ikz      = ikr0, ikz0
        plot_counter  = 0
        i_max         = 1000 if t_ana < t_decay else 100
        I_close_store = []
        sq_error      = np.ones(i_max)
        m_in          = 0.0

        # == 収束ループ ==
        for i in tqdm(range(i_max), leave=False, desc="i"):
            I_tor_def_new = -150*1e3 if (i == 0 and t_ana >= t_inj_0) else deepcopy(I_tor_def)

            f_max_l = np.zeros(21)
            f_min_l = np.zeros(21)
            elf     = np.ones((ir_max, iz_max))

            # -- 磁力線追跡 --
            wq_f  = np.zeros((ir_max, iz_max, 4), dtype=int)
            I_tor[:] = 0

            flux_mag_r, flux_mag_z = compute_B_field(A_phi, r, dr, dz, ir_max, iz_max)

            if rh > 1e29:
                ir0 = nint((elect[0, 0] - r_min)/dr) - 1
                iz0 = nint((elect[0, 1] - z_min)/dz)
                if wq[ir0, iz0] != -1:
                    wq0_s = wq.copy()
                    rr10  = ((A_phi[ir0, iz0] + A_phi[ir0, iz0+1])*r[ir0]
                             + (A_phi[ir0+1, iz0] + A_phi[ir0+1, iz0+1])*r[ir0+1])
                    wq_f[ir0, iz0, 1] = _trace_single(
                        ir0, iz0, 1, rr10, flux_mag_r, flux_mag_z,
                        A_phi, wq0_s, wq, r, ir_max+1, iz_max+1)
                    wq_f[ir0, iz0, 2] = _trace_single(
                        ir0, iz0, 2, rr10, flux_mag_r, flux_mag_z,
                        A_phi, wq0_s, wq, r, ir_max+1, iz_max+1)

            wq_f_traced = trace_all_field_lines(A_phi, flux_mag_r, flux_mag_z, r, wq)
            wq_f[:, :, 1] = wq_f_traced[:, :, 1]
            wq_f[:, :, 2] = wq_f_traced[:, :, 2]

            for iz in range(iz_max):
                for ir in range(ir_max):
                    if wq[ir, iz] != -1:
                        r_mid = r[ir] + 0.5*dr
                        if wq_f[ir,iz,1] == 20 and wq_f[ir,iz,2] == 20:
                            wq_f[ir, iz, 0] = 20
                        else:
                            if ((wq_f[ir,iz,2] == 1 or wq_f[ir,iz,2] == 2) and
                                    wq_f[ir,iz,1] not in (1, 2, 20) and
                                    elect[wq_f[ir,iz,1]-1, 2] != 0):
                                wq_f[ir, iz, 0] = wq_f[ir, iz, 1]
                                I_tor[wq_f[ir,iz,0]-1] += (
                                    mu*I_tf_total/(2*pi*r_mid)
                                    * elect[wq_f[ir,iz,0]-1, 2]*dr*dz)
                            elif ((wq_f[ir,iz,1] == 1 or wq_f[ir,iz,1] == 2) and
                                    wq_f[ir,iz,2] not in (1, 2, 20) and
                                    elect[wq_f[ir,iz,2]-1, 2] != 0):
                                wq_f[ir, iz, 0] = wq_f[ir, iz, 2]
                                I_tor[wq_f[ir,iz,0]-1] += (
                                    mu*I_tf_total/(2*pi*r_mid)
                                    * elect[wq_f[ir,iz,0]-1, 2]*dr*dz)
                            elif (np.all(elect[wq_f[ir,iz,1:3]-1, 2] > 0.1) and
                                    np.all(wq_f[ir,iz,1:3] != 20)):
                                if 3 <= wq_f[ir,iz,1] <= 10 and 3 <= wq_f[ir,iz,2] <= 10:
                                    wq_f[ir,iz,3] = wq_f[ir,iz,1]
                                    elf[ir,iz] = get_elf(
                                        min(wq_f[ir,iz,1], wq_f[ir,iz,2]), elf0)
            wq_f[:, :, 3] += wq_f[:, :, 0]

            # -- λ 計算 --
            J_inj_p0[:], J_inj_p[:], J_inj_p2[:] = 0, 0, 0
            J_inj_e0[:], J_inj_e[:] = 0, 0
            f_max, f_min = -1.e29, 1.e29

            for i_ele in range(19):
                w_ele = elect[i_ele+1, 0] - elect[i_ele, 0]
                h_ele = elect[i_ele+1, 1] - elect[i_ele, 1]
                k0    = nint(np.sqrt(w_ele**2 + h_ele**2)/dl)

                if elect[i_ele, 0] > 0.1 and elect[i_ele+1, 0] > 0.1 and k0 != 0:
                    w_ele = (elect[i_ele+1, 0] - elect[i_ele, 0])/k0
                    h_ele = (elect[i_ele+1, 1] - elect[i_ele, 1])/k0
                    f_min_l[i_ele], f_max_l[i_ele] = -1.e29, 1.e29

                    for k in range(int(k0)):
                        ir_mid = nint((elect[i_ele, 0] + w_ele*(0.5+k) - r_min)/dr)
                        iz_mid = nint((elect[i_ele, 1] + h_ele*(0.5+k) - z_min)/dz)
                        ir_ele = nint((elect[i_ele, 0] + w_ele*k - r_min)/dr)
                        iz_ele = nint((elect[i_ele, 1] + h_ele*k - z_min)/dz)
                        Br, Bz = self._cal_B(ir_mid, iz_mid, dr, dz, A_phi, r)
                        j_val  = (Br*sn_r[i_ele] + Bz*sn_z[i_ele])*elect[i_ele,2]*elf[ir,iz]/k0
                        if wq_f[ir_mid, iz_mid, 0] != 0:
                            J_inj_p0[i_ele] += j_val
                        if wq_f[ir_mid, iz_mid, 3] != 0:
                            J_inj_p[i_ele]  += j_val
                        J_inj_p2[i_ele] += j_val
                        self._update_flux_val(ir_mid, iz_mid, f_max_l, f_min_l, i_ele, A_phi)
                        self._update_flux_val(ir_ele, iz_ele, f_max_l, f_min_l, i_ele, A_phi)

                    fm = np.array([f_min_l[i_ele], f_max_l[i_ele]])
                    if np.min(fm) < f_min:
                        f_min = np.min(fm)
                    if np.max(fm) > f_max:
                        f_max = np.max(fm)

            lamb[:] = 0.
            emh = True
            for i_ele in range(19):
                k1 = int(I_inA[i_ele])
                J_inj_e[k1]  += J_inj_p[i_ele]
                J_inj_e0[k1] += J_inj_p0[i_ele]

            for i_ele in range(19):
                k1 = int(I_inA[i_ele])
                if abs(J_inj_e[k1]/sum(J_inj_e[:])) > 0.01 and k1 != 0:
                    if J_inj_e[k1]/J_inj_e0[k1] > 0:
                        lamb[i_ele] = elect[i_ele, 2]*I_inj[k1]/J_inj_e[k1]
                    else:
                        emh = False
                elif (abs(J_inj_e[k1]/sum(J_inj_e[:])) <= 0.01
                      and k1 != 0 and abs(I_inj[k1]) > 0.01):
                    emh = False
                I_tor[i_ele]    *= lamb[i_ele]
                J_inj_p2[i_ele] *= lamb[i_ele]

            lamb[-1] = sum(lamb[2:18])/sum(elect[2:18, 2])*elect[19, 2]

            # -- トロイダル電流 + A_phi 更新 --
            A_phi_before = deepcopy(A_phi)
            I_tor[:] = 0

            A_phi_close, A_phi_open, _I_tor_close, _I_tor_open_delta, wq_l = \
                gf.accumulate_fields(wq_f, elf, lamb, r, dr, dz, mu, I_tf_total, pi)
            I_tor[:20] = _I_tor_open_delta
            I_tor[-1]  = _I_tor_close

            Itor_cal_inp = np.abs(I_tor_def_new) - np.abs(np.sum(I_tor[0:20]))
            rh = 1.e30

            lambda_fac = 2/mu if t_ana >= t_decay else np.max(np.abs(lamb[0:20]))

            if Itor_cal_inp > 0:
                if ((2*np.abs(I_tor[-1])*lambda_fac > Itor_cal_inp) and
                        (np.abs(I_tor[-1])*lambda_fac > Itor_cal_inp or SM) > 1e-8):
                    lamb[-1]    = (I_tor_def_new - np.sum(I_tor[0:20])) / I_tor[-1]
                    A_phi[:, :] = deepcopy(A_phi_0)
                else:
                    lamb[-1] = 0
                    ik += 1
                    if ik == 1:
                        ikr, ikz = ikr0, ikz0
                    elif ik == 2:
                        ikr, ikz = ikr0+1, ikz0
                    elif ik == 3:
                        ikr, ikz = ikr0, ikz0+ikz_dir
                    else:
                        if SMmin[1] < SMmin[2]:
                            ikr0 += 1
                        else:
                            ikz0 += ikz_dir
                        ik = 2;  ikr = ikr0+1;  ikz = ikz0
                        SMmin[:] = 1.e30

                    if ikr > ir_max-1 or np.any(ikz > iz_max-1):
                        break

                    if t_decay <= t_ana < t_inj_0:
                        A_phi[:, :] = A_phi_0 + (I_tor_def_new - np.sum(I_tor[0:20])) / 9 \
                                      * np.sum([weights[ii]*A_0[:, :, ikr, ikz[ii]]
                                                for ii in range(9)], axis=0) / np.sum(weights)
                    elif t_ana >= t_inj_0 or t_ana < t_decay:
                        A_phi[:, :] = A_phi_0 + (I_tor_def_new - np.sum(I_tor[0:20])) / 9 * (
                            A_0[:,:,ikr,ikz-1] + A_0[:,:,ikr,ikz]   + A_0[:,:,ikr,ikz+1]
                          + A_0[:,:,ikr-1,ikz-1] + A_0[:,:,ikr-1,ikz] + A_0[:,:,ikr-1,ikz+1]
                          + A_0[:,:,ikr+1,ikz-1] + A_0[:,:,ikr+1,ikz] + A_0[:,:,ikr+1,ikz+1])

                    rh = I_tor_def_new - np.sum(I_tor[:-1])

            I_tor[-1]    *= lamb[-1]
            A_phi[:, :]  += A_phi_close*lamb[-1] + A_phi_open

            SM = np.sum((A_phi - A_phi_before)**2) / ((ir_max+1)*(iz_max+1))
            if SMmin[ik-1] > SM and rh > 1.e29:
                SMmin[ik-1] = SM

            # -- 保存 / プロット --
            if t_ana >= t_decay:
                update_con = SM < max(RSM, 1e-8) and rh > 1e29 and emh
            else:
                update_con = (SM < max(RSM, 1e-8) and rh > 1e29 and
                              lambda_fac > np.abs(lamb[-1]) and emh)

            if update_con:
                plot_counter += 1
                RSM       = SM
                mf_output = A_phi * 2*pi * r[:, np.newaxis]

                flux_mag_max = max(np.max(np.abs(flux_mag_r)), np.max(np.abs(flux_mag_z)))
                jf = []
                for iz in range(iz_max):
                    for ir in range(ir_max):
                        if iz%3 == 0 and ir%3 == 0:
                            br = -((A_phi_before[ir,iz+1] + A_phi_before[ir+1,iz+1])*0.5
                                  - (A_phi_before[ir,iz]   + A_phi_before[ir+1,iz])*0.5) / dz
                            bz = ((A_phi_before[ir+1,iz] + A_phi_before[ir+1,iz+1])*0.5*r[ir+1]
                                  - (A_phi_before[ir,iz]   + A_phi_before[ir,iz+1])*0.5*r[ir]
                                  ) / dr / (r[ir]+0.5*dr)
                            if wq_f[ir,iz,3] == 0:
                                jr = jz_val = 0
                            else:
                                fac = (lamb[wq_f[ir,iz,3]-1]*(r[ir]+0.5*dr)
                                       / np.abs(I_inj_total)/2*(1/flux_mag_max)*elf[ir,iz])
                                jr, jz_val = br*fac, bz*fac
                            jf.append([r[ir]+0.5*dr, z[iz]+0.5*dz, jr, jz_val])
                jf_array = np.array(jf)

                mv_flux = np.zeros((118, 5))
                for k in range(118):
                    if flux[k, 0] > 0:
                        ir_f = nint((flux[k,0]-r_min)/dr)
                        iz_f = nint((flux[k,1]-z_min)/dz)
                        br_fl, bz_fl = self._cal_B(ir_f, iz_f, dr, dz, A_phi, r)
                        mv_flux[k, :] = [flux[k,0], flux[k,1],
                                         A_phi[ir_f,iz_f]*2*pi*flux[k,0], br_fl, bz_fl]

                Bz_cal_puc = np.zeros((len(bz_puc_it), 2))
                for j, jz in enumerate(iz_puc):
                    Bz_cal_puc[j, 0] = z[jz]
                    _, Bz_cal_puc[j, 1] = self._cal_B(ir_puc, jz, dr, dz, A_phi, r)

                if i >= 2:
                    sq_error[i] = error_bz(Bz_cal_puc, Bz_vac_puc, fit_params, weights)

                save_con = (sq_error[i] <= min(sq_error[2:])) and i >= 2
                if save_con:
                    I_close_store.append([int(i), I_tor[-1]])

                    z_Bz_data = np.array([Bz_cal_puc[:,0], Bz_cal_puc[:,1], Bz_vac_puc[:,1]])
                    rm.save_iteration(i, mf_output, jf_array, mv_flux, z_Bz_data)

                    plot_bz_profile(
                        Bz_cal_puc[:,0], Bz_cal_puc[:,1], Bz_vac_puc[:,1],
                        z_puc, bz_puc_it,
                        f"{time_path}/img/Bz-Z_{i}.png",
                        f"No. {i}, sq_error = {sq_error[i]:.1e}")

                    ir_in, iz_in = minimum_position(wq_l)
                    ir_in, iz_in = ir_in+1, iz_in+1
                    m_in   = A_phi[ir_in, iz_in] * 2*pi*r[ir_in]
                    ir_out = nint((ele_lim[0]-r_min)/dr) + 1
                    iz_out = nint((ele_lim[1]-z_min)/dz)
                    m_out  = A_phi[ir_out, iz_out] * 2*pi*ele_lim[0]

                    plot_magnetic_field(
                        f"{time_path}/M_field_{i:03}.csv",
                        f"{time_path}/J_field_{i:03}.csv",
                        f"{path}/electrode.csv",
                        f"{time_path}/img/field_cont_{i:03}.png",
                        f"No. = {i:03}, I_tor = {I_tor[-1]*1e-3:.1f} [kA]",
                        m_in, m_out)

            if plot_counter > 50:
                break
        # == 収束ループ終了 ==

        return {
            "A_phi":         A_phi,
            "m_in":          m_in,
            "I_close_store": I_close_store,
            "sq_error":      sq_error,
        }
