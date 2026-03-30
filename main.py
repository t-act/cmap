"""C-MAP: CHI Magnetic Analysis Program
QUEST トカマク磁気フラックス面再構成 エントリポイント
"""
import os, time
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy.ndimage import minimum_position

from cmaplib.utils import nint, cd_main
from cmaplib.electromagnetics import cal_vecp_2_grid, compute_B_field, elect_posi_grid
from cmaplib.field_line_tracer import trace_all_field_lines, _trace_single
from cmaplib.greens_function import GreenFunction
from cmaplib.plotting import plot_magnetic_field, plot_bz_profile, plot_psi
from cmaplib.results import ResultsManager
from mfield_sub import cal_sn, get_PF, get_elf, cal_vecp_2, fitting_bz, error_bz, cal_r2
from Parameter import ele_lim, elect0, elect, flux, SMmin, r_c, z_c
import get_data as g


def cal_B(ir, iz, dr, dz, A_phi):
    """A_phi から中心差分で Br, Bz を計算する。"""
    Br = -(A_phi[ir, iz+1] - A_phi[ir, iz-1]) / (2*dz)
    Bz = (1/r[ir]) * (A_phi[ir+1, iz]*r[ir+1] - A_phi[ir-1, iz]*r[ir-1]) / (2*dr)
    return Br, Bz


def cal_flux(ir, iz):
    """ψ = 2π r A_phi を返す。"""
    return 2*np.pi*r[ir]*A_phi[ir, iz]


def update_flux_val(ir, iz, f_max_l, f_min_l, i):
    """フラックス最大/最小値を更新する。"""
    flux_eva = cal_flux(ir, iz)
    if f_max_l[i] > flux_eva:
        f_max_l[i] = flux_eva
    if f_min_l[i] < flux_eva:
        f_min_l[i] = flux_eva


if __name__ == "__main__":

    time_s = time.time()
    cd_main()
    SM = 0

    #. -- 物理定数 --
    pi = 3.1415926535
    mu = 4.*pi*1.e-7

    #. -- グリッド --
    ir_min, ir_max = 0, 100
    dr = 0.02
    r_min, r_max = 0.0, 2.0
    r = np.arange(r_min, r_max+dr, dr)

    iz_min, iz_max = 0, 200
    dz = 0.02
    z_min, z_max = -2.0, 2.0
    z = np.arange(z_min, z_max+dz, dz)

    dl = 0.02

    #. -- 実験データ取得 --
    count = 53034
    path  = f"test30_#{count:.0f}"
    os.makedirs(f"{path}",      exist_ok=True)
    os.makedirs(f"{path}/img",  exist_ok=True)
    os.makedirs(f"{path}/data", exist_ok=True)

    s = g.get_CHI_Data(count, True)

    t_ip,  ip  = s.get_ip()
    t_inj, inj = s.get_inj()

    it0 = np.where(t_inj > 0)[0][0]
    t_inj, inj = t_inj[it0:], inj[it0:]
    i_inj_max  = np.argmax(inj)
    inj_ave    = np.mean(inj[20000:25000])
    inj_re     = inj[i_inj_max:]
    i_inj_0    = np.where(inj_re < inj_ave)[0][0] + i_inj_max
    t_inj_0    = t_inj[i_inj_0]*1e3 + 0.05  # [ms]

    t_g, G_array = s.get_G()
    t_puc, bz_puc = s.get_bz()
    z_puc = np.array([0 if i == 0 else 687e-3 - ((i-1)*150e-3) for i in range(12)])

    it_decay = np.argmax(ip)
    t_decay  = t_ip[it_decay]*1e3 - 0.4

    t_ana_arr = [18.740]

    m_in_data = []
    I_close_before = -50e3

    #. -- 時間ループ外で一度だけロード --
    wq = np.loadtxt("modules/ele_posi.csv", delimiter=",").astype(np.int32)
    gf = GreenFunction("modules/mfile_py.bin")
    A_0 = gf.A_0  # フィラメント電流更新で直接参照

    #. == 時間ループ ==
    for t_ana in tqdm(t_ana_arr, leave=False, desc="time"):
        print(f"\nShot number : {count}\nt_ana = {t_ana:.3f} ms")

        #. TF / PF コイル設定
        PF_coil, TF_coil = get_PF()
        I_tf_total = TF_coil * 16

        I_pf_c = np.zeros(55)
        I_pf_c[[0, 1]]      = PF_coil[[0, 1]] * 41
        I_pf_c[[2, 5]]      = PF_coil[[2, 5]] * 36
        I_pf_c[[3, 4]]      = PF_coil[[3, 4]] * 12
        I_pf_c[[6, 7]]      = PF_coil[[6, 7]] * 41
        I_pf_c[[8, 9, 10]]  = PF_coil[[8, 9, 10]] * 36
        I_pf_c[8] /= 13;  I_pf_c[9] /= 17;  I_pf_c[10] /= 13

        k = 10
        for i in range(8, 11):
            for j in range(1, 9):
                k += 1;  I_pf_c[k] = I_pf_c[i]
                k += 1;  I_pf_c[k] = I_pf_c[i]
                if (i == 9 or i == 11) and j == 6:
                    break

        sn_r0, sn_z0, sn_r, sn_z = cal_sn(elect0, elect, path)

        #. 真空場計算
        _, Bz, _, A_phi = cal_vecp_2_grid(k, r_c, z_c, I_pf_c, r, z)
        A_phi_0 = A_phi.copy()
        mv_field   = A_phi_0 * 2*pi * r[:, np.newaxis]
        R_grid, Z_grid = np.meshgrid(r, z, indexing='ij')
        mv_field_l = np.where(elect_posi_grid(R_grid, Z_grid), 1e29,
                               A_phi * 2*pi * r[:, np.newaxis])

        np.savetxt(f"{path}/data/MV_field.csv",   mv_field.T,   delimiter=",", fmt="%12.4e")
        np.savetxt(f"{path}/data/MV_field_l.csv", mv_field_l.T, delimiter=",", fmt="%12.4e")

        vac_flux = []
        for i in range(118):
            if flux[i, 0] > 0:
                sbr, sbz, _, ssbz = cal_vecp_2(k, r_c, z_c, I_pf_c, flux[i, 0], flux[i, 1])
                vac_flux.append([flux[i,0], flux[i,1], ssbz*2*pi*flux[i,0], sbr, sbz])
        np.savetxt(f"{path}/MV_field_flux_000.csv", np.array(vac_flux),
                   delimiter=",", fmt='%.4e', header="R, Z, A_phi, Br, Bz", comments=" ")

        time_path = f"{path}/{t_ana:.3f}"
        os.makedirs(f"{time_path}",      exist_ok=True)
        os.makedirs(f"{time_path}/img",  exist_ok=True)

        k = 10  # PF コイル本数 (0-10)

        #. ピックアップコイル Bz 読み込み
        it = np.where(np.isclose(t_puc*1e3, t_ana, atol=5e-4))[0][0]
        bz_puc_it = bz_puc[it, :]
        z_puc_fit, bz_puc_fit, weights, fit_params = fitting_bz(z_puc, bz_puc_it)

        iz_puc     = np.array([np.where(np.isclose(z, zp,  atol=1e-2))[0][0] for zp in z_puc])
        iz_puc_fit = np.array([np.where(np.isclose(z, zpf, atol=1e-2))[0][0] for zpf in z_puc_fit])
        ir_puc     = np.where(r == 0.22)[0][0]

        z_re  = np.sort(z_puc)[::-1]
        bz_re = np.concatenate([bz_puc_it[1:6], [bz_puc_it[0]], bz_puc_it[6:]])
        z_re  = np.delete(z_re[1:],  [3, 5])
        bz_re = np.delete(bz_re[1:], [3, 5])
        r2_lin, r2_quad = cal_r2(z_re, bz_re)

        #. フィラメント電流位置
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
            t1, r1, t2, r2_lin_val = 18.9, 0.6, 19.7, 0.3
            kr0 = (r2_lin_val - r1)/(t2 - t1) * t_ana + r1 - (r2_lin_val - r1)/(t2 - t1) * t1
        else:
            kr0 = 0.3
        ikr0 = nint((kr0 - r_min)/dr)
        ik = 0
        ikr, ikz = ikr0, ikz0

        #. elf0 設定
        if t_decay <= t_ana < t_inj_0:
            elf0 = np.linspace(1, 0.3, 8)
        elif t_ana >= t_inj_0:
            a, b, c = np.polyfit([19.2, 19.4, 19.6], [1, 3, 6.5], 2)
            elf0 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])/5 * (a*t_ana**2 + b*t_ana + c)
        else:
            elf0 = np.ones(8)

        plot_counter = 0

        #. 入射電流設定
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

        #. トロイダル電流設定
        it_ip  = np.where(np.isclose(t_ip*1e3, t_ana, atol=5e-4))[0][0]
        I_tor_def = -np.round(ip[it_ip]*1e3, -2)

        #. ログファイル
        with open(f"{time_path}/_log.csv", "w") as f:
            f.write(f"#{count}, t_ana = {t_ana:.3f} ms\n")
            f.write("PF3-1, 3-2, 2, 1, 7, 6, 5-2, 5-1, 4-1, 4-2, 4-3\n")
            f.write(", ".join(str(v) for v in I_pf_c[:11]) + "\n")
            f.write(f"I_tor_def [A], {I_tor_def}\n")
            f.write(f"G [A], {G_round[0]}, {G_round[1]}, {G_round[2]}\n")
            f.write(f"elf0 ,{elf0}\n")
            f.write(f"t_decay [ms],{t_decay:.2f}\n")
            f.write("z [m], " + ", ".join(f"{zf:.2f}" for zf in z_puc_fit) + "\n")
            f.write("bz weight, " + ", ".join(str(w) for w in weights))

        #. ピックアップコイル位置の真空場 Bz
        Bz_vac_puc = np.zeros((len(bz_puc_it), 2))
        for j, jz in enumerate(iz_puc):
            Bz_vac_puc[j, 0] = z[jz]
            _, Bz_vac_puc[j, 1], _, _ = cal_vecp_2(k, r_c, z_c, I_pf_c, r[ir_puc], z[jz])

        #. 収束ループ用変数初期化
        RSM   = 1e29
        rh    = 0.0
        I_tor = np.zeros(20)
        lamb  = np.zeros(20)
        A_phi_close = np.zeros_like(A_phi)
        A_phi_open  = np.zeros_like(A_phi)
        J_inj_p0, J_inj_p, J_inj_p2 = np.zeros(20), np.zeros(20), np.zeros(20)
        J_inj_e,  J_inj_e0           = np.zeros(20), np.zeros(20)
        flux_mag_r = np.zeros((ir_max, iz_max))
        flux_mag_z = np.zeros((ir_max, iz_max))

        i_max       = 1000 if t_ana < t_decay else 100
        I_close_store = []
        sq_error      = np.ones(i_max)
        rm            = ResultsManager(path, t_ana)

        #. == 収束ループ ==
        for i in tqdm(range(i_max), leave=False, desc="i"):
            I_tor_def_new = -150*1e3 if (i == 0 and t_ana >= t_inj_0) else deepcopy(I_tor_def)

            f_max_l = np.zeros(21)
            f_min_l = np.zeros(21)
            elf     = np.ones((ir_max, iz_max))

        #. -- 磁力線追跡 --
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

        #. -- λ 計算 --
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
                        Br, Bz = cal_B(ir_mid, iz_mid, dr, dz, A_phi)
                        j_val  = (Br*sn_r[i_ele] + Bz*sn_z[i_ele])*elect[i_ele,2]*elf[ir,iz]/k0
                        if wq_f[ir_mid, iz_mid, 0] != 0:
                            J_inj_p0[i_ele] += j_val
                        if wq_f[ir_mid, iz_mid, 3] != 0:
                            J_inj_p[i_ele]  += j_val
                        J_inj_p2[i_ele] += j_val
                        update_flux_val(ir_mid, iz_mid, f_max_l, f_min_l, i_ele)
                        update_flux_val(ir_ele, iz_ele, f_max_l, f_min_l, i_ele)

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
                I_tor[i_ele]   *= lamb[i_ele]
                J_inj_p2[i_ele] *= lamb[i_ele]

            lamb[-1] = sum(lamb[2:18])/sum(elect[2:18, 2])*elect[19, 2]

        #. -- トロイダル電流 + A_phi 更新 --
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

        #. -- 保存 / プロット --
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
                                jr = jz = 0
                            else:
                                fac = lamb[wq_f[ir,iz,3]-1]*(r[ir]+0.5*dr)/np.abs(I_inj_total)/2*(1/flux_mag_max)*elf[ir,iz]
                                jr, jz = br*fac, bz*fac
                            jf.append([r[ir]+0.5*dr, z[iz]+0.5*dz, jr, jz])
                jf_array = np.array(jf)

                mv_flux = np.zeros((118, 5))
                for k in range(118):
                    if flux[k, 0] > 0:
                        ir_f, iz_f = nint((flux[k,0]-r_min)/dr), nint((flux[k,1]-z_min)/dz)
                        br_fl, bz_fl = cal_B(ir_f, iz_f, dr, dz, A_phi)
                        mv_flux[k, :] = [flux[k,0], flux[k,1],
                                         A_phi[ir_f,iz_f]*2*pi*flux[k,0], br_fl, bz_fl]

                Bz_cal_puc = np.zeros((len(bz_puc_it), 2))
                for j, jz in enumerate(iz_puc):
                    Bz_cal_puc[j, 0] = z[jz]
                    _, Bz_cal_puc[j, 1] = cal_B(ir_puc, jz, dr, dz, A_phi)

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
                    m_in  = A_phi[ir_in, iz_in] * 2*pi*r[ir_in]
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
        #. == 収束ループ終了 ==

        i_best = int(np.array(I_close_store)[-1, 0])
        rm.copy_best(i_best)

        plot_magnetic_field(
            f"{path}/data/M_field_{t_ana:.3f}.csv",
            f"{path}/data/J_field_{t_ana:.3f}.csv",
            f"{path}/electrode.csv",
            f"{path}/img/field_cont_{t_ana:.3f}.png",
            f"t = {t_ana:.3f} ms",
            m_in)
        plot_psi(m_in, t_ana, path)

        data_bz = np.loadtxt(f"{path}/data/z_Bz_{t_ana:.3f}.csv", delimiter=",")
        plot_bz_profile(
            data_bz[:,0], data_bz[:,1], data_bz[:,2],
            z_puc, bz_puc_it,
            f"{path}/img/Bz-Z_{t_ana:.3f}.png",
            f"t = {t_ana:.3f}, sq_error = {min(sq_error[2:]):.1e}")

        m_in_data.append([t_ana, m_in])
    #. == 時間ループ終了 ==

    np.savetxt(f"{path}/data/m_in.csv", np.array(m_in_data), fmt="%.4e", delimiter=",")

    total_time = time.time() - time_s
    print(f"\nTotal time : {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
