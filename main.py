"""C-MAP: CHI Magnetic Analysis Program
QUEST トカマク磁気フラックス面再構成 エントリポイント
"""
import os, time
import numpy as np
from tqdm import tqdm

from cmaplib.utils import nint, cd_main
from cmaplib.electromagnetics import cal_vecp_2_grid, elect_posi_grid
from cmaplib.greens_function import GreenFunction
from cmaplib.plotting import plot_magnetic_field, plot_bz_profile, plot_psi
from cmaplib.results import ResultsManager
from cmaplib.solver import EquilibriumSolver
from mfield_sub import cal_sn, get_PF, cal_vecp_2, fitting_bz, cal_r2
from Parameter import ele_lim, elect0, elect, flux, r_c, z_c
import get_data as g


if __name__ == "__main__":

    time_s = time.time()
    cd_main()

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

    solver = EquilibriumSolver(r, z, dr, dz, pi, mu, dl, elect, flux, ele_lim, wq, gf)

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

        #. elf0 設定
        if t_decay <= t_ana < t_inj_0:
            elf0 = np.linspace(1, 0.3, 8)
        elif t_ana >= t_inj_0:
            a, b, c = np.polyfit([19.2, 19.4, 19.6], [1, 3, 6.5], 2)
            elf0 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])/5 * (a*t_ana**2 + b*t_ana + c)
        else:
            elf0 = np.ones(8)

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

        #. == 収束ループ ==
        rm     = ResultsManager(path, t_ana)
        result = solver.solve_time_step(
            t_ana, t_decay, t_inj_0, A_phi_0, I_tf_total,
            elf0, weights, iz_puc, ir_puc, fit_params, Bz_vac_puc,
            I_tor_def, I_inj, I_inA, I_inj_total,
            ikr0, ikz0, ikz_dir,
            sn_r, sn_z, z_puc, bz_puc_it,
            rm, path, time_path)

        m_in          = result["m_in"]
        I_close_store = result["I_close_store"]
        sq_error      = result["sq_error"]

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
