"""C-MAP: CHI Magnetic Analysis Program
QUEST トカマク磁気フラックス面再構成 エントリポイント
"""
import os, time
import numpy as np
from tqdm import tqdm

from cmaplib.utils import cd_main
from cmaplib.grid import Grid
from cmaplib.tokamak_config import TokamakConfig
from cmaplib.greens_function import GreenFunction
from cmaplib.plotting import plot_magnetic_field, plot_bz_profile, plot_psi
from cmaplib.results import ResultsManager
from cmaplib.solver import EquilibriumSolver
from cmaplib.preparation import prepare_time_step
import get_data as g


if __name__ == "__main__":

    time_s = time.time()
    cd_main()

    pi = 3.1415926535
    mu = 4.*pi*1.e-7

    grid   = Grid()
    config = TokamakConfig()

    count = 53034
    path  = f"test30_#{count:.0f}"
    os.makedirs(f"{path}/img",  exist_ok=True)
    os.makedirs(f"{path}/data", exist_ok=True)

    s = g.get_CHI_Data(count, True)
    t_ip,  ip  = s.get_ip()
    t_inj, inj = s.get_inj()

    it0        = np.where(t_inj > 0)[0][0]
    t_inj, inj = t_inj[it0:], inj[it0:]
    i_inj_max  = np.argmax(inj)
    inj_ave    = np.mean(inj[20000:25000])
    i_inj_0    = np.where(inj[i_inj_max:] < inj_ave)[0][0] + i_inj_max
    t_inj_0    = t_inj[i_inj_0]*1e3 + 0.05  # [ms]

    t_g, G_array  = s.get_G()
    t_puc, bz_puc = s.get_bz()
    z_puc = np.array([0 if i == 0 else 687e-3 - ((i-1)*150e-3) for i in range(12)])

    t_decay = t_ip[np.argmax(ip)]*1e3 - 0.4
    t_ana_arr = [18.740]

    wq     = np.loadtxt("modules/ele_posi.csv", delimiter=",").astype(np.int32)
    gf     = GreenFunction("modules/mfile_py.bin")
    solver = EquilibriumSolver(grid, config, wq, gf, pi, mu)

    m_in_data = []

    #. == 時間ループ ==
    for t_ana in tqdm(t_ana_arr, leave=False, desc="time"):
        print(f"\nShot number : {count}\nt_ana = {t_ana:.3f} ms")

        time_path = f"{path}/{t_ana:.3f}"
        os.makedirs(f"{time_path}/img", exist_ok=True)

        p = prepare_time_step(
            t_ana, t_decay, t_inj_0, count, grid, config,
            t_ip, ip, t_g, G_array, t_puc, bz_puc, z_puc,
            path, time_path, pi)

        rm     = ResultsManager(path, t_ana)
        result = solver.solve_time_step(
            t_ana, t_decay, t_inj_0,
            p["A_phi_0"],    p["I_tf_total"],
            p["elf0"],       p["weights"],
            p["iz_puc"],     p["ir_puc"],
            p["fit_params"], p["Bz_vac_puc"],
            p["I_tor_def"],  p["I_inj"], p["I_inA"], p["I_inj_total"],
            p["ikr0"],       p["ikz0"],  p["ikz_dir"],
            p["sn_r"],       p["sn_z"],
            z_puc, p["bz_puc_it"],
            rm, path, time_path)

        m_in, I_close_store, sq_error = result["m_in"], result["I_close_store"], result["sq_error"]

        rm.copy_best(int(np.array(I_close_store)[-1, 0]))

        plot_magnetic_field(
            f"{path}/data/M_field_{t_ana:.3f}.csv",
            f"{path}/data/J_field_{t_ana:.3f}.csv",
            f"{path}/electrode.csv",
            f"{path}/img/field_cont_{t_ana:.3f}.png",
            f"t = {t_ana:.3f} ms", m_in)
        plot_psi(m_in, t_ana, path)

        data_bz = np.loadtxt(f"{path}/data/z_Bz_{t_ana:.3f}.csv", delimiter=",")
        plot_bz_profile(
            data_bz[:,0], data_bz[:,1], data_bz[:,2],
            z_puc, p["bz_puc_it"],
            f"{path}/img/Bz-Z_{t_ana:.3f}.png",
            f"t = {t_ana:.3f}, sq_error = {min(sq_error[2:]):.1e}")

        m_in_data.append([t_ana, m_in])
    #. == 時間ループ終了 ==

    np.savetxt(f"{path}/data/m_in.csv", np.array(m_in_data), fmt="%.4e", delimiter=",")

    total_time = time.time() - time_s
    print(f"\nTotal time : {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
