import os, time, numpy as np
from icecream import ic
from tqdm import tqdm
from copy import deepcopy
from scipy.ndimage import minimum_position
from cmap.utils import nint, cd_main, copy_file
from cmap.electromagnetics import cal_vecp_2_grid, compute_B_field, elect_posi_grid
from cmap.field_line_tracer import trace_all_field_lines, _trace_single
from cmap.greens_function import GreenFunction
from mfield_sub import (cal_sn,
                            get_PF, elect_posi, get_elf,
                            cal_vecp_2, fitting_bz, error_bz, cal_r2,
                            plot_field, plot_field2, plot_z_Bz, plot_psi, plot_z_Bz2
                        )
import get_data as g    # Get CHI data from QUEST server

#. Parameters (Parameters.py)
from Parameter import ele_lim, elect0, elect, flux, SMmin, r_c, z_c





#. --
def track_mag_line(ir, iz, r, z, i_dir, rr10,
                    flux_mag_r, flux_mag_z, A_phi, wq0, wq):
    """
    Tracked r and z indices along the magnetic field line.

    Parameters
    ---
    ir      (int)   : R index 0-100
    iz      (int)   : Z index 0-200
    i_dir   (int)   : Direction index 1 or 2
    rr10    (float) : Reference value for A_phi evaluation

    Returns
    ---
    wq_f    (np.array) : Index of where the magnetic field lines end up
    """

    # 入力パラメータに基づいて符号を設定
    if i_dir == 1:
        sign = 1
    elif i_dir == 2:
        sign = -1
    else:
        raise ValueError("i_dir must be 1 or 2")

    # 開始位置を設定
    jr, jz = ir, iz

    for _ in range(4000):
        wq0[jr, jz] = 20

        # 磁場方向の決定
        ir_ad = int(np.sign(flux_mag_r[jr, jz] * sign))
        iz_ad = int(np.sign(flux_mag_z[jr, jz] * sign))

        # 移動方向の候補
        directions = [
            (0, iz_ad),      # Z方向
            (ir_ad, 0),      # R方向
            (ir_ad, iz_ad),  # 対角方向
        ]

        i_ad = -1
        A_phi_eva = 1e29

        # 候補方向を評価
        for j, (ir_offset, iz_offset) in enumerate(directions):
            ir_eva = jr + ir_offset
            iz_eva = jz + iz_offset

            # 配列の境界チェック
            if not (0 <= ir_eva < A_phi.shape[0]-1 and 0 <= iz_eva < A_phi.shape[1]-1):
                continue

            # A_phi の評価
            A_phi_eva_new = abs(
                (A_phi[ir_eva, iz_eva] + A_phi[ir_eva, iz_eva+1]) * r[ir_eva]
                + (A_phi[ir_eva+1, iz_eva] + A_phi[ir_eva+1, iz_eva+1]) * r[ir_eva+1]
                - rr10
            )

            # 最適な方向を更新
            if A_phi_eva > A_phi_eva_new:
                A_phi_eva = A_phi_eva_new
                i_ad = j

        # 移動可能な方向がない場合は終了
        if i_ad == -1:
            break

        ir_offset, iz_offset = directions[i_ad]

        # 容器外に出た場合
        if wq[jr + ir_offset, jz + iz_offset] == -1:
            wq_f[ir, iz, i_dir] = wq[jr, jz]
            break

        # 閉磁気面に到達した場合
        if wq0[jr + ir_offset, jz + iz_offset] == 20:
            wq_f[ir, iz, i_dir] = 20
            break

        # 次の位置に移動
        jr += ir_offset
        jz += iz_offset

    return wq_f[ir, iz, i_dir]


def cal_B(ir, iz, dr, dz, A_phi):
    """A_phi -> Br, Bz
       B = rotA
    """
    Br = -(A_phi[ir, iz+1] - A_phi[ir, iz-1]) / (2*dz)
    Bz = (1/r[ir]) * (A_phi[ir+1, iz]*r[ir+1] - A_phi[ir-1, iz]*r[ir-1]) / (2*dr)
    return Br, Bz


def cal_flux(ir, iz):
    """Calculate flux from A_phi
    psi = \int_s B ds
        = \int_s (nabla X A) ds
        = \int_c A dl
        = 2pi r A_phi
    """
    return 2*np.pi*r[ir]*A_phi[ir, iz]


def update_flux_val(ir, iz, f_max_l, f_min_l, i):
    """Update the maximum and minimum flux values."""    
    #. flux value
    flux_eva = cal_flux(ir, iz)
    
    #. max
    if f_max_l[i] > flux_eva:
        f_max_l[i] = flux_eva
    
    #. min
    if f_min_l[i] < flux_eva:
        f_min_l[i] = flux_eva
#. --










#. -- main --
if __name__=="__main__":
    """Draw CHI plasma magnetic line and current vector"""



    #.  -- Parameters --
    total_time = 0  # 時間計測
    time_s = time.time()

    cd_main()   # Change directory to main directory
    SM = 0


    #. -- Physical value --
    pi = 3.1415926535
    mu = 4.*pi*1.e-7


    #. -- Coordinate --
    ir_min, ir_max = 0, 100
    dr = 0.02
    r_min, r_max = 0.0, 2.0
    r = np.arange(r_min, r_max+dr, dr)

    iz_min, iz_max = 0, 200
    dz = 0.02
    z_min, z_max = -2.0, 2.0
    z = np.arange(z_min, z_max+dz, dz)

    dl = 0.02



    #. -- get data from QUEST server --
    count = 53034   # Shotnumber to be analyzed

    path = f"test30_#{count:.0f}"   # Folder name
    os.makedirs(f"{path}", exist_ok=True)
    os.makedirs(f"{path}/img", exist_ok=True)
    os.makedirs(f"{path}/data", exist_ok=True)

    # Initialize
    s = g.get_CHI_Data(count, True)

    # Plasma current
    t_ip, ip = s.get_ip()


    # Injection current
    t_inj, inj = s.get_inj()

    # I_inj time setting
    it0 = np.where(t_inj > 0)[0][0]
    t_inj, inj = t_inj[it0:], inj[it0:]
    i_inj_max = np.argmax(inj)
    inj_ave = np.mean(inj[20000:25000])
    inj_re = inj[i_inj_max:]
    i_inj_0 = np.where(inj_re < inj_ave)[0][0] + i_inj_max
    t_inj_0 = t_inj[i_inj_0]*1e3 + 0.05  #[ms]


    # wall-Injection current
    t_g, G_array = s.get_G()


    # Bz (pick up coil)
    t_puc, bz_puc = s.get_bz()

    # Height of Bz coil
    z_puc = np.array([0 if i == 0 else 687e-3 - ((i-1) * 150e-3) for i in range(12)])



    # === Analysis Time setting === 
    it_decay = np.argmax(ip)
    t_decay = t_ip[it_decay]*1e3-0.4

    #t_ana_arr = [19.04, 19.05, 19.06, 19.07]
    #t_ana_arr = np.arange(19.2, 19.6, 0.1)
    #t_ana_arr = np.arange(19.2, 19.7, 0.005)
    #t_ana_arr = np.arange(18.85, 19.7, 0.05)
    #t_ana_arr = [19.4, 19.5, 19.6]
    t_ana_arr = [18.740]

    # ===



    m_in_data = []
    I_close_before = -50e3

    #. -- 時間ループ外でロード (毎 time step で再ロードしない) --
    #. wq -> 容器壁判定グリッド (Numba JIT に渡すため int32)
    wq = np.loadtxt("modules/ele_posi.csv", delimiter=",").astype(np.int32)

    #. GreenFunction: A_0 (101×201×101×201 ≈ 3.4 GB) を一度だけロード
    gf = GreenFunction("modules/mfile_py.bin")
    A_0 = gf.A_0  # フィラメント電流更新で直接参照



#. == Time loop start ==
    for t_ana in tqdm(t_ana_arr, leave = False, desc = "time"):
        print(f"\nShot number : {count}\nt_ana = {t_ana:.3f} ms")
        #. -- TF, PF coil setting --
        #. TF coil
        PF_coil, TF_coil = get_PF()
        I_tf_total = TF_coil*16        

        #. PF coil
        # Initialize arrays for coil positions
        I_pf_c = np.zeros(55)
        I_pf_c[[0, 1]] = PF_coil[[0, 1]] * 41           # PF3-1, 2
        I_pf_c[[2, 5]] = PF_coil[[2, 5]] * 36           # PF2, 6
        I_pf_c[[3, 4]] = PF_coil[[3, 4]] * 12           # PF1, 7
        I_pf_c[[6, 7]] = PF_coil[[6, 7]] * 41           # PF5-1, 2
        I_pf_c[[8, 9, 10]] = PF_coil[[8, 9, 10]] * 36   # PF4-1, 2, 3
        
        #PF4-1, 2, 3 は分割して計算
        I_pf_c[8] /= 13
        I_pf_c[9] /= 17
        I_pf_c[10] /= 13

        #PF4-1,2,3 = 0のため意味なさそう
        k = 10

        for i in range(8, 11):
            for j in range(1, 9):
                k += 1
                I_pf_c[k] = I_pf_c[i]

                k += 1
                I_pf_c[k] = I_pf_c[i]

                if (i == 9 or i == 11) and j == 6:
                    break



        #. -- Calcilate electode -- 
        sn_r0, sn_z0, sn_r, sn_z = cal_sn(elect0, elect, path)
            

        #. wq, A_0 はループ外でロード済み (時間ループ前を参照)
        

        #. Start calcurating
        #. Calculate the vector potential of the vacuum field for each grid
        #  (ベクトル化版: 二重ループを cal_vecp_2_grid に置換)
        _, Bz, _, A_phi = cal_vecp_2_grid(k, r_c, z_c, I_pf_c, r, z)
        A_phi_0 = A_phi.copy()
        mv_field = A_phi_0 * 2 * pi * r[:, np.newaxis]

        R_grid, Z_grid = np.meshgrid(r, z, indexing='ij')
        electrode_mask = elect_posi_grid(R_grid, Z_grid)
        mv_field_l = np.where(electrode_mask, 1e29, A_phi * 2 * pi * r[:, np.newaxis])
        gmin_0 = np.min(mv_field)
        gmax_0 = np.max(mv_field)

        n = 0
        np.savetxt(f"{path}/data/MV_field.csv", mv_field.T, delimiter=",", fmt="%12.4e")
        np.savetxt(f"{path}/data//MV_field_l.csv", mv_field_l.T, delimiter=",", fmt="%12.4e")
        
        # Calculate vacuum field on flux loop positions
        vac_flux = []
        for i in range(118):
            if flux[i, 0] > 0:
                sbr, sbz, _, ssbz = cal_vecp_2(k, r_c, z_c, I_pf_c, flux[i, 0], flux[i, 1])
                vac_flux.append([
                                flux[i,0], flux[i,1], 
                                ssbz * 2*pi*flux[i,0], sbr, sbz
                                ])
        vac_flux_np = np.array(vac_flux)
        np.savetxt(f"{path}/MV_field_flux_{n:03}.csv", vac_flux_np, delimiter = ",", fmt = '%.4e',
                    header = "R, Z, A_phi, Br, Bz", comments=" ")


        # Path setting
        time_path = f"{path}/{t_ana:.3f}"
        os.makedirs(f"{time_path}", exist_ok = True)
        os.makedirs(f"{time_path}/img", exist_ok = True)


        # Coil number
        k = 10  #Number of PF coil -> 11 (0-10)



    #. -- Read Bz from Pick up coil on QUEST --
        # get index of t_ana
        it = np.where(np.isclose(t_puc*1e3, t_ana, atol = 5e-4))[0][0]

        # rearange Bz(t_ana)
        bz_puc_it = bz_puc[it,:]

        # Fitting 
        z_puc_fit, bz_puc_fit, weights, fit_params = fitting_bz(z_puc, bz_puc_it)

        # Pick up coil position ref. TAKEDA, master thesis
        # z = 0.687 ~ -0.813 m
        iz_puc = np.array([np.where(np.isclose(z, zp, atol=1e-2))[0][0] for zp in z_puc])
        # z = 0.537 ~ -0.813 m 　(一番上のコイル信号を除外)
        iz_puc_fit = np.array([np.where(np.isclose(z, zpf, atol=1e-2))[0][0] for zpf in z_puc_fit])
        # r ~ 0.215 m (mesh の関係で r = 0.22 m)
        ir_puc = np.where(r == 0.22)[0][0]

        z_re = np.sort(z_puc)[::-1]
        bz_re = np.concatenate([bz_puc_it[1:6], [bz_puc_it[0]], bz_puc_it[6:]])
        z_re = z_re[1:]
        bz_re = bz_re[1:]

        z_re = np.delete(z_re, [3, 5])
        bz_re = np.delete(bz_re, [3, 5])
        r2_lin, r2_quad = cal_r2(z_re, bz_re)



    #. -- Filament current popsition
        # Z position
        if t_decay <= t_ana < t_inj_0:
            ikz0 = iz_puc_fit
            # マイナス方向にずらしていく
            ikz_dir = 1    #or +1

        
        elif t_ana >= t_inj_0:    #Bz 分布が2次関数のFitting の方が支配的        
            # Bz の最小値を探索 （Fitting した9点の中から）
            ibz_re_min = np.argmin(bz_re)
            z_re_min = z_re[ibz_re_min]
            iz_puc_min = np.abs(z - z_re_min).argmin()
            z_puc_min = z[iz_puc_min]
            ikz0 = iz_puc_min
            # マイナス方向にずらしていく
            ikz_dir = 1    #or +1

        else:
            kz0 = -1.5
            ikz0 = nint((kz0 - z_min)/dz)
        
            # マイナス方向にずらしていく
            ikz_dir = 1    #or +1


        # R position
        if t_ana >= t_decay:
            t1, r1 = 18.9, 0.6
            t2, r2 = 19.7, 0.3

            a = (r2-r1) / (t2-t1)
            b = r1 -a*t1

            kr0 = a*t_ana + b

        else:
            kr0 = 0.3

        ikr0 = nint((kr0 - r_min)/dr)        
        ik = 0
        ikr, ikz = ikr0, ikz0



    #. -- Setting elf0 --
        # 2024.10.30 Update by MOTOKI
        # なんとなくで合わせています．今後はここを要検討

        # Decay phase
        if t_decay <= t_ana < t_inj_0:
            elf0 = np.linspace(1, 0.3, 8)

        elif t_ana >= t_inj_0:
            # 3点を通る曲線を2次関数でFitting
            t_points = np.array([19.2, 19.4, 19.6])
            elf0_points = np.array([1, 3, 6.5])
            coeff = np.polyfit(t_points, elf0_points, 2)
            a, b, c = coeff
            y = a*t_ana**2 + b*t_ana + c
            elf0 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])/5 * y


        # Ramp up phase
        else:
            elf0 = np.ones(8)
            #elf0 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])/5 * y
            #elf0 = 1.   # 容器壁間を繋ぐ電流 = 1


        plot_counter = 0    #画像をplot しすぎないように設定




    #. Setup coil, Injection
        it_g = np.where(np.isclose(t_g*1e3, t_ana, atol = 1e-4))[0][0]
        G = np.array([G_array[it_g,0], np.sum(G_array[it_g,1:4]), G_array[it_g,4]])
        #G = np.array([np.sum(G_array[it_g,0:3]), G_array[it_g,3], G_array[it_g,4]])
        G[-100 < G] = 0
        G_round = np.round(G, -2)
        if t_ana >= t_inj_0 and G_round[1] == 0:
            G_round[1] = -100

        #. Injection Area
        I_inA = np.zeros(21)
        I_inA[2:4] = 1      #Fortran 3, 4 -> Python 2, 3
        I_inA[4:8] = 2      #Fortran 5, 6, 7, 8 -> Python 4, 5, 6, 7
        I_inA[8:10] = 3     #Fortran 9, 10 -> Python 8, 9
        
        I_inj_total = np.sum(G)
        I_inj = np.zeros(21)
        I_inj[1:4] = G_round[0:3]
        


    #. Setup Toroidal current
        it_ip = np.where(np.isclose(t_ip*1e3, t_ana, atol = 5e-4))[0][0]
        I_tor_def = -np.round(ip[it_ip]*1e3, -2)   #[kA]



    #. log file
        with open(f"{time_path}/_log.csv", "w") as f:
            f.write(f"#{count}, t_ana = {t_ana:.3f} ms\n")
            f.write(f"PF3-1, 3-2, 2, 1, 7, 6, 5-2, 5-1, 4-1, 4-2, 4-3\n")
            for i,val in enumerate(I_pf_c[:11]):
                if i == len(I_pf_c[:11])-1:
                    f.write(f"{val}\n")
                else:
                    f.write(f"{val}, ")
            f.write(f"I_tor_def [A], {I_tor_def}\n")
            f.write(f"G [A], {G_round[0]}, {G_round[1]}, {G_round[2]}\n")
            f.write(f"elf0 ,{elf0}\n")
            f.write(f"t_decay [ms],{t_decay:.2f}\n")
            f.write(f"z [m], ")
            for zf in z_puc_fit:
                f.write(f"{zf:.2f}, ")
            f.write(f"\nbz weight, ")
            for w in weights:
                f.write(f"{w}, ")



        # Cal vacuum field on Pick up coil position
        Bz_vac_puc = np.zeros((len(bz_puc_it),2))
        for j,jz in enumerate(iz_puc):
            Bz_vac_puc[j,0] = z[jz]
            _, Bz_vac_puc[j,1], _, _ = cal_vecp_2(k, r_c, z_c, I_pf_c, r[ir_puc], z[jz])
        
        
        #. Setup parameter
        RSM = 1e29
        fm = np.zeros((20, 2))

        flux_mag_r, flux_mag_z = np.zeros((ir_max, iz_max)), np.zeros((ir_max, iz_max))
        rr = np.zeros(10)
        rr_kai = np.zeros((ir_max, iz_max))

        wr, wq0 = np.zeros(200), np.zeros(200)

        rh = 0.0

        kr, kz = np.zeros(4000), np.zeros(4000)
        kr_f, kz_f = np.zeros((4000, 3)), np.zeros((4000, 3))
        I_tor = np.zeros(20)

        J_inj_p0, J_inj_p, J_inj_p2 = np.zeros(20), np.zeros(20), np.zeros(20)
        J_inj_e, J_inj_e0 = np.zeros(20), np.zeros(20)

        lamb = np.zeros(20)

        A_phi_close, A_phi_open = np.zeros_like(A_phi), np.zeros_like(A_phi)



        

    #. == Calculating loop ==

        if t_ana < t_decay:
            i_max = 1000    #Ramp up phase は回数多め
        else:
            i_max = 100

        I_close_store = []

        sq_error = np.ones(i_max)
        Lcfs = []       #Last closed-flux surface


        for i in tqdm(range(i_max), leave = False, desc = "i"):
            if i == 0 and t_ana >= t_inj_0: 
                I_tor_def_new = -150*1e3

            else:
                I_tor_def_new = deepcopy(I_tor_def)
        
            #. Initialise variables
            Jinj = 0
            f_max_l = np.zeros(21)
            f_min_l = np.zeros(21)

            Meg = 0
            elf = np.ones((ir_max,iz_max))
            wr[:] = 1e4


        #. -- Track mag. line --
            #. wq_f -> 磁力線の交点情報
                #last index 
                #   保存用             -> 0
                #   開磁気面（前進方向） -> 1 (i_dir)
                #   開磁気面（後進方向） -> 2 (i_dir)
                #   保存用             -> 3
            wq_f = np.zeros((ir_max, iz_max, 4), dtype = int)
            I_tor[:] = 0

            #. Setup magnitude of flux and wq
            #  (ベクトル化版: 二重ループを compute_B_field に置換)
            flux_mag_r, flux_mag_z = compute_B_field(A_phi, r, dr, dz, ir_max, iz_max)

            
            #. Track mag. line main
            #  (特殊ケース: 電極位置から単一磁力線追跡)
            if rh > 1e29:   #rh = I_tor-open - I_tor-close
                ir = nint((elect[0, 0] - r_min)/dr) -1
                iz = nint((elect[0, 1] - z_min)/dz)

                if wq[ir, iz] != -1:    #真空容器内部 or 境界だったら
                    wq0_single = wq.copy()
                    rr10 = ((A_phi[ir, iz] + A_phi[ir, iz+1])*r[ir]
                            + (A_phi[ir+1, iz] + A_phi[ir+1, iz+1])*r[ir+1])
                    wq_f[ir, iz, 1] = _trace_single(
                        ir, iz, 1, rr10, flux_mag_r, flux_mag_z,
                        A_phi, wq0_single, wq, r, ir_max+1, iz_max+1)
                    wq_f[ir, iz, 2] = _trace_single(
                        ir, iz, 2, rr10, flux_mag_r, flux_mag_z,
                        A_phi, wq0_single, wq, r, ir_max+1, iz_max+1)


            #. 全グリッド点の磁力線を Numba prange で並列追跡
            wq_f_traced = trace_all_field_lines(A_phi, flux_mag_r, flux_mag_z, r, wq)
            wq_f[:, :, 1] = wq_f_traced[:, :, 1]
            wq_f[:, :, 2] = wq_f_traced[:, :, 2]

            #. 後処理: wq_f[:,:,0], wq_f[:,:,3], I_tor, elf の設定
            for iz in range(iz_max):
                for ir in range(ir_max):

                    if wq[ir, iz] != -1:

                        r_mid = r[ir] + 0.5*dr  #cell の中心

                        #閉磁気面内
                        closed_con = (wq_f[ir,iz,1] == 20 and wq_f[ir,iz,2] == 20)  #wq_f = 20 -> 閉磁気面
                        if closed_con:
                            wq_f[ir, iz, 0] = 20

                        #開磁気面（閉磁気面以外）
                        else:
                            #. 磁力線の一方が特定のCHI電極に交差している場合
                            if ((wq_f[ir, iz, 2] == 1 or wq_f[ir, iz, 2] == 2) and
                                wq_f[ir, iz, 1] != 1 and
                                wq_f[ir, iz, 1] != 2 and
                                wq_f[ir, iz, 1] != 20 and
                                elect[wq_f[ir, iz, 1]-1, 2] != 0
                                ):

                                wq_f[ir, iz, 0] = wq_f[ir, iz, 1]

                                I_tor[wq_f[ir, iz, 0]-1] += \
                                    mu*I_tf_total/(2*pi*r_mid)*elect[wq_f[ir, iz, 0]-1, 2]*dr*dz

                            #. 磁力線の他方の端が特定の電極に交差している場合
                            elif ((wq_f[ir, iz, 1] == 1 or wq_f[ir, iz, 1] == 2) and
                                    wq_f[ir, iz, 2] != 1 and
                                    wq_f[ir, iz, 2] != 2 and
                                    wq_f[ir, iz, 2] != 20 and
                                    elect[wq_f[ir, iz, 2]-1, 2] != 0
                                    ):

                                wq_f[ir, iz, 0] = wq_f[ir, iz, 2]

                                I_tor[wq_f[ir, iz, 0]-1] += \
                                    mu*I_tf_total/(2*pi*r_mid)*elect[wq_f[ir, iz, 0]-1, 2]*dr*dz

                            #. 電極の間に位置する場合
                            elif (np.all(elect[wq_f[ir, iz, 1:3]-1, 2] > 0.1)
                                    and np.all(wq_f[ir, iz, 1:3] != 20)
                                    ):

                                if (3 <= wq_f[ir,iz,1] <= 10
                                    and 3 <= wq_f[ir,iz,2] <= 10
                                    ):
                                    wq_f[ir,iz,3] = wq_f[ir,iz,1]
                                    elf[ir,iz] = get_elf(min(wq_f[ir,iz,1], wq_f[ir,iz,2]), elf0)

            wq_f[:, :, 3] += wq_f[:, :, 0]
        #. --   end track mag. line



        #. -- Calculate Lambda --
            J_inj_p0[:], J_inj_p[:], J_inj_p2[:] = 0, 0, 0 
            J_inj_e0[:], J_inj_e[:] = 0, 0

            f_max, f_min = -1.e29, 1.e29

            
            #. 分割した電極本数(20-1) +1して回すので
            for i_ele in range(19): 
                #. 電極の幅とセグメント計算
                w_ele = elect[i_ele+1, 0] - elect[i_ele, 0]
                h_ele = elect[i_ele+1, 1] - elect[i_ele, 1]
                k0 = nint(np.sqrt(w_ele**2 + h_ele**2)/dl)
                
                
                if (elect[i_ele, 0] > 0.1 and 
                    elect[i_ele+1, 0] > 0.1 and
                    k0 != 0
                    ):

                    #. 電極の幅 (R)
                    w_ele = (elect[i_ele+1, 0] - elect[i_ele, 0])/k0
                    #. 電極の幅 (Z)
                    h_ele = (elect[i_ele+1, 1] - elect[i_ele, 1])/k0

                    f_min_l[i_ele], f_max_l[i_ele] = -1.e29, 1.e29
                    
                    #. 分割したセグメントの合計
                    for k in range(int(k0)):

                        ir_mid = nint((elect[i_ele, 0] + w_ele*(0.5 + k) - r_min)/dr)
                        iz_mid = nint((elect[i_ele, 1] + h_ele*(0.5 + k) - z_min)/dz)                    
                        
                        ir_ele = nint((elect[i_ele, 0] + w_ele*k - r_min)/dr)
                        iz_ele = nint((elect[i_ele, 1] + h_ele*k - z_min)/dz)
                        
                        Br, Bz = cal_B(ir_mid, iz_mid, dr, dz, A_phi)

                        if wq_f[ir_mid, iz_mid, 0] != 0:
                            # j=∇×B
                            J_inj_p0[i_ele] += (Br*sn_r[i_ele] + Bz*sn_z[i_ele])*elect[i_ele,2]*elf[ir,iz]/k0

                        if wq_f[ir_mid, iz_mid, 3] != 0:
                            J_inj_p[i_ele] += (Br*sn_r[i_ele] + Bz*sn_z[i_ele])*elect[i_ele,2]*elf[ir,iz]/k0

                        # 2024.10.30 Update by MOTOKI
                        J_inj_p2[i_ele] += (Br*sn_r[i_ele] + Bz*sn_z[i_ele])*elect[i_ele,2]*elf[ir,iz]/k0
                        
                        #. Flux value の記録(max and min)
                        #. セルの中心
                        update_flux_val(ir_mid, iz_mid, f_max_l, f_min_l, i_ele)

                        #. セル毎
                        update_flux_val(ir_ele, iz_ele, f_max_l, f_min_l, i_ele)

                    #. -- end for k range(k0)

                    fm = np.array([f_min_l[i_ele], f_max_l[i_ele]])

                    if np.min(fm[0:2]) < f_min:
                        f_min = np.min(fm[0:2])
                    
                    if np.max(fm[0:2]) > f_max:
                        f_max = np.max(fm[0:2])


                #. -- end if elect[i_ele, 0] > 0.1 and elect[i_ele+1, 0] > 0.1
            #. -- end for i_ele 0 - 19


            lamb[:] = 0.
            emh = True

            for i_ele in range(19):
                k1 = int(I_inA[i_ele])
                
                J_inj_e[k1] += J_inj_p[i_ele]
                J_inj_e0[k1] += J_inj_p0[i_ele]

            for i_ele in range(19):
                k1 = int(I_inA[i_ele])
                
                if (abs(J_inj_e[k1]/sum(J_inj_e[:])) > 0.01 
                    and k1 != 0
                    ):

                    if J_inj_e[k1]/J_inj_e0[k1] > 0:
                        lamb[i_ele] = elect[i_ele, 2]*I_inj[k1]/J_inj_e[k1]
                    else:
                        emh = False
                
                elif (abs(J_inj_e[k1]/sum(J_inj_e[:])) <= 0.01
                        and k1 != 0
                        and abs(I_inj[k1]) > 0.01
                    ):
                    emh = False
                
                I_tor[i_ele] *= lamb[i_ele]
                J_inj_p2[i_ele] *= lamb[i_ele]

            lamb[-1] = sum(lamb[2:18])/sum(elect[2:18, 2])*elect[19, 2]

            #. note
                #lamb, J_inj_p2, I_tor はfortran に比べて要素が一個少ない
        #. --

        #. -- Caluculate Toroidal current from Lambda value --
            #  (ベクトル化版: 二重ループを GreenFunction.accumulate_fields に置換)
            A_phi_before = deepcopy(A_phi[:,:])
            I_tor[:] = 0

            A_phi_close, A_phi_open, _I_tor_close, _I_tor_open_delta, wq_l = \
                gf.accumulate_fields(wq_f, elf, lamb, r, dr, dz, mu, I_tf_total, pi)
            I_tor[:20] = _I_tor_open_delta   # indices 0-18: 開磁気面 (λ適用済み), 19=0
            I_tor[-1]  = _I_tor_close        # 閉磁気面合計 (λ[-1] はこの後適用)

            Itor_cal_inp = np.abs(I_tor_def_new) - np.abs(np.sum(I_tor[0:20]))
            rh = 1.e30

            
            if t_ana >= t_decay:
                # Itor が減少していく段階では λ_close > λ_open 
                lambda_fac = 2/mu

            else:
                lambda_fac = np.max(np.abs(lamb[0:20]))


            if Itor_cal_inp > 0:
                if ((2*np.abs(I_tor[-1]) * lambda_fac >  Itor_cal_inp)
                    and (np.abs(I_tor[-1]) * lambda_fac > Itor_cal_inp or SM) > 1e-8
                    ):
                    
                    lamb[-1] = (I_tor_def_new - np.sum(I_tor[0:20])) / I_tor[-1]
                    A_phi[:,:] = deepcopy(A_phi_0[:,:])

                else:
                    lamb[-1] = 0
                    ik += 1

                    if ik == 1:
                        ikr = ikr0
                        ikz = ikz0
                        
                    elif ik == 2:
                        ikr = ikr0 + 1
                        ikz = ikz0

                    elif ik == 3:
                        ikr = ikr0
                        ikz = ikz0 + ikz_dir

                    else:
                        if SMmin[1] < SMmin[2]:
                            ikr0 += 1
                        else:
                            ikz0 += ikz_dir
                        
                        ik = 2
                        ikr = ikr0 + 1
                        ikz = ikz0
                        SMmin[:] = 1.e30
                    
                    # フィラメント電流位置がgrid を超えたら終了
                    if ikr > ir_max-1 or np.any(ikz > iz_max-1):
                        break
                    
                    # 2024.11.04 Update by MOTOKI
                    if t_decay <= t_ana < t_inj_0:
                        # z_puc の位置に9点 Z = 0.537, 0.387, 0.237, 0, -0.213, -0.363, -0.513, -0.663, -0.813 m
                        A_phi[:, :] = A_phi_0[:, :] + (I_tor_def_new - np.sum(I_tor[0:20])) / 9 \
                                     * np.sum(
                                             [weights[i] * A_0[:, :, ikr, ikz[i]] for i in range(9)], axis = 0
                                             )/np.sum(weights)

                    elif t_ana >= t_inj_0 or t_ana < t_decay:     #Bz 分布が2次関数のFitting の方が支配的
                        # min(Bz_puc) を中心に9点
                        A_phi[:, :] = A_phi_0[:, :] + (I_tor_def_new - np.sum(I_tor[0:20])) / 9 \
                                        * (
                                              A_0[:,:,ikr,ikz-1] + A_0[:,:,ikr,ikz] + A_0[:,:,ikr,ikz+1] 
                                            + A_0[:,:,ikr-1,ikz-1] + A_0[:,:,ikr-1,ikz] + A_0[:,:,ikr-1,ikz+1]
                                            + A_0[:,:,ikr+1,ikz-1] + A_0[:,:,ikr+1,ikz] + A_0[:,:,ikr+1,ikz+1]
                                            )

                    rh = I_tor_def_new - np.sum(I_tor[:-1])

            I_tor[-1] *= lamb[-1]
            A_phi[:,:] += A_phi_close[:,:]*lamb[-1] + A_phi_open[:,:]


            #前回計算結果(A_phi_before)と比較
            SM = np.sum((A_phi[:,:] - A_phi_before[:,:])**2) / ((ir_max+1)*(iz_max+1))

            if SMmin[ik-1] > SM and rh > 1.e29:
                SMmin[ik-1] = SM


        #. -- Save text and graph --
            # 2024.10.30 update by MOTOKI
            if t_ana >= t_decay:
                update_con = (
                    SM < max(RSM, 1e-8) and
                    rh > 1e29 and
                    emh
                )
            else:
                update_con = (
                    SM < max(RSM, 1e-8) and
                    rh > 1e29 and
                    lambda_fac > np.abs(lamb[-1]) and
                    emh
                )

            if update_con:
                plot_counter += 1
                RSM = SM
                mf_output = A_phi[:,:] * 2*pi*r[:,np.newaxis]

                jf = []
                flux_mag_max = max(np.max(np.abs(flux_mag_r[:,:])), np.max(np.abs(flux_mag_z[:,:])))

                for iz in range(iz_max):
                    for ir in range(ir_max):

                        # for vector figure
                        jf_con = (
                            iz != iz_max and
                            ir != ir_max and
                            iz%3 == 0 and
                            ir%3 == 0
                        )

                        if jf_con:
                            br = -((A_phi_before[ir,iz+1] + A_phi_before[ir+1,iz+1])*0.5 \
                               - (A_phi_before[ir,iz] + A_phi_before[ir+1,iz])*0.5) / dz
                            bz = ((A_phi_before[ir+1,iz] + A_phi_before[ir+1,iz+1])*0.5*r[ir+1] \
                               - (A_phi_before[ir,iz] + A_phi_before[ir,iz+1])*0.5*r[ir])/dr/(r[ir]+0.5*dr)
                            if wq_f[ir,iz,3] == 0:
                                jr = 0
                                jz = 0
                            else:
                                jr = br * lamb[wq_f[ir,iz,3]-1]*(r[ir]+0.5*dr)/np.abs(I_inj_total)/2*(1/flux_mag_max)*elf[ir,iz]
                                jz = bz * lamb[wq_f[ir,iz,3]-1]*(r[ir]+0.5*dr)/np.abs(I_inj_total)/2*(1/flux_mag_max)*elf[ir,iz]
                            
                            jf.append([r[ir]+0.5*dr, z[iz]+0.5*dz, jr, jz])

                jf_array = np.array(jf)

                #. Flux file
                mv_flux = np.zeros((118,5))
                for k in range(118):
                    if flux[k,0] > 0:
                        ir_f = nint((flux[k,0] - r_min)/dr)
                        iz_f = nint((flux[k,1] - z_min)/dz)
                        br_fl, bz_fl = cal_B(ir_f, iz_f, dr, dz, A_phi)

                        mv_flux[k,:] = [flux[k,0], flux[k,1], 
                                        A_phi[ir_f,iz_f]*2*pi*flux[k,0], 
                                        br_fl, bz_fl]

                #. Comaparison PUC to Cal. Bz
                # cal Bz PUC position 
                Bz_cal_puc = np.zeros((len(bz_puc_it),2)) 
                for j, jz in enumerate(iz_puc):
                    Bz_cal_puc[j,0] = z[jz]
                    _, Bz_cal_puc[j,1] = cal_B(ir_puc, jz, dr, dz, A_phi)

                # Cal Sq error
                if i >= 2:
                    sq_error[i] = error_bz(Bz_cal_puc, Bz_vac_puc, fit_params, weights)

                #if t_ana >= t_decay:
                save_con = (sq_error[i] <= min(sq_error[2:])) and i >= 2
                #else:
                #    save_con = True

                if save_con:
                    I_close_store.append([int(i), I_tor[-1]])
                    
                    # save txt
                    np.savetxt(f"{time_path}/M_field_{i:03}.csv", mf_output.T, delimiter = ",", fmt = "%12.4e")
                    np.savetxt(f"{time_path}/J_field_{i:03}.csv", jf_array, delimiter = ",", fmt = "%12.4e")
                    np.savetxt(f"{time_path}/MV_field_flux_{i:03}.csv", mv_flux, fmt = "%.3e", delimiter = ",")

                    # plot Z- Bz distribution
                    plot_z_Bz(Bz_cal_puc, Bz_vac_puc, time_path, i, z_puc, bz_puc_it, sq_error[i])
                    z_Bz_data = np.array([Bz_cal_puc[:,0], Bz_cal_puc[:,1], Bz_vac_puc[:,1]])
                    np.savetxt(f"{path}/data/z_Bz_{t_ana:.3f}.csv", z_Bz_data.T, fmt = "%.3e", delimiter = ",")


                    #. plot 
                    # mag. line of Last closed flux surface
                    ir_in, iz_in = minimum_position(wq_l)
                    ir_in, iz_in = ir_in + 1, iz_in + 1
                    m_in = A_phi[ir_in,iz_in] * 2*pi*r[ir_in]

                    ir_out = nint((ele_lim[0] - r_min)/dr) + 1
                    iz_out = nint((ele_lim[1] - z_min)/dz)
                    m_out = A_phi[ir_out,iz_out] * 2*pi*ele_lim[0]
                    
                    plot_field(m_in, m_out, I_tor[-1], path, time_path, i)

            
            if plot_counter > 50:
                break
        #. -- 



    #. == end calculate lambda ==


        # Update Motoki 2024.11.12
        I_close_store = np.array(I_close_store)
        # if t_ana >= t_decay:
        #     i_error_best = np.argmin(np.abs(I_close_store[:,1] - I_close_before))
        #     i_best = int(I_close_store[i_error_best,0])
        #     I_close_before = I_close_store[i_error_best,1]
        # else:
        #     i_best = int(I_close_store[-1,0])
        i_best = int(I_close_store[-1,0])
        
        copy_file(f"{path}/{t_ana:.3f}/M_field_{i_best:03}.csv", f"{path}/data/M_field_{t_ana:.3f}.csv")
        copy_file(f"{path}/{t_ana:.3f}/J_field_{i_best:03}.csv", f"{path}/data/J_field_{t_ana:.3f}.csv")
        copy_file(f"{path}/{t_ana:.3f}/MV_field_flux_{i_best:03}.csv", f"{path}/data/MV_field_flux_{t_ana:.3f}.csv")

        plot_field2(m_in, t_ana, path)
        plot_psi(m_in, t_ana, path)
        plot_z_Bz2(path, t_ana, z_puc, bz_puc_it, min(sq_error[2:]))
        m_in_data.append([t_ana, m_in])



#. == Time loop end ==
    m_in_data = np.array(m_in_data)
    np.savetxt(f"{path}/data/m_in.csv", m_in_data, fmt = "%.4e", delimiter = ",")

    #. Time
    time_e = time.time()
    total_time = (time_e - time_s)
    
    print("\n"+f"Time loop Total time : {(total_time)//3600:.0f}h {(total_time)%60:.0f}m {(total_time)%60:.0f}s")