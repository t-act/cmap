import os
import numpy as np
import matplotlib.pyplot as plt


class get_CHI_Data:
    """CHI 実験データローダー。

    Parameters
    ----------
    count : int
        ショット番号
    switch : bool
        True = macOS (後方互換用)。local_base が指定された場合は無視される。
    local_base : str or None
        ローカルデータディレクトリのパス。
        指定した場合、サーバーマウント (/Volumes/share) の代わりに使用する。
        サーバーと同じサブディレクトリ構造でファイルを配置すること。

        例::

            local_base="data"
            → ip   : data/yokogawa/oscillo3/<date>_000/<count>.csv
            → inj  : data/chi/<date>/<count>.csv
            → bz   : data/magnetics/mag_<count>.csv
            → G    : data/Motoki/I_Injector/<count>_Iinj_p.csv
    """

    def __init__(self, count: int, switch: bool = True, local_base: str | None = None):
        self.count = count
        self.date = self.check_date(count)
        self.switch = switch  # 後方互換

        if local_base is not None:
            self._share = local_base
        elif switch:
            self._share = "/Volumes/share"
        else:
            # Windows 用パスは未実装
            self._share = None

    # ------------------------------------------------------------------
    def get_vloop(self):
        """外側フラックスループ (T.MOTOKI 設置)。

        Returns
        -------
        t      (np.array) : [s] 10MS/s
        FL6_M  (np.array) : [V] Z = 650 mm
        FL9_M  (np.array) : [V] Z = 350 mm
        FL19_M (np.array) : [V] Z = -650 mm
        """
        path = f"{self._share}/SL1000"
        fname = f"{path}/{self.date}/0{self.count}/TFL_0{self.count}.csv"

        data = np.loadtxt(fname, delimiter=",", usecols=[0, 5, 6, 7, 8])
        data_max = len(data[:, 0])
        t = np.cumsum(np.full(data_max, 1e-7))
        return t, data[:, 3] / 2, data[:, 2] / 2, data[:, 4] / 2  # FL6_M, FL9_M, FL19_M

    # ------------------------------------------------------------------
    def get_ip(self):
        """プラズマ電流 Ip。

        Returns
        -------
        t  (np.array) : [s] 1MS/s
        Ip (np.array) : [kA]
        """
        fname = f"{self._share}/yokogawa/oscillo3/{self.date}_000/{self.count}.csv"
        data = np.loadtxt(fname, skiprows=10, delimiter=",", usecols=[2])
        data_max = len(data)
        dt = 1e-6

        t = np.cumsum(np.full(data_max, dt)) - dt

        total = np.sum(data[int(data_max / 2):])
        data_ave = data - total / (data_max / 2)

        integ_data = np.cumsum((data_ave[1:] + data_ave[:-1]) / 2) * dt
        integ_data = np.hstack((0, integ_data)) * 17974

        return t, integ_data

    # ------------------------------------------------------------------
    def get_inj(self):
        """入射電流。

        Returns
        -------
        t     (np.array) : [s] 1MS/s
        I_inj (np.array) : [kA]
        """
        fname = f"{self._share}/chi/{self.date}/{self.count}.csv"
        data = np.loadtxt(fname, skiprows=10, delimiter=",", usecols=[6])
        data_max = len(data)
        dt = 1e-6

        t = np.cumsum(np.full(data_max, dt)) - 5e-3
        data *= 3e4

        return t, data * 1e-3

    # ------------------------------------------------------------------
    def get_fhx(self):
        """高速硬 X 線。

        Returns
        -------
        t        (np.array) : [t]
        BW count (np.array) : [count/s]
        FW count (np.array) : [count/s]
        """
        path = f"{self._share}/Motoki/FHX"
        TDC3 = np.loadtxt(f"{path}/{self.count}_TDC3.txt", skiprows=1, delimiter=" ")
        TDC4 = np.loadtxt(f"{path}/{self.count}_TDC4.txt", skiprows=1, delimiter=" ")

        dt = 1e-5
        t = np.arange(3.5, 3.6, dt)

        tdc3h, _ = np.histogram(TDC3, bins=t)
        tdc4h, _ = np.histogram(TDC4, bins=t)

        return t[:-1] - 3.5, tdc3h, tdc4h  # BW, FW

    # ------------------------------------------------------------------
    def get_thom(self, t_set):
        """Thomson 散乱。

        Parameters
        ----------
        t_set : float  Thomson 計測時刻

        Returns
        -------
        Radius, Te, dTe, ne18, dne18, Pa, dPa  または全 None
        """
        if self.switch:
            path = "/Volumes/Thomson/data/temp"
        else:
            path = None

        if path is None:
            return (None,) * 7

        fname = f"{path}/{str(self.count)[:2]}000/Thomson_{self.count}@{t_set}s.dat"
        if os.path.exists(fname):
            data = np.loadtxt(fname, skiprows=22, delimiter=",")
            return (data[:, i] for i in range(1, 8))
        return (None,) * 7

    # ------------------------------------------------------------------
    def get_bz(self):
        """ピックアップコイル Bz。

        Returns
        -------
        t  (np.array) : [s] 1MS/s
        Bz (np.array) : [T]
            [0]  Z =    0 mm,  [1]  Z = +687 mm,  [2]  Z = +537 mm
            [3]  Z = +387 mm,  [4]  Z = +237 mm,  [5]  Z = + 87 mm
            [6]  Z = - 63 mm,  [7]  Z = -213 mm,  [8]  Z = -363 mm
            [9]  Z = -513 mm,  [10] Z = -663 mm,  [11] Z = -813 mm
            ref. TAKEDA, master thesis. P18
        """
        path = f"{self._share}/magnetics"
        data = np.loadtxt(f"{path}/mag_{self.count}.csv", delimiter=",")
        t = data[:, 0]
        ref_data = np.genfromtxt(
            f"{path}/MParray_setting{self.count}.txt",
            skip_header=1, skip_footer=1, usecols=[6, 7, 9], dtype=float,
        )
        amp, atten, area = ref_data[1:13, :].T
        area *= 1e-4
        vol_z = -data[:, 1:13] / (amp * atten)

        vol_z_pad = np.vstack((np.zeros((1, 12)), vol_z))

        it_s, it_e = 18000, 40000
        vol_z_pad = vol_z_pad[it_s:it_e, :]
        t = t[it_s:it_e - 1]

        vol_trapz = np.cumsum((vol_z_pad[1:] + vol_z_pad[:-1]) / 2, axis=0) * (t[1] - t[0])

        drift = (vol_trapz[-1, :] - vol_trapz[0, :]) / (t[-1] - t[0])
        vol_trapz -= drift * (t - t[0])[:, np.newaxis]

        bz = vol_trapz / area
        return t, bz

    # ------------------------------------------------------------------
    def get_G(self):
        """壁への注入電流。

        Returns
        -------
        t (np.array) : [s] 1MS/s
        G (np.array) : [A] Iinj1–5
        """
        fname = f"{self._share}/Motoki/I_Injector/{self.count}_Iinj_p.csv"
        data = np.loadtxt(fname, delimiter=",", skiprows=1)
        drift = (data[-1, 1:6] - data[0, 1:6]) / (data[-1, 0] - data[0, 0])
        data[:, 1:6] -= drift * (data[:, 0] - data[0, 0])[:, np.newaxis]

        return data[:, 0], -data[:, 1:6] * 1e3

    # ------------------------------------------------------------------
    def check_date(self, count: int) -> str:
        if 52863 <= count < 52885:
            return "20240112"
        elif 52885 <= count < 52912:
            return "20240116"
        elif 52912 <= count < 52954:
            return "20240117"
        elif 52954 <= count < 53001:
            return "20240118"
        elif count >= 53001:
            return "20240119"


if __name__ == "__main__":
    s = get_CHI_Data(53044, True)
    r, a, b, c = s.get_vloop()
