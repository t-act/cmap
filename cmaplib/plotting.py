"""プロット関数モジュール。

mfield_sub.py の重複プロット関数を統合する:
  plot_field / plot_field2   → plot_magnetic_field()
  plot_z_Bz  / plot_z_Bz2   → plot_bz_profile()
  plot_psi, plot_t_Bz        そのまま移動
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def _apply_ax_style(ax) -> None:
    """共通軸スタイルを適用する。"""
    plt.rcParams["font.family"] = "Arial"
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position("both")
    ax.tick_params(axis="x", which="major", direction="in")
    ax.tick_params(axis="y", which="major", direction="in")


def plot_magnetic_field(
    m_field_path: str,
    j_field_path: str,
    electrode_path: str,
    save_path: str,
    title: str,
    m_in: float,
    m_out: float = None,
    n_levels: int = 25,
) -> None:
    """磁場コンタと電流ベクトルをプロットする。

    plot_field (反復ごと) と plot_field2 (時刻ごと) を統合した版。

    Parameters
    ----------
    m_field_path   : M_field CSV ファイルパス
    j_field_path   : J_field CSV ファイルパス
    electrode_path : electrode CSV ファイルパス
    save_path      : 保存先 PNG パス
    title          : 図タイトル
    m_in           : 内側 LCFS フラックス値 (コンタ上限)
    m_out          : 外側フラックス値。指定時は [m_in, m_out] の 2 本、
                     None の場合は n_levels 本のコンタを描画
    n_levels       : m_out=None のときのコンタ本数 (デフォルト 25)
    """
    qvessel_img = Image.open("modules/Qvessel.png")
    M_field   = np.loadtxt(m_field_path,    delimiter=",")
    J_field   = np.loadtxt(j_field_path,    delimiter=",")
    electrode = np.loadtxt(electrode_path,  delimiter=",")

    fig, ax = plt.subplots(figsize=(5, 10))
    _apply_ax_style(ax)

    ax.imshow(qvessel_img, extent=[0, 2, -2, 2], aspect='auto')

    # 電流ベクトル (非ゼロ成分のみ)
    jnz = J_field[(J_field[:, 2] != 0) | (J_field[:, 3] != 0)]
    ax.quiver(jnz[:, 0], jnz[:, 1], jnz[:, 2], jnz[:, 3],
              color='red', scale=1, scale_units='xy', width=0.005)

    # 磁場コンタ
    Z = M_field.reshape(201, 101)
    x = np.linspace(0, 201, Z.shape[1])
    y = np.linspace(-201, 201, Z.shape[0])
    X, Y = np.meshgrid(x, y)
    if m_out is not None:
        levels = sorted([m_in, m_out])
    else:
        levels = np.linspace(np.min(Z) / 8, m_in, n_levels)
    ax.contour(X * 1e-2, Y * 1e-2, Z,
               levels=levels, colors="blue", alpha=0.5, linewidths=2)

    # 電極・ピックアップコイル
    ax.plot(electrode[:, 0], electrode[:, 1], color='steelblue', linewidth=5)
    z_puc = np.array([0 if i == 0 else 687e-3 - (i - 1) * 150e-3 for i in range(12)])
    ax.scatter(np.full_like(z_puc, 0.215), z_puc, marker=",", color="black")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"$\mathrm{R \, [m]}$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Z \, [m]}$", fontsize=15)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.close()


def plot_bz_profile(
    z: np.ndarray,
    bz: np.ndarray,
    bz_vac: np.ndarray,
    z_puc: np.ndarray,
    bz_puc: np.ndarray,
    save_path: str,
    title: str,
) -> None:
    """Z-Bz プロファイルをプロットする。

    plot_z_Bz (データ直接渡し) と plot_z_Bz2 (ファイル経由) を統合した版。
    呼び出し側でデータを用意して渡す。

    Parameters
    ----------
    z, bz     : フラックスループ Z 座標と計算 Bz [T]
    bz_vac    : 真空場 Bz [T]
    z_puc     : ピックアップコイル Z 座標 [m]
    bz_puc    : ピックアップコイル計測 Bz [T]
    save_path : 保存先 PNG パス
    title     : 図タイトル
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    fig.subplots_adjust(wspace=0.2, hspace=0.16)
    _apply_ax_style(ax)

    ax.scatter((bz - bz_vac) * 1e3, z, lw=2, marker="o",
               label="Result", c="steelblue")
    ax.scatter(bz_puc * 1e3, z_puc, lw=2, marker="x",
               label="Pick up coil", c="orangered")

    ax.legend(fontsize=15, framealpha=0.0, facecolor="white",
              markerscale=2, handlelength=1)
    ax.set_ylim(np.min(z) * 1.1, np.max(z) * 1.1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"$B_\mathrm{z} \, \mathrm{[mT]}$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Z} \, \mathrm{[m]}$", fontsize=15)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.close()


def plot_psi(m_in: float, t_ana: float, path: str) -> None:
    """磁束面コンタフィルをプロットする。"""
    qvessel_img = Image.open("modules/Qvessel.png")
    M_field   = np.loadtxt(f"{path}/data/M_field_{t_ana:.3f}.csv", delimiter=",")
    electrode = np.loadtxt("modules/electrode.csv", delimiter=",")

    fig, ax = plt.subplots(figsize=(5, 10))
    _apply_ax_style(ax)

    ax.imshow(qvessel_img, extent=[0, 2, -2, 2], aspect='auto')

    Z = M_field.reshape(201, 101)
    x = np.linspace(0, 201, Z.shape[1])
    y = np.linspace(-201, 201, Z.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.contourf(X * 1e-2, Y * 1e-2, Z, cmap="rainbow", alpha=0.5)
    ax.contour(X * 1e-2, Y * 1e-2, Z,
               levels=np.linspace(np.min(Z) / 8, m_in, 25),
               colors="blue", alpha=0.5, linewidths=1.5)

    ax.plot(electrode[:, 0], electrode[:, 1], color='black', linewidth=5)
    z_puc = np.array([0 if i == 0 else 687e-3 - (i - 1) * 150e-3 for i in range(12)])
    ax.scatter(np.full_like(z_puc, 0.215), z_puc, marker=",", color="black")

    ax.set_title(f"t = {t_ana:.3f} ms", fontsize=16)
    ax.set_xlabel(r"$\mathrm{R \, [m]}$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Z \, [m]}$", fontsize=15)

    plt.tight_layout()
    fig.savefig(f"{path}/img/psi_cont_{t_ana:.3f}.png",
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.close()


def plot_t_Bz(t_puc, bz_puc, t_ip, ip, it, count, path) -> None:
    """時刻-Ip/Bz プロファイルをプロットする。"""
    import get_data as g  # QUESTサーバ依存のため遅延インポート
    s = g.get_CHI_Data(count, True)
    t_inj, inj = s.get_inj()

    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    fig.subplots_adjust(wspace=0.2, hspace=0.16)
    plt.rcParams["font.family"] = "Arial"

    for a in ax:
        a.tick_params(labelsize=14)
        a.xaxis.set_ticks_position("both")
        a.tick_params(axis="x", which="major", direction="in")
        a.axvline(t_puc[it] * 1e3, ls=":", c="black")
        a.set_xlim(18.65, 20)

    ax[0].plot(t_ip * 1e3,  -ip,  label=r"$I_{\mathrm{p}}$",   lw=2)
    ax[0].plot(t_inj * 1e3, -inj, label=r"$I_{\mathrm{inj}}$", lw=2)
    ax[0].axhline(0, c="black", alpha=0.5)
    ax[0].invert_yaxis()

    for i in range(12):
        ax[1].plot(t_puc * 1e3, bz_puc[:, i] * 1e3)

    ax[-1].set_xlabel(r"$\mathrm{Time \, [ms]}$", fontsize=15)
    ax[0].set_ylabel(r"$I \, \mathrm{[kA]}$", fontsize=15)
    ax[1].set_ylabel(r"$B_\mathrm{z} \, \mathrm{[mT]}$", fontsize=15)
    ax[0].set_title(f"#{count}, t_ana = {t_puc[it]*1e3:.3f} ms", fontsize=16)

    fig.savefig(f"{path}/img/{count}_Bz.png",
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.close()
