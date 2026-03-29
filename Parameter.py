#. Last update 2024.07.25
#  Phase 2 refactor: Grid / TokamakConfig を利用。後方互換エクスポートを維持。

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from cmap.utils import nint, cd_main
from cmap.grid import Grid
from cmap.tokamak_config import TokamakConfig


# ---------------------------------------------------------------------------
# インスタンス生成
# ---------------------------------------------------------------------------
_grid = Grid()
_config = TokamakConfig()


# ---------------------------------------------------------------------------
# 後方互換エクスポート (main.py が参照する名前をそのまま維持)
# ---------------------------------------------------------------------------
ele_lim  = _config.ele_lim
elect0   = _config.elect0
elect    = _config.elect
flux     = _config.flux
r_c      = _config.r_c
z_c      = _config.z_c
SMmin    = _config.SMmin

#. ソルバー初期値 (Phase 5 で EquilibriumSolver に移動予定)
ik   = 0
rxc0 = 0.3
rzc0 = -1.0
ikr0 = nint((rxc0 - _grid.r_min) / _grid.dr)
ikz0 = nint((rzc0 - _grid.z_min) / _grid.dz)


if __name__ == "__main__":
    cd_main()
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot()
    plt.rcParams["font.family"] = "Arial"
    ax.tick_params(labelsize=14)
    qvessel_img = Image.open("modules/Qvessel.png")
    ax.imshow(qvessel_img, extent=[0, 2, -2, 2], aspect='auto')

    ax.plot(elect[:16, 0], elect[:16, 1], markersize=8, marker="o", color="steelblue")
    ax.plot(elect0[:2, 0], elect0[:2, 1], markersize=8, marker="o", color="steelblue")

    ax.set_xlabel("R [m]", fontsize=16)
    ax.set_ylabel("Z [m]", fontsize=16)
    ax.set_xlim(0, 2)
    ax.set_ylim(-2, 2)
    path = "/Users/tact/Documents/01_Lab/AnnualMeeting"
    fig.savefig(f"{path}/vacuum_vessel.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
