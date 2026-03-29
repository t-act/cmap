#. Last update 2024.07.25

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from cmap.utils import nint, cd_main


"""
Overview
---
Import parameter file


Parameter
---
None


Return
---
- elect0 (np.array) : electrode position
- elect (np.array)  : Injection current positions
- flux (np.array)   : Flux loop positions


Example
---
main.py
>>> from Parameter import elect0, elect, flux


"""

#. outside of electrode position
r_ele_lim = 0.47
z_ele_lim = -1.165
ele_lim = [r_ele_lim, z_ele_lim]

#. electrode position
elect0 = np.zeros((20, 3))
elect0[0, :] = [0.389, -1.315, 1.0]
elect0[1, :] = [0.389, -1.165, 1.0]
elect0[2, :] = [0.500, -1.165, 1.0]


# Injection current position // 容器境界の位置
elect = np.array([
    [0.50, -1.165, 1.0],     # el1
    [0.389, -1.165, 1.0],    # el2
    [0.389, -1.32, 1.0],     # g_in1
    [0.28, -1.32, 1.0],      # g_in2
    [0.28, -1.13, 1.0],      # g_cs0
    [0.22, -1.13, 1.0],      # g_cs1
    [0.22, -0.49, 1.0],      # g_cs2
    [0.22, 0.11, 1.0],       # g_cs3
    [0.22, 0.86, 1.0],       # g_cs4
    [0.22, 1.0, 1.0],        # g_top

    [0.722, 1.0, 0.0],       # g_up-out-wall
    [1.2, 0.284, 0.0],
    [1.2, -0.284, 0.0],
    [0.719, -0.918, 0.0],
    [0.753, -1.165, 0.0],
    [0.50, -1.165, 0.0],     # same as elect[0]
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1]
    ])


#. Flux loop 
flux = np.zeros((118, 2))   #R, Z

# FLC
flux[:24, 0] = 0.1985
flux[:24, 1] = np.linspace(1.15, -1.15, 24)

# Vessel side top (FLT)
flux[24:30, :] = np.array([
    [0.214746, 1.300],
    [0.263, 1.394],
    [0.37479, 1.394],
    [0.47479, 1.394],
    [0.57479, 1.394],
    [0.699524, 1.394]
])

# Vessel side semi top (FLTS)
flux[30:41, :] = np.array([
    [0.732669, 1.350],
    [0.770334, 1.300],
    [0.827584, 1.224],
    [0.894628, 1.135],
    [0.953385, 1.057],
    [1.015532, 0.9745],
    [1.071652, 0.900],
    [1.133799, 0.8175],
    [1.202726, 0.726],
    [1.271653, 0.6345],
    [1.326267, 0.562]
])

# Vessel side center (FLS)
flux[41:50, :] = np.array([
    [1.374, 0.450],
    [1.374, 0.327],
    [1.374, 0.257],
    [1.374, 0.1335],
    [1.374, 0.0],
    [1.374, -0.137],
    [1.374, -0.263],
    [1.374, -0.3325],
    [1.374, -0.481]
])

# Vessel side ???
flux[50:61, :] = np.array([
    [1.327773, -0.560],
    [1.270146, -0.6365],
    [1.200843, -0.7285],
    [1.136813, -0.8135],
    [1.077679, -0.892],
    [1.013649, -0.977],
    [0.951878, -1.059],
    [0.886342, -1.146],
    [0.834364, -1.215],
    [0.784270, -1.2815],
    [0.732669, -1.350]
])

# ????
flux[61:67, :] = np.array([
    [0.699524, -1.394],
    [0.574790, -1.394],
    [0.474790, -1.394],
    [0.374790, -1.394],
    [0.263, -1.394],
    [0.214746, -1.300]
])

# ???
flux[67:79, :] = np.array([
    [0.23, 0.05],
    [0.21, 0.05],
    [0.19, 0.05],
    [0.17, 0.05],
    [0.15, 0.05],
    [0.13, 0.05],
    [0.11, 0.05],
    [0.09, 0.05],
    [0.07, 0.05],
    [0.05, 0.05],
    [0.03, 0.05],
    [0.01, 0.05]
])

# ???
flux[79, :] = [1.345, 0.159]

# Input injection current position
flux[80:99, :] = elect[:19, :2]

# Input electrode position
flux[99:118, :] = elect0[:19, :2]



#. -- Coordinate --
ir_min = 0
ir_max = 100
dr = 0.02
r_min = 0.0
r_max = 2.0
r = np.arange(r_min, r_max+dr, dr)

iz_min = 0
iz_max = 200
dz = 0.02
z_min = -2.0
z_max = 2.0
z = np.arange(z_min, z_max+dz, dz)

#. -- Initial position of Toroidal current filament. --
ik = 0
rxc0 = 0.3
rzc0 = -1.0
ikr0 = nint((rxc0 - r_min)/dr)
ikz0 = nint((rzc0 - z_min)/dz)
SMmin = np.array([1.e30, 1.e30, 1.e30])

#. -- PF coil position --
r_c, z_c  = np.zeros(55), np.zeros(55)
#. PF coil positions
#PF3-1, 2
r_c[[0, 1]] = 0.273, 0.8
z_c[[0, 1]] = 1.615

# PF2, 6
r_c[[2, 5]] = 1.2608
z_c[[2, 5]] = 1.035, -1.035

# PF1, 7
r_c[[3, 4]] = 1.5552
z_c[[3, 4]] = 0.54, -0.54

# PF5-1, 2
r_c[[6, 7]] = 0.800, 0.273
z_c[[6, 7]] = -1.615

# PF4-1
r_c[8] = 0.1574
z_c[8] = 0.740

#. PF4-2
r_c[9] = 0.1632
z_c[9] = 0.0

#. PF4-3
r_c[10] = 0.1574
z_c[10] = -0.740

#PF4-1,2,3 = 0のため意味なさそう
k = 10 #k = 10

for i in range(8, 11):
    for j in range(1, 9):
        k += 1
        r_c[k] = r_c[i]
        z_c[k] = z_c[i] + j*0.05

        k += 1
        r_c[k] = r_c[i]
        z_c[k] = z_c[i] - j*0.05

        if (i == 9 or i == 11) and j == 6:
            break






if __name__ == "__main__":
    cd_main()
    #. plot
    fig = plt.figure(figsize = (5, 10))
    ax = fig.add_subplot()
    plt.rcParams["font.family"] = "Arial"
    ax.tick_params(labelsize = 14)
    qvessel_img = Image.open(f"modules/Qvessel.png")
    ax.imshow(qvessel_img, extent = [0, 2, -2, 2], aspect = 'auto')

    
    ax.plot(elect[:16,0], elect[:16,1], markersize = 8, marker = "o",
            color = "steelblue")
    ax.plot(elect0[:2,0], elect0[:2,1], markersize = 8, marker = "o",
            color = "steelblue")
    # for n in range(16):
    #     ax.annotate(f"{n}", (elect[n, 0] - 0.025, elect[n, 1] + 0.03), fontsize=13)
    
    ax.set_xlabel("R [m]", fontsize = 16)
    ax.set_ylabel("Z [m]", fontsize = 16)
    ax.set_xlim(0, 2)
    ax.set_ylim(-2, 2)
    path = "/Users/tact/Documents/01_Lab/AnnualMeeting"
    fig.savefig(f"{path}/vacuum_vessel.png",
                dpi = 300,
                bbox_inches = 'tight', pad_inches = 0.1)
