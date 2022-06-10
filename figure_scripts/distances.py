"""
Plot distances
K. Luchsinger
Last updated: 12/16/21 (CJTU)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from moonpies import moonpies as mp
from moonpies import default_config

# Set Fig paths
FIGDIR = ''  # Set or leave blank to use default (moonpies/figs)
if not FIGDIR:
    FIGDIR = default_config.Cfg().figs_path
FIGDIR = str(Path(FIGDIR).resolve() / "_")[:-1]  # add trailing slash

CFG = default_config.Cfg(mode='moonpies')

fig_dir = "/home/kristen/codes/code/moonpies_package/figs/"

# Values from Ries crater (citation)
ries = np.array([[1.375,0.9],[1.583333333,0.29], [1.916666667,0.22], [2.125,0.09], [2.233333333,0.21], [2.291666667,0.17], [2.666666667,0.07], [2.933333333,0.18], [3.041666667,0.10]])

# Get mixing ratio and vol frac
df = mp.get_crater_list(basins=True)
dists = mp.get_coldtrap_dists(df, CFG)
thick = mp.get_ejecta_thickness_matrix(df, CFG)
mixing_ratio = mp.get_mixing_ratio_oberbeck(dists, CFG)
vol_frac = mp.mixing_ratio_to_volume_fraction(mixing_ratio)


# #DT = np.load("temp.npy") 
# #DT = DT + 250
# R = df["rad"]

# #d = np.linspace(1.*R,4.*R,len(DT[0]))/1000 
# #e_temp = [200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400]
# #dist_array = np.zeros((len(R), len(e_temp)))

# c1 = 11
# c2 = 0

# #ej_T = dtemp(ej_distances[c1,c2]/1000)
# vf = 1 - (2.913*(ej_distances[c1,c2]/df["rad"][c1])**(-3.978))

# #thick = 0.14 * (df["rad"][c1]**0.74) * ((ej_distances[c1,c2]*df["rad"][c1]/df["rad"][c1])**-3.)# pm 0.5
# #thick = get_ejecta_thickness(ej_distances[c1,c2], df["rad"][c1])

# #print("From", df["cname"][c1], "to", df["cname"][c2], "is volume fraction", vf, "at", ej_T, "Kelvin and ", ej_distances[c1,c2], "m with", thick, "m thickness")

# #print(ej_distances[c1,c2] / (4 * df["rad"][c1]))
# dists = 60*1e3*np.linspace(1.35, 4., 449) #crater radii
# vf_plot = 1-(2.913*(dists**(-3.978)))

# rads = df["rad"][3] 

# ries = np.array([[1.375,0.9],[1.583333333,0.29], [1.916666667,0.22], [2.125,0.09], [2.233333333,0.21], [2.291666667,0.17], [2.666666667,0.07], [2.933333333,0.18], [3.041666667,0.10]])

# thick = 0.14 * (rads**0.74) * ((dists/rads)**-3.)# pm 0.5

# #depth = thick * (3-1) #((1/(1-vf_plot)) - 1)
# print(df["rad"][3])

# cfg = default_config.Cfg(mode='moonpies')

# #depth = thick * (3-1) #((1/(1-vf_plot)) - 1)
# print(df["rad"][3])

# cfg = default_config.Cfg(mode='moonpies')
# vol_frac = cfg.vol_frac_a * 1e-3 * dists ** cfg.vol_frac_b

# if cfg.vol_frac_petro:
#     vol_frac[vol_frac > 5] = 0.5 * vol_frac[vol_frac > 5] + 2.5

# print(ej_distances[3,2])
plt.figure(1, figsize=(6,4))
idx = df[df.cname == 'Shackleton'].index[0]
dist_crad = dists/df.rad.values[:, None]
plt.figure(1, figsize=(6,4))
plt.plot(dist_crad[idx], 1-vol_frac[idx], 'x')
plt.plot(ries[:,0], ries[:,1], 'ro')
mp.plot_version(CFG, loc='lr')
plt.title("Volume Fraction Function with Points from Ries Crater")
plt.xlabel("Distance [Crater Radii]")
plt.ylabel("Volume Fraction [%Target Material]")
# plt.show()

plt.figure(2, figsize=(6,4))
plt.plot(dists, thick, label="Ejecta thickness")
plt.plot(dists, np.zeros(len(thick)), label="Original Surface")
plt.plot(dists, -1*thick*vol_frac, label="Ballistic Sed. Mixing Region")
plt.axvline(dists[3,2], color='red', ls='--', label='Amundsen->Faustini')
plt.axvline(dists[3,1], color='orange', ls='--', label='Amundsen->Shoemaker')
plt.axvline(dists[3,0], color='gold', ls='--', label='Amundsen->Haworth')
#plt.plot(dists, DT[3], label = "Ejecta Temperature [K]")
mp.plot_version(CFG, loc='lr')
plt.legend()
plt.title("Ballistic Sedimentation Mixing Region")
plt.xlabel("Distance [m]")
plt.ylabel("Thickness [m]")
plt.savefig(FIGDIR + "teq_mixing_region.png", dpi=300)
# plt.show()
