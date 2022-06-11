"""
Plots ice from basin impactors
K. Frizzell
Last updated 11/16/21 (CJTU)
"""
import numpy as np
import matplotlib.pyplot as plt 
from moonpies import moonpies as mp
from moonpies import default_config

# Set Fig paths
FIGDIR = ''  # Set or leave blank to use default (moonpies/figs)
if not FIGDIR:
    FIGDIR = default_config.Cfg().figs_path
FIGDIR = str(Path(FIGDIR).resolve() / "_")[:-1]  # add trailing slash

# IMPORTANT: Need to set ctype_frac and ctype_hydrated to both be 1 in the config
def_cfg = default_config.Cfg(mode = 'moonpies', ctype_frac=1, ctype_hydrated=1)

df = mp.read_basin_list(def_cfg)
time_arr = mp.get_time_array(def_cfg)
b_ice_t = mp.get_basin_ice(time_arr, df, def_cfg)

i=0
n = 100
plt.figure(figsize=(8,5))
mp.plot_version(def_cfg,loc='ul')
all_basin_ice = np.zeros([len(time_arr),n])
mean_basin_ice = np.zeros_like(time_arr)
for i in range(n):
    mp.clear_cache()
    time_arr = mp.get_time_array(def_cfg)
    b_ice_t = mp.get_basin_ice(time_arr, df, def_cfg)
    all_basin_ice[:,i] = b_ice_t
    y = b_ice_t
    x = time_arr/1e9
    x = x[b_ice_t>0]
    y = y[b_ice_t>0]    
    plt.plot(x,y,'x')
plt.xlim(4.25,3.7)
plt.ylim(0,120)
plt.grid('on')
plt.title('Ice Delivered to South Pole by Basins during Basin-forming Epoch')
plt.xlabel('Age [Ga]')
plt.ylabel('Ice Thickness [m]')
mean_basin_ice = np.mean(all_basin_ice,axis=1)
median_basin_ice = np.median(all_basin_ice,axis=1)
pct95 = np.percentile(all_basin_ice,95,axis=1) #2 sigma

plt.plot(time_arr/1e9,mean_basin_ice,'r-',lw = 3,label='Mean')
plt.plot(time_arr/1e9,pct95,'k-',lw = 3,label = '95th Percentile')
plt.legend(loc='upper right')
plt.savefig(FIGDIR + 'Basin_Ice.png', dpi=300)