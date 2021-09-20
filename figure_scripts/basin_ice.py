#Plots ice from basin impactors
#IMPORTANT: Before running- edit ctype_frac and ctype_hyd to both be 1 in the config
#K. Frizzell
#Last updated 8/5/21

import moonpies as mp
import default_config
import matplotlib.pyplot as plt 
import numpy as np
def_cfg = default_config.Cfg(mode = 'moonpies')

df = mp.read_basin_list(def_cfg)
time_arr = mp.get_time_array(def_cfg)
b_ice_t = mp.get_basin_ice(time_arr,def_cfg)

i=0
n = 100
plt.figure(figsize=(8,5))
mp.plot_version(def_cfg,loc='ul')
all_basin_ice = np.zeros([len(time_arr),n])
mean_basin_ice = np.zeros_like(time_arr)
for i in range(n):
    mp.clear_cache()
    time_arr = mp.get_time_array(def_cfg)
    b_ice_t = mp.get_basin_ice(time_arr,def_cfg)
    all_basin_ice[:,i] = b_ice_t
    mean_basin_ice = np.mean([mean_basin_ice,b_ice_t],axis=0)#ppend(b_ice_t)
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
median_basin_ice = np.median(all_basin_ice,axis=1)
pct95 = np.percentile(all_basin_ice,95,axis=1) #2 sigma

plt.plot(time_arr/1e9,mean_basin_ice,'r-',lw = 3,label='Mean')
plt.plot(time_arr/1e9,pct95,'k-',lw = 3,label = '95th Percentile')
plt.legend(loc='upper right')
plt.savefig('Basin_Ice.jpg', dpi=300)