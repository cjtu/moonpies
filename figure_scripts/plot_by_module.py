'''
Script to plot 4 timesteps of the model in both cannon and moonpies mode. At each timestep, 
every module is plotted as its own bar along with the gardening rate at that timestep.
This script only plots 4 timesteps, though more can be added with a bit of work if desired. 
(Sorry it isn't easily extendable, just ran out of time)
'''

import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 

# import sys and use the following command to point to the directory where moonpies.py is (This is not necessary if this script is in that directory)
import sys
sys.path.insert(0, "/Users/tylerpaladino/Documents/ISU/LPI_NASA/Codes/moonpies_package/moonpies")

from moonpies import default_config
import moonpies as mp


mp.clear_cache()

out_dir = '/Users/tylerpaladino/Documents/ISU/LPI_NASA/figs/'

# Create configuration objects for each mode. 
seed = 9531
cfg_mp = default_config.Cfg(mode='moonpies',volc_mode='NK', seed = seed)
cfg_cn = default_config.Cfg(mode='cannon', seed = seed)

# Random number generator for each run
rng_mp = mp.get_rng(cfg_mp)
rng_cn = mp.get_rng(cfg_cn)

# Time array won't vary between runs
TIME_ARR = mp.get_time_array(cfg_mp)

# Indices of timesteps of interests 
t_now = [2, 45, 175, len(TIME_ARR) - 1]

# values at those indices
time_val = [round(TIME_ARR[t_now[0]]/1e9, 2), round(TIME_ARR[t_now[1]]/1e9, 2), round(TIME_ARR[t_now[2]]/1e9, 2), round(TIME_ARR[t_now[3]]/1e9, 2)] 



def get_ice_by_module(TIME_ARR, cfg, rng):
    '''
    Returns tuple of all modules present in MoonPies
    '''

    # Volcanic ice
    volc_ice = mp.get_volcanic_ice(TIME_ARR, cfg)
    # volc_ice = mp.get_ice_thickness(volc_ice_all, cfg)

    # Solar wind ice
    solar_wind_ice = mp.get_solar_wind_ice(TIME_ARR, cfg)

    # solar_wind_ice = mp.get_ice_thickness(solar_wind_ice_all, cfg)
    # Micrometeoroids

    micrometerorite_ice = mp.get_micrometeorite_ice(TIME_ARR, cfg)
    # Small impactors

    small_impactor_ice = mp.get_small_impactor_ice(TIME_ARR, cfg)

    # Small simple craters 

    small_simple_ice = mp.get_small_simple_crater_ice(TIME_ARR, cfg)
    # Large simple craters
    rng = mp.get_rng(cfg)
    large_simple_craters_ice = mp.get_large_simple_crater_ice(TIME_ARR, cfg, rng)
    # Large complex craters
    rng = mp.get_rng(cfg)
    large_complex_craters_ice = mp.get_complex_crater_ice(TIME_ARR, cfg, rng)

    # Basins

    rng = mp.get_rng(cfg)
    basin_ice = mp.get_basin_ice(TIME_ARR, cfg, rng)
    if cfg.mode == 'cannon':
        basin_ice = basin_ice*0



    total = volc_ice + solar_wind_ice + micrometerorite_ice + small_impactor_ice + small_simple_ice + large_simple_craters_ice + large_complex_craters_ice + basin_ice
    
    # return round(time/1e9,2), volc_ice, solar_wind_ice, micrometerorite_ice, small_impactor_ice, small_simple_ice, large_simple_craters_ice, large_complex_craters_ice, basin_ice, total
    return [total, basin_ice, large_complex_craters_ice, micrometerorite_ice, large_simple_craters_ice, volc_ice, solar_wind_ice, small_impactor_ice, small_simple_ice]

def ice_wrangle(ice, cfg, t_now, time_val):
    '''
    Returns dataframe of all modules with impact gardening as well. 
    '''

    ice_t0 = list(np.array(ice)[:,t_now[0]])
    ice_t0.insert(0, time_val[0])

    ice_t1 = list(np.array(ice)[:,t_now[1]])
    ice_t1.insert(0, time_val[1])

    ice_t2 = list(np.array(ice)[:,t_now[2]])
    ice_t2.insert(0, time_val[2])

    ice_t3 = list(np.array(ice)[:,t_now[3]])
    ice_t3.insert(0, time_val[3])

    ice_garden = [] # m
    for i in range(0,len(t_now)):
        ice_garden.append(mp.get_overturn_depth(TIME_ARR, t_now[i], cfg))

    ice_t0.append(ice_garden[0])

    ice_t1.append(ice_garden[1])

    ice_t2.append(ice_garden[2])

    ice_t3.append(ice_garden[3])


    # order of column labels. Must be same as order in get_ice_by_module()
    return pd.DataFrame(data=[ice_t0, ice_t1, ice_t2, ice_t3], 
        columns=['Time [Ga]',
            'Total Ice', 
            'Basin Ice', 
            'Large complex crater ice',
            'Micrometeorite ice',
            'Large simple crater ice', 
            'Volcanic ice', 
            'Solar wind ice', 
            'Small impactor ice', 
            'Small simple crater ice', 
            'Impact Gardening'])


ice_mp = get_ice_by_module(TIME_ARR, cfg_mp, rng_mp)
ice_cn = get_ice_by_module(TIME_ARR, cfg_cn, rng_cn)

df_mp = ice_wrangle(ice_mp, cfg_mp, t_now, time_val)
df_cn = ice_wrangle(ice_cn, cfg_cn, t_now, time_val)


# ------------ PLOTTING -----------

# MoonPies plot
f,ax =plt.subplots(figsize=(15, 5))
df_mp.drop('Impact Gardening', axis=1).plot(x="Time [Ga]", ylabel= 'Deposited Ice Thickness [m]' , kind='bar', stacked=False, logy=True, rot = 0, ax=ax )
ax.plot([0,1,2,3], df_mp['Impact Gardening'], '-o', color= 'r', label='Ice Gardened')
ax.text(.5,.93 ,'MoonPIES Ice Thickness by Module',
    horizontalalignment='center',
    transform=ax.transAxes,
    fontsize=18)
plt.minorticks_on()
matplotlib.rcParams.update({'font.size': 14})
ax.tick_params(axis='both', labelsize=14)
plt.xlabel('Age [Ga]', fontsize=14)
plt.ylabel('Deposited Ice Thickness [m]', fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.plot_version(cfg_mp, loc = 'ur')
f = plt.gcf()
f.savefig(out_dir + 'plot_by_mod_mp.png')
plt.show()


# Cannon Plot (for some reason, when this figure is saved, the title text is put in the wrong place.
# Get around this by saving figure directly form plotting window - sorry!)
f1,ax1 =plt.subplots(figsize=(15, 5))
df_cn.drop('Impact Gardening', axis=1).plot(x="Time [Ga]", ylabel= 'Deposited Ice Thickness [m]' , kind='bar', stacked=False, logy=True, rot = 0, ax=ax1 )
plt.axhline(y=.01, color= 'r', linestyle= '-', label='Ice Gardened')
plt.text(.5,.93 ,'Cannon Ice Thickness by Module',
    horizontalalignment='center',
    transform=ax.transAxes,
    fontsize=18)
plt.minorticks_on()
matplotlib.rcParams.update({'font.size': 14})
ax1.tick_params(axis='both', labelsize=14)
plt.xlabel('Age [Ga]', fontsize=14)
plt.ylabel('Deposited Ice Thickness [m]', fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.plot_version(cfg_cn, loc = 'ur')
f1 = plt.gcf()
# f1.set_size_inches(15,5)
f1.savefig(out_dir + 'plot_by_mod_cn.png')
plt.show()



