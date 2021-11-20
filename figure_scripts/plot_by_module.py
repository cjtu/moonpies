'''
Script to plot 4 timesteps of the model in both cannon and moonpies mode. At each timestep, 
every module is plotted as its own bar along with the gardening rate at that timestep.
This script only plots 4 timesteps, though more can be added with a bit of work if desired. 
(Sorry it isn't easily extendable, just ran out of time)
'''
from pathlib import Path

import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
from moonpies import moonpies as mp
from moonpies import default_config

matplotlib.rcParams.update({'font.size': 14})

# Set Fig paths
FIGDIR = ''  # Set or leave blank to use default (moonpies/figs)
if not FIGDIR:
    FIGDIR = default_config.Cfg().figspath
FIGDIR = str(Path(FIGDIR).resolve() / "_")[:-1]  # add trailing slash

# Create configuration objects for each mode. 
seed = 91
cfg_mp = default_config.Cfg(mode='moonpies', volc_mode='Head', seed=seed)
cfg_cn = default_config.Cfg(mode='cannon', seed=seed)
mp.clear_cache()

# Random number generator for each run
rng_mp = mp.get_rng(cfg_mp)
rng_cn = mp.get_rng(cfg_cn)

# Time array won't vary between runs
TIME_ARR = mp.get_time_array(cfg_mp)

# Geologic eras
geo_labels_abbr = ['pNe.', 'Ne.', 'Im.', 'Era', 'Cop', '-']
geo_labels = ['Pre-Nectarian (>3.95 Ga)', 'Nectarian (3.95-3.85 Ga)', 'Imbrian (3.85-3.2 Ga)', 'Eratosthenian (3.2-1.1 Ga)', 'Copernican (1.1 Ga - pres.)', 'Present']
geo_ages = [4.26, 3.95, 3.85, 3.2, 1.1, 0]
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

    # Solar wind ice
    solar_wind_ice = mp.get_solar_wind_ice(TIME_ARR, cfg)
    
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

    # Gardening
    gardening = mp.overturn_depth_time(TIME_ARR, cfg)
    return [total, basin_ice, large_complex_craters_ice, micrometerorite_ice, large_simple_craters_ice, volc_ice, solar_wind_ice, small_impactor_ice, small_simple_ice, gardening]

def ice_wrangle(ice, cfg, t_now, time_val):
    '''
    Returns dataframe of all modules with impact gardening as well. 
    '''
    ice = ice[:-1]  # excl impact gardening
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
        mp.clear_cache()
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

def ice_by_era(ice, geo_ages, geo_labels):
    d = {}
    for i, (age, label) in enumerate(zip(geo_ages[:-1], geo_labels[:-1])):
        tmin = np.searchsorted(-TIME_ARR, -age*1e9)
        tmax = np.searchsorted(-TIME_ARR, -geo_ages[i+1]*1e9)
        d[label] = np.array(ice)[:,tmin:tmax].mean(axis=1)
    columns = [
        'Total Ice', 
        'Basin Ice', 
        'Large complex crater ice',
        'Micrometeorite ice',
        'Large simple crater ice', 
        'Volcanic ice', 
        'Solar wind ice', 
        'Small impactor ice', 
        'Small simple crater ice', 
        'Impact Gardening']
    idf = pd.DataFrame.from_dict(d, orient='index', columns=columns)
    return idf

ice_mp = get_ice_by_module(TIME_ARR, cfg_mp, rng_mp)
ice_cn = get_ice_by_module(TIME_ARR, cfg_cn, rng_cn)

# df_mp = ice_wrangle(ice_mp, cfg_mp, t_now, time_val)
# mp.clear_cache()
# df_cn = ice_wrangle(ice_cn, cfg_cn, t_now, time_val)

df_mp = ice_by_era(ice_mp, geo_ages, geo_labels)
df_cn = ice_by_era(ice_cn,  geo_ages, geo_labels)

# ------------ PLOTTING -----------

# MoonPies plot
f,ax =plt.subplots(figsize=(15, 5))
# plt.minorticks_on()
# df_mp.drop('Impact Gardening', axis=1).plot(x="Time [Ga]", ylabel= 'Deposited Ice Thickness [m]' , kind='bar', stacked=False, logy=True, rot = 0, ax=ax )
df_plot = df_mp.copy()
df_plot.insert(1,'Impactor ice', df_mp[['Micrometeorite ice','Small impactor ice','Small simple crater ice','Large simple crater ice','Large complex crater ice']].sum(axis=1))
df_plot = df_plot.drop(['Impact Gardening','Micrometeorite ice', 'Small impactor ice', 'Small simple crater ice', 'Large simple crater ice', 'Large complex crater ice'], axis=1)
df_plot.plot(ylabel= 'Deposited Ice Thickness [m]' , kind='bar', stacked=False, logy=True, rot = 0, ax=ax )
ax.fill_between([-1, 1, 2, 3, 5], [0]*5, df_mp['Impact Gardening'], color='gray', alpha=0.4, label='Ice Gardened', hatch='/')
# ax.plot([0,1,2,3], df_mp['Impact Gardening'], '-o', color= 'r', label='Ice Gardened')
ax.text(.5,.93 ,'MoonPIES Ice Thickness by Module',
    horizontalalignment='center',
    transform=ax.transAxes,
    fontsize=18)
ax.set_ylim(1e-5, 1e2)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Lunar Geologic Era', fontsize=14)
ax.set_ylabel('Deposited Ice Thickness [m]', fontsize=14)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.plot_version(cfg_mp, loc='ur')
f.savefig(FIGDIR + 'plot_by_mod_mp.png')
# plt.show()

# Cannon mode
f1,ax1 = plt.subplots(figsize=(15, 5))
df_plot = df_cn.copy()
df_plot.insert(1,'Impactor ice', df_cn[['Micrometeorite ice','Small impactor ice','Small simple crater ice','Large simple crater ice','Large complex crater ice']].sum(axis=1))
df_plot = df_plot.drop(['Impact Gardening','Micrometeorite ice', 'Small impactor ice', 'Small simple crater ice', 'Large simple crater ice', 'Large complex crater ice'], axis=1)
df_plot.plot(ylabel= 'Deposited Ice Thickness [m]' , kind='bar', stacked=False, logy=True, rot = 0, ax=ax1)
# ax1.axhline(y=.01, color= 'r', linestyle= '-', label='Ice Gardened')
ax1.fill_between([-1, 1, 2, 3, 5], [0]*5, df_cn['Impact Gardening'], color='gray', alpha=0.4, label='Ice Gardened', hatch='/')
ax1.text(.5,.93 ,'Cannon Ice Thickness by Module',
    horizontalalignment='center',
    transform=ax1.transAxes,
    fontsize=18)
# plt.minorticks_on()
ax1.set_ylim(1e-5, 1e2)
ax1.tick_params(axis='both', labelsize=14)
ax1.set_xlabel('Lunar Geologic Era', fontsize=14)
ax1.set_ylabel('Deposited Ice Thickness [m]', fontsize=14)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
mp.plot_version(cfg_cn, loc='ur')
f1.savefig(FIGDIR + 'plot_by_mod_cn.png')
# plt.show()



