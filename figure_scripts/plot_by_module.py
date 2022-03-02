"""
Script to plot 4 timesteps of the model in both cannon and moonpies mode. At each timestep, 
every module is plotted as its own bar along with the gardening rate at that timestep.
This script only plots 4 timesteps, though more can be added with a bit of work if desired. 
(Sorry it isn't easily extendable, just ran out of time)
Last Update: 21/12/01 (CJTU)
"""
from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from moonpies import moonpies as mp
from moonpies import default_config

mpl.rcParams.update({
    'font.size': 12,
    'axes.grid': False,
    'xtick.top': False,
    'xtick.bottom': False,
    'axes.prop_cycle': mpl.cycler(color=("#0072B2", "#E69F00", "#009E73", "#D55E00", "#F0E442", "#CC79A7")) 
})

# Set Fig paths
FIGDIR = ''  # Set or leave blank to use default (moonpies/figs)
if not FIGDIR:
    FIGDIR = default_config.Cfg().figspath
FIGDIR = str(Path(FIGDIR).resolve() / "_")[:-1]  # add trailing slash

# Create configuration objects for each mode. 
COI = 'Faustini'
seed = 65103
cfg_mp = default_config.Cfg(mode='moonpies', volc_mode='Head', seed=seed)
cfg_cn = default_config.Cfg(mode='cannon', seed=seed)
mp.clear_cache()

# Random number generator for each run
rng_mp = mp.get_rng(cfg_mp)
rng_cn = mp.get_rng(cfg_cn)

# Time array won't vary between runs
TIME_ARR = mp.get_time_array(cfg_mp)

# Geologic eras
geo_labels_abbr = ['pNe.', 'Ne.', 'Im.', 'Era.', 'Cop.', '-']
geo_labels_abbr_long = ['Pre-Nec.', 'Nec.', 'Imb.', 'Era.', 'Cop.', '-']
geo_labels = ['Pre-Nectarian (>3.95 Ga)', 'Nectarian (3.95-3.85 Ga)', 'Imbrian (3.85-3.2 Ga)', 'Eratosthenian (3.2-1.1 Ga)', 'Copernican (1.1 Ga - pres.)', 'Present']
# geo_ages = [4.26, 3.95, 3.85, 3.2, 1.1, 0]
geo_ages = [4.26, 3.97, 3.83, 3.2, 1.1, 0]  # TODO: remove
# Indices of timesteps of interests 
t_now = [2, 45, 175, len(TIME_ARR) - 1]

# values at those indices
time_val = [round(TIME_ARR[t_now[0]]/1e9, 2), round(TIME_ARR[t_now[1]]/1e9, 2), round(TIME_ARR[t_now[2]]/1e9, 2), round(TIME_ARR[t_now[3]]/1e9, 2)] 


def get_ice_by_module(coi, time_arr, cfg, rng):
    """Returns tuple of all modules present in MoonPies."""
    # Get ice from each source
    volc_ice = mp.get_volcanic_ice(time_arr, cfg)
    solar_wind_ice = mp.get_solar_wind_ice(time_arr, cfg)
    micrometerorite_ice = mp.get_micrometeorite_ice(time_arr, cfg)
    small_impactor_ice = mp.get_small_impactor_ice(time_arr, cfg)
    small_simple_ice = mp.get_small_simple_crater_ice(time_arr, cfg)
    
    # Need to add rng for these ones
    rng = mp.get_rng(cfg)
    large_simple_craters_ice = mp.get_large_simple_crater_ice(time_arr, cfg, rng)
    rng = mp.get_rng(cfg)
    large_complex_craters_ice = mp.get_complex_crater_ice(time_arr, cfg, rng)
    rng = mp.get_rng(cfg)
    basin_ice = mp.get_basin_ice(time_arr, cfg, rng)
    if cfg.mode == 'cannon':
        basin_ice = basin_ice*0

    total = volc_ice + solar_wind_ice + micrometerorite_ice + small_impactor_ice + small_simple_ice + large_simple_craters_ice + large_complex_craters_ice + basin_ice

    # Gardening
    gardening = mp.overturn_depth_time(time_arr, cfg)

    # Ballistic sed
    rng = mp.get_rng(cfg)
    df = mp.get_crater_list(True, cfg, rng)
    bsed_d, bsed_frac = mp.bsed_depth_petro_pieters(time_arr, df, cfg)
    bsed_all = bsed_d*bsed_frac
    bsed = bsed_all[:, np.array(cfg.coldtrap_names) == coi].flatten()
    formation_age = df[df.cname == coi].age.values
    bsed[time_arr > formation_age] = 0
    
    return [total, basin_ice, large_complex_craters_ice, micrometerorite_ice, 
            large_simple_craters_ice, volc_ice, solar_wind_ice, small_impactor_ice, 
            small_simple_ice, gardening, bsed]


def ice_by_era(ice, geo_ages, geo_labels, agg=np.mean, sum_impact_regimes=False):
    d = {}
    for i, (age, label) in enumerate(zip(geo_ages[:-1], geo_labels[:-1])):
        tmin = np.searchsorted(-TIME_ARR, -age*1e9)
        tmax = np.searchsorted(-TIME_ARR, -geo_ages[i+1]*1e9)
        d[label] = np.apply_along_axis(agg, 1, np.array(ice)[:,tmin:tmax])
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
        'Gardening', 
        'Ballistic Sed']
    idf = pd.DataFrame.from_dict(d, orient='index', columns=columns)
    if sum_impact_regimes:
        idf.insert(1,'Impactor ice', idf[['Micrometeorite ice','Small impactor ice','Small simple crater ice','Large simple crater ice','Large complex crater ice']].sum(axis=1))
        idf = idf.drop(['Micrometeorite ice', 'Small impactor ice', 'Small simple crater ice', 'Large simple crater ice', 'Large complex crater ice'], axis=1)
    return idf

ice_mp = get_ice_by_module(COI, TIME_ARR, cfg_mp, rng_mp)
ice_cn = get_ice_by_module(COI, TIME_ARR, cfg_cn, rng_cn)

df_mp = ice_by_era(ice_mp, geo_ages, geo_labels_abbr_long, np.mean, True)
df_cn = ice_by_era(ice_cn,  geo_ages, geo_labels_abbr_long, np.mean, True)
df_mp_max = ice_by_era(ice_mp, geo_ages, geo_labels_abbr_long, np.max, True)
df_cn_max = ice_by_era(ice_cn,  geo_ages, geo_labels_abbr_long, np.max, True)
df_mp_min = ice_by_era(ice_mp, geo_ages, geo_labels_abbr_long, np.min, True)
df_cn_min = ice_by_era(ice_cn,  geo_ages, geo_labels_abbr_long, np.min, True)

data = {'MoonPIES': (df_mp_min, df_mp, df_mp_max), 'Cannon': (df_cn_min, df_cn, df_cn_max)}

# ------------ PLOTTING -----------
n = len(df_mp)
width = 0.9/n
for mode in ('MoonPIES', 'Cannon'):
    df_min, df_mean, df_max = data[mode]
    f, ax =plt.subplots(figsize=(7.5, 4))
    ax.set_yscale('log')

    # Bar charts of ice by era
    for i, (label, col) in enumerate(df_mean.iteritems()):
        if label == 'Gardening' or label == 'Ballistic Sed':
            continue
        x = np.arange(n) - width*(n-1)/2 + (i*width)
        ax.bar(x, col, label=label, width=width, yerr=[col-df_min.iloc[:,i], df_max.iloc[:,i]-col], capsize=4)
    
    # Shade impact gardening and ballistic sed
    x = np.arange(n+1) - 0.5
    x = np.repeat(x, 2)[1:-1]
    for label, c, a, h in zip(('Gardening', 'Ballistic Sed'), ('gray', 'red'), (0.3, 0.2), ('\\', '')):
        y = df_mean[label].values
        y = np.repeat(y, 2)
        ax.fill_between(x, [0]*len(y), y, color=c, alpha=a, label='Max ' + label, hatch=h)
    ax.set_ylim(1e-4, 1e2)
    ax.set_xlim(-0.5, 4.5)
    ax.tick_params(axis='both')
    ax.set_xticklabels([0] + list(df_mean.index))
    ax.set_ylabel('Ice deposited each timestep [m]')

    # Legend (customize order)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 3, 4, 5, 6, 0, 1]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, 
              loc='upper right', prop={'size': 11})
    # ax.set_title(mode + ' Ice Thickness by Module')
    # ax.set_xlabel('Lunar Geologic Era')
    # mp.plot_version(cfg_mp, loc='ul')
    f.savefig(FIGDIR + f'plot_by_mod_{mode}.pdf')

