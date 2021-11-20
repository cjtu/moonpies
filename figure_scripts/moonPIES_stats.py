'''
Script to produce bulk statistics of ensemble runs. Produces boxplot, error bar plot, and violin plot
'''
import os
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from moonpies import moonpies as mp
from moonpies import default_config

# Set Fig paths
FIGDIR = ''  # Set or leave blank to use default (moonpies/figs)
if not FIGDIR:
    FIGDIR = default_config.Cfg().figspath
FIGDIR = str(Path(FIGDIR).resolve() / "_")[:-1]  # add trailing slash

# Set data path
DATADIR = '/home/cjtu/projects/moonpies/data/out/211119_mpies'
if not DATADIR:
    DATADIR = default_config.Cfg().outpath
DATADIR = str(Path(DATADIR).resolve() / "_")[:-1]  # add trailing slash
os.chdir(DATADIR)


cfg= default_config.Cfg()

# Find model run directories
model_run_pths = DATADIR + '/*/'
fn = 'ice_columns_mpies.csv'
dir_list = glob(model_run_pths)

# Original crater order in MoonPIES
col_names = ['time','Haworth','Shoemaker','Faustini','Shackleton','Slater','Amundsen','Cabeus','Sverdrup','de Gerlache',"Idel'son L",'Wiechert J', 'Cabeus B']
# Craters sorted by latitude
col_names_new = ['time', 'Cabeus B', "Idel'son L", 'Amundsen', 'Wiechert J', 'Cabeus','Faustini', 'Haworth', 'Shoemaker', 'Slater', 'Sverdrup', 'de Gerlache', 'Shackleton']


# Initialize array for ice sum for each crater
net_ice = np.zeros((len(dir_list),len(col_names[1:])))

# Initialize array for final stats table for every crater
stats_data = np.zeros((len(col_names[1:]),5))

# Loop through every directory, read in the retained ice .csv and calculate sum of every column in dataframe (every crater)
for i, dir in enumerate(dir_list):
    df = pd.read_csv(dir+fn, names = col_names, header=0)
    df = df[col_names_new]
    net_ice[i,:] = np.sum(df.drop('time', axis=1))

# Loop through every column in summed ice data, pull out summed vals for each crater and do stats on those summed vals.
for i,col in enumerate(col_names[1:]):
    stats_data[i,:] = [np.mean(net_ice[:,i]), np.max(net_ice[:,i]), np.min(net_ice[:,i]), np.std(net_ice[:,i]), np.std(net_ice[:,i])/np.sqrt(len(net_ice[:,i]))]

# Put into dataframe
stat_df = pd.DataFrame(data=stats_data,index=col_names[1:], columns=['mean', 'max', 'min', 'SD', 'SE'])


# ---- BoxPlot ----
fig,ax = plt.subplots()
ax.boxplot(net_ice,whis=2.5)
# Boxplot index starts at 1 not 0
x = [1,2,3,4,5,6,7,8,9,10,11,12]
plt.title(f'Statistics From {len(dir_list)} Ensemble Run')
plt.xticks(x,col_names_new[1:])
plt.xticks(rotation = 45)
plt.ylabel('Ice Thickness Retained [m]')
mp.plot_version(cfg, loc = 'ur')
ax.set_yscale('log')
ax.set_xlim(4.5, 12.5)
# plt.show()
fig.savefig(FIGDIR + 'boxplot_enesemble.png')

# ---- Error bars ----
fig1,ax1 = plt.subplots()
x1 = np.arange(0,len(cfg.coldtrap_names))
ax1.errorbar(x1,stat_df['mean'],yerr=stat_df['SD'], linestyle='None', marker = 'o', capsize=3)
plt.title(f'Mean w/SD From {len(dir_list)} Ensemble Run')
plt.xticks(x1,col_names_new[1:])
plt.xticks(rotation = 45)
plt.ylabel('Ice Thickness Retained [m]')
mp.plot_version(cfg, loc = 'ur')
# plt.show()
fig1.savefig(FIGDIR + 'errorbar_enesemble.png')

# ---- Violin plot ----
fig2,ax2 = plt.subplots()
ax2.violinplot(net_ice,quantiles = list(np.ones((len(x),2))*[.25,.75]))
# Violin plots index starts at 1 not 0
x2 = np.arange(1,len(cfg.coldtrap_names) + 1)
ax2.plot(x2, stat_df['mean'], 'o')
plt.title(f'Statistics From {len(dir_list)} Ensemble Run')
plt.xticks(x2,col_names_new[1:])
plt.xticks(rotation = 45)
plt.ylabel('Ice Thickness Retained [m]')
ax2.set_yscale('log')
mp.plot_version(cfg, loc = 'ur')
# plt.show()
fig2.savefig(FIGDIR + 'violin_plot_ensemble.png')
