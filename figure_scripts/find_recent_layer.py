'''
Script to find the distribution of most recent layer of ice in every crater for all ensemble runs. 
Craters are sorted by latitude on the x axis. Distributions are shown with a violin plot

'''

import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# import sys and use the following command to point to the directory where moonpies.py is (This is not necessary if this script is in that directory)
import sys
sys.path.insert(0, "/Users/tylerpaladino/Documents/ISU/LPI_NASA/Codes/moonpies_package/moonpies")

import moonpies as mp
from moonpies import config


cfg= config.Cfg()

# Craters sorted by latitude
col_names_new = ['time', 'Cabeus B', "Idel'son L", 'Amundsen', 'Wiechert J', 'Cabeus','Faustini', 'Haworth', 'Shoemaker', 'Slater', 'Sverdrup', 'de Gerlache', 'Shackleton']

# Directory where ensemble results exist
out_dir = '/Users/tylerpaladino/Documents/ISU/LPI_NASA/figs/'
data_pth = '/Users/tylerpaladino/Documents/ISU/LPI_NASA/Codes/moonpies_package/data/210804_mpies'
os.chdir(data_pth)



col_names = ['label', 'time', 'ice', 'ejecta', 'depth', 'icepct']

# Find model run directories
model_run_pths = data_pth + '/*/'
dir_list = glob(model_run_pths)

# Initialize depth array
depths = np.zeros((len(dir_list),len(cfg.coldtrap_names)))

# This is real slow with 10,000 runs, sorry  ¯\_(ツ)_/¯
for i, dir in enumerate(dir_list):
    for j, coldtrap in enumerate(col_names_new[1:]):
        fn = coldtrap + '_strat.csv'
        df = pd.read_csv(dir+fn, names=col_names, header=0)
        df_ice = df[df.ice > 0]
        depths[i,j] = df_ice.depth.iloc[-1]



# Plotting
fig,ax = plt.subplots()
ax.violinplot(depths)
# Violin plots index starts at 1 not 0
x = np.arange(1,len(cfg.coldtrap_names) + 1)
plt.title(f'Latest Depth From MoonPIES {len(dir_list)} Ensemble Run')
plt.xticks(x,col_names_new[1:])
plt.xticks(rotation = 45)
plt.ylabel('Ice Depth [m]')
ax.set_ylim(10**-2,10**4)
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_yscale('log')
mp.plot_version(cfg, loc = 'll')
plt.show()
plt.savefig(out_dir + 'depth_moonpies_all_on_semilog.png')